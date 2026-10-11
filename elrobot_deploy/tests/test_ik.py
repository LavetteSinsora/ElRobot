"""P6's IK wrapper against the real mink QP.

These are deliberately few: the QP is the slow part of the whole test suite and
the loop that calls it is tested separately with a fake solver. What is worth
paying for here is only what a fake cannot show — that the seed really is the
seed, that the waist really is pinned where P6 commanded it, and that a solve
which cannot succeed gives up on the clock rather than on an iteration count.

`ik_early_stop` is raised from the config default in most fixtures below, and
`test_the_error_floor_is_the_posture_cost` says why: the QP's steady-state
flange error is set by the posture cost, not by how long it iterates.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.robot import kinematics as kin
from elrobot_deploy.runtime.ik.arm_qp import DualArmSolver


@pytest.fixture(scope="module")
def fk():
    """P1's FK — the same `kin.place()` the solver uses, which is the point."""
    return kin.ForwardKinematics()


@pytest.fixture(scope="module")
def ikcfg():
    return cfgmod.DeployConfig(
        prompt="t", checkpoint="none", sync=True, replan_period_ticks=3,
        action_horizon=8, control_hz=30.0,
        cameras={"rwrist": cfgmod.CameraSpec(8, 8)}, primary_camera="rwrist",
        home_q=np.zeros(14), home_waist=np.zeros(3),
        home_hand=np.zeros((2, 6)), exit_q=np.zeros(14), exit_waist=np.zeros(3),
        exit_hand=np.zeros((2, 6)),
        ik_early_stop=3e-3)


@pytest.fixture(scope="module")
def solver(ikcfg):
    return DualArmSolver(ikcfg)


@pytest.fixture(scope="module")
def kinematic_solver(ikcfg):
    """The same solver with the self-collision limit switched off.

    For the one test whose subject is a pure kinematic identity. The avoidance
    constraint is not a function of the arm chain alone — `Yaw_Link` sits BELOW
    the bow, so bowing the waist really does present the arms a different
    obstacle — and leaving it on would mean asserting a frame identity through a
    term that legitimately is not frame-invariant."""
    return DualArmSolver(dataclasses.replace(ikcfg, ik_collision_min_dist=None))


def arm_q(scale: float) -> np.ndarray:
    """A small, collision-free, in-limits arm posture — shoulder pitch and
    elbow on both sides, which is the direction the model has most room in."""
    q = np.zeros(14)
    q[[0, 3, 7, 10]] = [scale, -scale, -scale, scale]
    return q


# --- reaching ---------------------------------------------------------------


def test_a_reachable_target_is_solved_and_reported_solved(solver, fk, ikcfg):
    target = fk.flange(arm_q(0.25), np.zeros(3)).copy()
    q, err, ok = solver.solve(target, np.zeros(3), np.zeros(14))
    assert ok
    assert err <= ikcfg.ik_early_stop
    assert q.shape == (14,)
    # The returned joints really do put the flange there — the error the solver
    # reports is a distance in the world, not a QP residual.
    achieved = fk.flange(q, np.zeros(3))
    assert np.abs(achieved[:, :3] - target[:, :3]).max() <= ikcfg.ik_early_stop


def test_a_target_the_seed_already_satisfies_costs_no_iterations(solver, fk):
    """The early exit is checked before the first iterate, not after it."""
    seed = arm_q(0.2)
    target = fk.flange(seed, np.zeros(3)).copy()
    q, err, ok = solver.solve(target, np.zeros(3), seed)
    assert ok and err == 0.0
    assert np.allclose(q, seed)


def test_every_solve_is_explicitly_seeded(solver, fk):
    """P6 jumps back to `k_exec + 2` and re-solves ticks it has
    already solved from a *different* predecessor. A warm start alone would
    make the answer depend on the previous call; the seed must erase it."""
    solver.solve(fk.flange(arm_q(0.4), np.zeros(3)).copy(), np.zeros(3), np.zeros(14))
    seed = arm_q(-0.15)
    q, err, ok = solver.solve(fk.flange(seed, np.zeros(3)).copy(), np.zeros(3), seed)
    assert ok and err == 0.0
    assert np.allclose(q, seed)


# --- The waist pin ----------------------------------------------------------


def test_the_waist_holds_the_commanded_angles_through_the_solve(solver, fk):
    """`kin.place()` writes the commanded waist and the
    model's 1e-9 rad/s velocity limit holds it there — the solve must not be
    able to route arm motion through the trunk.

    Indexed through `waist_cols` rather than positionally: the contract's waist
    vector is 3 wide but this robot has two waist motors, so the slot with no
    joint behind it has no qpos address to compare against.
    """
    waist = np.array([0.3, -0.1, 0.15])
    target = fk.flange(arm_q(0.25), waist).copy()
    solver.solve(target, waist, np.zeros(14))
    jmap = kin.joint_addresses(solver.backend)
    held = waist[jmap.waist_cols]
    assert np.allclose(solver.backend.data.qpos[jmap.waist], held)
    assert np.allclose(solver.ik.config.q[jmap.waist], held)


def test_the_arm_solution_is_invariant_to_a_CONSISTENTLY_placed_waist(
        kinematic_solver, fk):
    """The sharpest statement of why the pin matters. A base-frame target
    generated with the waist at `w` and solved with the waist at `w` is the same
    problem in the torso frame as one generated and solved at zero — so it must
    yield the same arm joints. Solving that same target against a waist of zero
    — the historical behaviour — does not, by a third of a radian.

    Asked of `kinematic_solver`, because the identity being claimed is the
    ARM CHAIN's and the self-collision limit is not part of it; see
    `test_the_collision_limit_is_the_one_term_the_waist_moves`.
    """
    solver = kinematic_solver
    waist = np.array([0.3, 0.0, 0.0])
    q_ref, _, ok_ref = solver.solve(fk.flange(arm_q(0.25), np.zeros(3)).copy(),
                                    np.zeros(3), np.zeros(14))
    q_pin, _, ok_pin = solver.solve(fk.flange(arm_q(0.25), waist).copy(),
                                    waist, np.zeros(14))
    q_wrong, _, _ = solver.solve(fk.flange(arm_q(0.25), waist).copy(),
                                 np.zeros(3), np.zeros(14))
    assert ok_ref and ok_pin
    assert np.allclose(q_pin, q_ref, atol=1e-9)
    assert np.abs(q_wrong - q_ref).max() > 0.1


def test_the_collision_limit_is_the_one_term_the_waist_moves(solver, fk):
    """The other half of the test above, and the reason it needs its own solver.

    Everything about the arm is rigid with `torso_link`, so a consistently placed
    waist is a pure change of frame for the tasks and the posture. The avoidance
    constraint is not: `Yaw_Link` and the base are BELOW the bow, and bowing
    swings the arms relative to them for real — the shoulder-to-`Yaw_Link`
    distance moves 80.0 -> 80.6 mm at a 0.3 rad bow. So the solved arm may differ,
    and the assertion worth making is that it differs by an amount consistent
    with a constraint being nudged (~0.05 rad here) rather than by the third of a
    radian that a MIS-placed waist costs — the two were indistinguishable while
    this was folded into one test.
    """
    waist = np.array([0.3, 0.0, 0.0])
    q_ref, _, _ = solver.solve(fk.flange(arm_q(0.25), np.zeros(3)).copy(),
                               np.zeros(3), np.zeros(14))
    q_pin, _, _ = solver.solve(fk.flange(arm_q(0.25), waist).copy(),
                               waist, np.zeros(14))
    q_wrong, _, _ = solver.solve(fk.flange(arm_q(0.25), waist).copy(),
                                 np.zeros(3), np.zeros(14))
    assert np.abs(q_pin - q_ref).max() < 0.1
    assert np.abs(q_wrong - q_ref).max() > 0.1


def test_a_body_target_closes_only_against_the_waist_it_was_solved_at(
        solver, fk, ikcfg):
    """What `body_targets["waist_ik"]` is FOR, stated as the identity it buys.

    A recorded target holds `pose` (the IK's input), `q[:14]` (its output),
    `err` (its residual) and `q[14:17]` (the waist the tick COMMANDED). Those
    four do not close: the arm was solved in the trunk the robot is actually in,
    which on this machine is 0.039 rad away from the one it was told to hold, so
    `FK(q[:14], q[14:17])` misses `pose` by the droop and nothing in the record
    says why. Reading `err` against that gap is how a frame error comes to look
    like an IK failure.

    With the pinned waist stored, the entry closes to the residual it claims —
    and the difference between the two readings is the whole 8-11 mm, not a
    rounding.
    """
    pinned = np.array([0.0, 0.039, 0.0])        # the trunk the arm hangs off
    commanded = np.zeros(3)                     # what goes on the wire
    target = fk.flange(arm_q(0.25), pinned).copy()

    q, err, ok = solver.solve(target, pinned, np.zeros(14))
    assert ok

    closed = np.abs(fk.flange(q, pinned).copy()[:, :3] - target[:, :3]).max()
    assert closed <= max(err, ikcfg.ik_early_stop) + 1e-9

    # The same three numbers read against the commanded trunk instead.
    open_ = np.abs(fk.flange(q, commanded).copy()[:, :3] - target[:, :3]).max()
    assert open_ > 8e-3


# --- giving up --------------------------------------------------------------


def test_an_unreachable_target_is_approximated_and_reported(solver, fk):
    """The IK: `ok` means "there was time", not "this is good".

    A target three metres away is reached as far as the arm goes, published, and
    the residual reported — the same behaviour as the old stack. Quality is not
    a gate here because this QP's error floor makes a quality gate refuse every
    realistic tick (see the floor test below), and a system that publishes
    nothing is a robot that never moves.
    """
    target = fk.flange(arm_q(0.2), np.zeros(3)).copy()
    target[:, 0] += 3.0
    _q, err, ok = solver.solve(target, np.zeros(3), np.zeros(14), budget=1.0)
    assert ok
    assert err > 1.0, "the residual is the honest signal, and it is huge"


def test_the_budget_is_wall_clock_and_max_iters_is_only_a_backstop(ikcfg, fk):
    """With the iteration cap set absurdly high, the only
    thing that can stop an unreachable solve is the clock."""
    solver = DualArmSolver(cfgmod.replace(ikcfg, ik_max_iters=10_000_000))
    target = fk.flange(arm_q(0.2), np.zeros(3)).copy()
    target[:, 0] += 3.0
    t0 = time.monotonic()
    _q, _err, ok = solver.solve(target, np.zeros(3), np.zeros(14), budget=5e-3)
    elapsed = time.monotonic() - t0
    assert not ok
    assert elapsed < 0.2, "the cap, not the clock, ended the solve"


def test_an_exhausted_budget_abandons_before_the_first_iterate(solver, fk):
    """A tick that has already run out of time must not spend more of it."""
    target = fk.flange(arm_q(0.2), np.zeros(3)).copy()
    target[:, 0] += 3.0
    q, err, ok = solver.solve(target, np.zeros(3), arm_q(0.1), budget=-1.0)
    assert not ok and err > 1.0
    assert np.allclose(q, arm_q(0.1)), "the seed comes back untouched"


def test_the_error_floor_is_the_posture_cost_not_the_iteration_count(ikcfg, fk):
    """Documented, not asserted as desirable.

    The QP settles where the frame tasks and the `PostureTask` balance, and that
    fixed point sits ~1e-3 m from the target for an ordinary 0.3 rad step. The
    config default `ik_early_stop = 1e-4` is *below* that floor, so with the
    defaults a solve iterates to `ik_max_iters` and is abandoned no matter how
    much budget it is given. The threshold is a metre-scale tolerance on the
    flange and has to be set above the floor the posture cost implies.
    """
    tight = DualArmSolver(cfgmod.replace(ikcfg, ik_early_stop=1e-4,
                                         ik_max_iters=400))
    target = fk.flange(arm_q(0.3), np.zeros(3)).copy()
    _q, err, ok = tight.solve(target, np.zeros(3), np.zeros(14), budget=10.0)
    assert 1e-4 < err < 1e-2, "the floor is above the config default"
    # ...and precisely because the floor is unreachable, it must NOT gate `ok`:
    # this is the tick that, under a quality gate, would have starved the system.
    assert ok
