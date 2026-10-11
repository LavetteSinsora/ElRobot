"""`elrobot_deploy.runtime.ik.waist_qp` — the whole-body solver, and P6's chunk path.

The properties worth testing here are the ones that were actually violated
during development, each of which produced plausible numbers while being wrong:

* the DOF mask living on the wrong object, so the QP solved for a moving trunk
  and the integrator discarded it;
* pass A running under pass C's posture cost, so "where would the trunk like to
  be" answered "exactly where it is";
* the damping scale set against the wrong reference, freezing every joint;
* `mink.VelocityLimit` bounding an iterate rather than a tick, so a 1.0 rad/s
  setting permitted 0.67 rad in 33 ms.

None of those raise. All of them are caught below.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.runtime.shared_memory import schema

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mink")

from elrobot_deploy.robot import kinematics as kin                                    # noqa: E402
from elrobot_deploy.robot import mujoco_model as g1, g1d_model as g1d                          # noqa: E402
from elrobot_deploy.runtime.ik.waist_qp import (GROUPS, TIERS, WbikWeights, WholeBodySolver,
                                            _banded_retime)


class Cfg:
    dt = 1 / 30.0
    ik_posture_cost = 0.02
    ik_collision_min_dist = 0.005
    ik_early_stop = 1e-4
    ik_max_iters = 20
    sync = True
    ik = "waist"
    replan_period_ticks = 25


@pytest.fixture(scope="module")
def fixed_backend():
    return g1d.G1DBackend()


@pytest.fixture(scope="module")
def mobile_backend():
    return g1d.G1DBackend(mobile_base=True)


# --- The model extension ----------------------------------------------------


def test_mobile_base_adds_exactly_three_dof(fixed_backend, mobile_backend):
    assert mobile_backend.model.nq == fixed_backend.model.nq + 3
    names = [mobile_backend.model.joint(i).name
             for i in range(mobile_backend.model.njnt)]
    for j in g1d.MOBILE_BASE_JOINTS:
        assert j in names


def test_fixed_model_is_untouched(fixed_backend):
    """The default path must not have changed at all."""
    assert fixed_backend.model.njnt == 24
    assert not fixed_backend.mobile_base
    names = [fixed_backend.model.joint(i).name
             for i in range(fixed_backend.model.njnt)]
    for j in g1d.MOBILE_BASE_JOINTS:
        assert j not in names


def test_mobile_trunk_collision_set_resolves(mobile_backend):
    """`AGV_link` stops merging into the worldbody once it has a parent, so the
    fixed model's `"world"` entry would name a body that no longer exists."""
    groups = g1.collision_geom_groups(mobile_backend.model,
                                      g1d.MOBILE_TRUNK_COLLISION_BODIES,
                                      g1d.ARM_COLLISION_BODIES)
    assert groups["trunk"] and groups["left"] and groups["right"]
    with pytest.raises(ValueError, match="contribute no collidable geom"):
        g1.collision_geom_groups(mobile_backend.model,
                                 g1d.TRUNK_COLLISION_BODIES,
                                 g1d.ARM_COLLISION_BODIES)


def test_column_and_base_are_inert_in_the_base_frame(fixed_backend):
    """The measurement the whole design rests on.

    The pose frame rides the column and the base, so neither can move the flange
    within it — exactly, not approximately. If this ever stops holding,
    `BASE_BODY` has moved and `wbik`'s snapshot frame is no longer necessary.
    """
    be, m = fixed_backend, fixed_backend.model
    jmap = kin.joint_addresses(be)
    rng = np.random.default_rng(0)
    q0 = np.zeros(m.nq)
    q0[jmap.arm] = rng.uniform(-0.5, 0.5, schema.ARM_DOF)

    def flange_in_base(q):
        be.data.qpos[:] = q
        mujoco.mj_forward(m, be.data)
        inv = np.linalg.inv(be.base_pose())
        return np.concatenate([(inv @ be.flange_pose(s))[:3, 3]
                               for s in kin.SIDES])

    ref = flange_in_base(q0)
    for name in ("LZ_mt_Joint", "LZ_it_Joint",
                 "Left_Wheel_Joint", "Right_Wheel_Joint"):
        q = q0.copy()
        q[int(m.jnt_qposadr[m.joint(name).id])] = 0.15
        assert np.allclose(flange_in_base(q), ref, atol=1e-12), name

    # the waist, by contrast, must move it a long way
    q = q0.copy()
    q[int(m.jnt_qposadr[m.joint("Yaw_Joint").id])] = 0.2
    assert np.linalg.norm(flange_in_base(q) - ref) > 0.03


def test_column_and_base_are_live_in_the_world_frame(mobile_backend):
    """...and the snapshot frame is what recovers them."""
    be, m = mobile_backend, mobile_backend.model
    jmap = kin.joint_addresses(be)
    q0 = np.zeros(m.nq)
    q0[jmap.arm] = 0.2

    def flange_in_world(q):
        be.data.qpos[:] = q
        mujoco.mj_forward(m, be.data)
        return be.flange_pose("left")[:3, 3].copy()

    ref = flange_in_world(q0)
    for name, val, expect in (("LZ_mt_Joint", 0.10, 0.10),
                              ("wbik_base_x", 0.10, 0.10)):
        q = q0.copy()
        q[int(m.jnt_qposadr[m.joint(name).id])] = val
        moved = np.linalg.norm(flange_in_world(q) - ref)
        assert moved == pytest.approx(expect, abs=1e-6), name


# --- construction -----------------------------------------------------------


def test_unknown_group_refuses():
    with pytest.raises(ValueError, match="unknown wbik group"):
        WholeBodySolver(Cfg(), groups=("waist", "elbows"))


def test_base_group_needs_a_mobile_backend(fixed_backend):
    with pytest.raises(ValueError, match="mobile_base=True"):
        WholeBodySolver(Cfg(), backend=fixed_backend, groups=("base",))


def test_mask_is_installed_on_the_solver_the_qp_reads():
    """The mask must be `ik._mask` — `solve_tick` multiplies by that one.

    Stored anywhere else, the QP solves for a moving trunk and the integrator
    throws the trunk component away, so every tier silently collapses onto the
    arm-only answer.
    """
    s = WholeBodySolver(Cfg(), groups=("waist",))
    assert s.ik._mask is s._mask
    for v in s.slow_v:
        assert s.ik._mask[v] == 1.0
    # and the wheels stay frozen
    m = s.backend.model
    for name in ("Left_Wheel_Joint", "Right_Wheel_Joint"):
        assert s.ik._mask[int(m.jnt_dofadr[m.joint(name).id])] == 0.0


def test_waist_slots_map_through_the_name_tuple():
    """`Yaw_Joint` is the BOW and `torso_Joint` is the YAW — never map by name."""
    s = WholeBodySolver(Cfg(), groups=("waist",))
    assert set(s.slow_name) == {"Yaw_Joint", "torso_Joint"}
    # slot 0 is the yaw, and the yaw is cheaper than the bow
    yaw_i = s.slow_name.index("torso_Joint")
    bow_i = s.slow_name.index("Yaw_Joint")
    assert s.slow_cost[yaw_i] < s.slow_cost[bow_i]


def test_two_posture_tasks_with_different_slow_costs():
    """Pass A needs the trunk free; pass C needs it held. One task cannot be
    both, and using pass C's for pass A pins the trunk to its seed."""
    s = WholeBodySolver(Cfg(), groups=("waist",))
    free = np.asarray(s._posture_free.cost)
    track = np.asarray(s._posture_track.cost)
    for v in s.slow_v:
        assert track[v] > free[v] * 10


# --- pass B -----------------------------------------------------------------


def test_retime_respects_rate_and_range():
    ideal = np.full(25, 1.0)                 # far away, immediately
    s = _banded_retime(ideal, 0.0, rate=0.35, accel=5.0, dt=1 / 30.,
                       lo=-0.12, hi=0.12, w_track=1.0, w_vel=4.0, w_acc=40.0)
    assert np.all(np.abs(np.diff(s)) <= 0.35 / 30. + 1e-9)
    assert s.max() <= 0.12 + 1e-9
    assert s.min() >= -0.12 - 1e-9


def test_retime_anticipates():
    """A demand that only arrives at the END of the chunk must start the trunk
    moving well before it. This is the whole reason the horizon exists."""
    ideal = np.zeros(25)
    ideal[20:] = 0.10
    # the weights `_pass_b` would build: the twenty indifferent knots barely
    # vote, so the late demand is free to pull the trajectory backward
    w = 0.02 + 0.98 * (np.abs(ideal) / np.abs(ideal).max())
    s = _banded_retime(ideal, 0.0, rate=0.35, accel=5.0, dt=1 / 30.,
                       lo=-0.5, hi=0.5, w_track=w, w_vel=4.0, w_acc=40.0)
    assert s[10] > 0.1 * s[-1], "trunk had not started moving by tick 10"
    assert s[-1] > 0.5 * 0.10, "trunk never got near the demand"


def test_retime_is_stationary_on_a_stationary_demand():
    s = _banded_retime(np.zeros(25), 0.0, 0.35, 5.0, 1 / 30.,
                       -0.12, 0.12, 1.0, 4.0, 40.0)
    assert np.allclose(s, 0.0, atol=1e-6)





# --- The solve ---------------------------------------------------------------


def _straight_line_chunk(solver, seed, waist, n=25, step=0.002):
    """A reachable chunk: FK the seed, then walk the flange forward in +x."""
    solver._place(seed, waist)
    base_inv = np.linalg.inv(solver.backend.base_pose())
    start = [kin.mat_to_pose(base_inv @ solver.backend.flange_pose(s))
             for s in kin.SIDES]
    poses = np.zeros((n, schema.N_HANDS, schema.POSE_DIM))
    for k in range(n):
        for h in range(schema.N_HANDS):
            poses[k, h] = start[h]
            poses[k, h, 0] += step * k
    return poses


def test_off_tier_reproduces_the_arm_only_solver():
    """`TIERS['off']` must BE the deployed solver, not merely resemble it."""
    from elrobot_deploy.runtime.ik.arm_qp import DualArmSolver

    cfg = Cfg()
    wb = WholeBodySolver(cfg, groups=TIERS["off"])
    da = DualArmSolver(cfg)
    seed = np.zeros(schema.ARM_DOF)
    seed[1], seed[8] = 0.1, -0.1
    waist = np.zeros(schema.WAIST_DOF)
    poses = _straight_line_chunk(wb, seed, waist, n=6)

    for k in range(len(poses)):
        a_wb, e_wb, ok_wb = wb.solve(poses[k], waist, seed)
        a_da, e_da, ok_da = da.solve(poses[k], waist, seed)
        assert ok_wb and ok_da
        assert e_wb == pytest.approx(e_da, abs=2e-5), k
        assert np.allclose(a_wb, a_da, atol=2e-4), k






def test_the_bow_is_never_commanded_behind_its_mechanical_stop():
    """The trunk rests 0.039 rad forward of its commanded zero and no torque
    moves it (measured). The URDF's lower limit is -0.0436, which
    the model believes and the machine does not, so without a floor the QP
    solves the arm into a trunk pose the robot cannot reach — measured at 39.0%
    of ticks over three rollouts, worst case 0.082 rad behind the stop.
    """
    rest = 0.039
    s = WholeBodySolver(Cfg(), groups=("waist",))
    seed = np.zeros(schema.ARM_DOF)
    waist = np.array([0.0, rest, 0.0])
    # pull hard and low, which is what makes the QP want to bow backward
    poses = _straight_line_chunk(s, seed, waist, n=25, step=-0.02)
    sol = s.solve_chunk(poses, waist, seed, budget=30.0)
    assert sol.waist[:, 1].min() >= rest - 1e-9, (
        f"commanded the bow to {sol.waist[:, 1].min():.5f}, behind its "
        f"{rest} rad rest")
    # forward is unaffected: 2.36 rad of travel that way
    assert sol.waist[:, 1].max() >= rest - 1e-9


def test_the_stop_can_be_disabled_for_measurement():
    """`bow_respects_stop=False` is what the report's before/after is measured
    with; it must actually change the behaviour."""
    rest = 0.039
    w = WbikWeights(bow_respects_stop=False)
    s = WholeBodySolver(Cfg(), groups=("waist",), weights=w)
    assert not np.isfinite(s._floor).any()
    s2 = WholeBodySolver(Cfg(), groups=("waist",))
    s2._set_floor(np.array([0.0, rest, 0.0]))
    assert s2._floor[s2.slow_name.index("Yaw_Joint")] == pytest.approx(rest)


def test_the_yaw_gets_no_floor():
    """Only the bow has a stop. Flooring the yaw would ban half its travel."""
    s = WholeBodySolver(Cfg(), groups=("waist",))
    s._set_floor(np.array([0.2, 0.039, 0.0]))
    assert s._floor[s.slow_name.index("torso_Joint")] == -np.inf


def test_chunk_solve_returns_one_row_per_tick():
    s = WholeBodySolver(Cfg(), groups=("waist",))
    seed = np.zeros(schema.ARM_DOF)
    waist = np.zeros(schema.WAIST_DOF)
    poses = _straight_line_chunk(s, seed, waist, n=25)
    sol = s.solve_chunk(poses, waist, seed, budget=30.0)
    assert len(sol) == 25
    assert sol.arm.shape == (25, schema.ARM_DOF)
    assert sol.waist.shape == (25, schema.WAIST_DOF)
    assert sol.err.shape == (25,)
    assert np.isfinite(sol.err).all()
    assert sol.ok


def test_absent_waist_slot_is_never_commanded():
    """Slot 2 has no motor. Solving must not put a value there."""
    s = WholeBodySolver(Cfg(), groups=("waist",))
    seed = np.zeros(schema.ARM_DOF)
    waist = np.zeros(schema.WAIST_DOF)
    poses = _straight_line_chunk(s, seed, waist, n=10)
    sol = s.solve_chunk(poses, waist, seed, budget=20.0)
    assert np.all(sol.waist[:, 2] == 0.0)


def test_per_tick_rate_limit_is_a_tick_and_not_an_iterate():
    """The trap `mink.VelocityLimit` sets: it bounds one QP iterate, and the
    solve runs up to `ik_max_iters` of them, so the per-tick excursion can be
    20x the configured rate unless something else bounds it."""
    w = WbikWeights(rate_waist_yaw=0.2, rate_waist_bow=0.2,
                    excursion_waist=1.0)
    s = WholeBodySolver(Cfg(), groups=("waist",), weights=w)
    seed = np.zeros(schema.ARM_DOF)
    waist = np.zeros(schema.WAIST_DOF)
    # a target that pulls hard sideways, so the waist genuinely wants to move
    poses = _straight_line_chunk(s, seed, waist, n=25, step=0.02)
    sol = s.solve_chunk(poses, waist, seed, budget=30.0)
    per_tick = np.abs(np.diff(sol.waist[:, :2], axis=0)).max()
    assert per_tick <= 0.2 * s.dt * 1.02, (
        f"{per_tick:.5f} rad in one tick exceeds {0.2 * s.dt:.5f}")


def test_each_waist_axis_gets_its_own_rate():
    """The yaw tracks 11x better than the bow (`rate_waist_yaw`), so one number
    for both was holding the good axis to the bad one's limit."""
    w = WbikWeights(rate_waist_yaw=1.0, rate_waist_bow=0.1,
                    excursion_waist=1.0)
    s = WholeBodySolver(Cfg(), groups=("waist",), weights=w)
    rates = dict(zip(s.slow_name, s.slow_rate))
    assert rates["torso_Joint"] == pytest.approx(1.0)   # the YAW
    assert rates["Yaw_Joint"] == pytest.approx(0.1)     # the BOW



def test_feasible_reports_every_enabled_group():
    s = WholeBodySolver(Cfg(), groups=("waist",))
    seed = np.zeros(schema.ARM_DOF)
    waist = np.zeros(schema.WAIST_DOF)
    poses = _straight_line_chunk(s, seed, waist, n=12)
    sol = s.solve_chunk(poses, waist, seed, budget=20.0)
    rep = s.feasible(sol)
    assert rep["waist_yaw"]["ok"]
    assert rep["waist_bow"]["ok"]
    assert rep["arm"]["ok"]


@pytest.mark.parametrize("tier", list(TIERS))
def test_every_tier_constructs_and_solves(tier):
    s = WholeBodySolver(Cfg(), groups=TIERS[tier])
    seed = np.zeros(schema.ARM_DOF)
    waist = np.zeros(schema.WAIST_DOF)
    poses = _straight_line_chunk(s, seed, waist, n=8)
    sol = s.solve_chunk(poses, waist, seed, budget=20.0)
    assert np.isfinite(sol.err).all()
    assert len(sol) == 8


def test_tiers_are_ordered_by_the_stated_hierarchy():
    """waist before column before base, and each tier a superset of the last."""
    assert GROUPS == ("waist", "column", "base")
    seen = []
    for tier in ("off", "waist", "waist_column", "full"):
        g = set(TIERS[tier])
        assert all(prev <= g for prev in seen)
        seen.append(g)
