"""P6's loop, one `step()` at a time.

The solver is fake here on purpose. Every failure this file is looking for is a
scheduling failure — the wrong tick, the wrong epoch, the wrong `k_exec`, a
write that should not have happened — and each of them produces a plausible
joint angle at a wrong time, which is exactly the class of bug that a real QP in
the loop would hide behind its own noise.

The fake returns `seed + f(target)` so that a solution is still traceable to
both of its inputs: a test can tell which seed was used and which chunk won the
blend without knowing anything about kinematics.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.runtime.shared_memory import open_all, schema
from g1d_deploy.runtime.processes.p6_target import OWN_WAIST, Target, run

ARM, BODY = schema.ARM_DOF, schema.BODY_DOF
T0 = 1000.0


class FakeSolver:
    """Identity-ish. `on_solve` is how a test makes P7 move `k_exec` *during*
    the solve, which is the only moment at which the write gate can close."""

    def __init__(self, *, ok=True, on_solve=None, err=0.0):
        self.ok = ok
        self.on_solve = on_solve
        self.err = err
        self.calls = []

    def solve(self, pose, waist, seed_q14, budget=None):
        seed = np.asarray(seed_q14, dtype=np.float64).copy()
        self.calls.append({"pose": np.array(pose), "waist": np.array(waist),
                           "seed": seed, "budget": budget})
        if self.on_solve is not None:
            self.on_solve()
        return seed + float(pose[0, 0]), self.err, self.ok


class FakeClock:
    """`now()` under the test's control: an epoch stamp is a *time*, and the
    only way to assert `deadline = now + 2·dt` exactly is to own `now`."""

    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)
        return self.t


def stage(rings, chunk_id, k_first, n, *, x=0.0, hand=0.0, waist=None,
          arm_q=None):
    """Write a chunk's payload and STOP — the window between payload and id, held open.

    P5 does exactly this and then stores `newest_chunk_id` (p5_policy.py:119
    then :126). Between the two the chunk is coverable and unannounced, and on
    the real robot that gap is the tens of microseconds it takes to memcpy the
    payload — wide enough that P6's idle spin landed in it on every chunk of
    every rollout of 2026-08-20. Holding it open forever is how a test reaches
    a window it could never reliably hit by racing.

    `arm_q` makes it a JOINT chunk (a scalar, broadcast to every joint), and
    only a joint chunk carries `waist` — always, 0 unless given (joint mode
    requires `waist_q`); the eef contract has none.
    """
    if arm_q is not None and waist is None:
        waist = 0.0
    if arm_q is None:
        pose = np.zeros((n, schema.N_HANDS, schema.POSE_DIM))
        pose[..., 0] = x
        pose[..., 3] = 1.0                  # identity quaternion, wxyz
        payload = {"pose": pose}
    else:
        payload = {"arm_q": np.full((n, schema.N_HANDS, 7), arm_q)}
    if waist is not None:
        payload["waist"] = np.full((n, schema.WAIST_DOF), waist)
    rings["chunks"].write(chunk_id=chunk_id, k_first_action=k_first, n_actions=n,
                          hand=np.full((n, schema.N_HANDS, schema.HAND_DIM), hand),
                          **payload,
                          t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
                          t_cpu_processed=0.0, t_chunk_received=0.0,
                          t_chunk_written=0.1)


def publish(rings, chunk_id, k_first, n, *, x=0.0, hand=0.0, waist=None,
            arm_q=None):
    """Write a chunk and make it visible, in that order."""
    stage(rings, chunk_id, k_first, n, x=x, hand=hand, waist=waist, arm_q=arm_q)
    rings["control"].newest_chunk_id = chunk_id


@pytest.fixture
def clk():
    return FakeClock()


@pytest.fixture
def make(rings, cfg, hw, clk):
    """A Target over the shared arena, with homing's `k = 0` already written —
    the startup state, which is what gives the first solve a seed."""
    def _make(*, solver=None, config=None, seed_q=None):
        rings["targets"].write(0, clk.t, np.zeros(BODY) if seed_q is None else seed_q, intervene=0)
        return Target(rings, config or cfg, hw, solver or FakeSolver(), now=clk)
    return _make


# --- step 1: which tick -----------------------------------------------------


def test_a_new_chunk_anchors_the_cursor_at_k_exec_plus_two(rings, make):
    """The `+2` buys a full dt of solve time before the write gate can close on
    the tick, and it is the same `+2` P5's window opens at."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make()
    step = loop.step()
    assert step.k == 12 and step.written


def test_the_cursor_will_not_walk_onto_a_chunk_that_is_not_yet_announced(
        rings, make):
    """The 2026-08-20 regression.

    Only the new-chunk branch re-stamps the epoch, so the advance branch
    must never be able to reach a chunk that branch has not seen. It could:
    coverage goes true when P5's payload lands and the id arrives after it,
    so P6 walked onto each new chunk's first action one iteration early
    and stamped it from the OLD epoch. Held open with `stage`, since the real
    window is microseconds wide.
    """
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)          # covers 11, 12, 13
    loop = make()
    assert loop.step().k == 12
    assert loop.step().k == 13
    assert loop.step().reason == "idle"         # horizon exhausted, cursor at 13

    stage(rings, 2, k_first=14, n=3)            # payload in, id withheld
    assert loop.step().reason == "idle", "walked onto an unannounced chunk"
    assert rings["body"].get(14) is None
    assert loop.k_to_solve == 13                # the cursor did not move

    rings["control"].newest_chunk_id = 2        # now announced
    rings["control"].k_exec = 13
    assert loop.step().k == 15                  # k_exec + 2, never 14


def test_an_unannounced_chunk_cannot_plant_a_stale_epoch_deadline(
        rings, cfg, make, clk):
    """The consequence that reached the record: 36 of 37 chunks had their first
    action written with a deadline ~100 ms in the PAST, because it carried the
    previous chunk's clock. P7 discards past deadlines, so it cost a wasted
    solve and nothing else — but the dropped first action was then holding through P7's stale-slot
    filter rather than through this cursor."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make()
    loop.step(); loop.step()                    # solve 12, 13
    assert loop.step().reason == "idle"
    stale = loop.deadline(14)                   # what chunk 1's clock says for 14

    clk.advance(0.6)                            # 600 ms of inference
    rings["control"].k_exec = 13
    stage(rings, 2, k_first=14, n=3)
    loop.step()                                 # must refuse to touch 14
    rings["control"].newest_chunk_id = 2
    step = loop.step()

    assert rings["body"].get(14) is None, "action 0 must be a hole"
    assert step.k == 15
    assert step.deadline == pytest.approx(clk.t + 2 * cfg.dt)   # the fresh stamp
    # `stale` is the deadline tick 14 would have carried had the cursor walked
    # onto it: chunk 1's epoch, which by now is behind the clock. That sign is
    # the defect — a target published already expired — and it is what the
    # record shows as a −99 ms margin on 36 of 37 chunks.
    assert stale < clk.t


def test_the_cursor_advances_only_onto_a_tick_a_live_chunk_covers(rings, make):
    """The 'does the next tick exist' test is the coverage test
    step 2 performs, narrowed to chunks already adopted — so an iteration that
    advances has already proven its contributors exist."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)          # covers 11, 12, 13
    loop = make()
    assert loop.step().k == 12
    assert loop.step().k == 13
    exhausted = loop.step()
    assert exhausted.k is None and exhausted.reason == "idle"
    assert loop.k_to_solve == 13, "the cursor must not run past the horizon"
    assert rings["body"].get(14) is None


def test_the_horizon_being_exhausted_is_not_a_deadlock(rings, make):
    """Naively tracking `k_exec` would work in async and deadlock in sync, where
    `k` freezes during inference and the tick after the chunk's last one never
    appears. Idling is the correct behaviour, and the next chunk resumes it."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make()
    for _ in range(4):
        loop.step()
    assert loop.step().reason == "idle"
    publish(rings, 2, k_first=14, n=3)
    rings["control"].k_exec = 13
    assert loop.step().k == 15


def test_a_new_chunk_snaps_the_cursor_back_and_re_solves(rings, cfg, make):
    """There is exactly one seed chain, and it holds even when P6
    jumps back to `k_exec + 2` and re-solves a tick from a new predecessor."""
    rings["control"].k_exec = 10
    solver = FakeSolver()
    publish(rings, 1, k_first=11, n=3, x=1.0)
    loop = make(solver=solver, config=cfgmod.replace(cfg, blend="newest"))
    assert loop.step().k == 12
    assert loop.step().k == 13
    publish(rings, 2, k_first=12, n=3, x=2.0)   # k_exec still 10: no progress
    step = loop.step()
    assert step.k == 12 and step.written
    # Re-solved from the tick-11 seed... which does not exist, so from tick 0.
    assert np.allclose(solver.calls[-1]["seed"], 0.0)
    # ... and against the NEW chunk's action, newest-wins over the overlap.
    assert np.allclose(solver.calls[-1]["pose"][:, 0], 2.0)


def test_nothing_is_solved_before_the_first_chunk_is_published(rings, make):
    loop = make()
    step = loop.step()
    assert step.k is None and step.reason == "idle"
    assert loop.k_to_solve is None


def test_a_tick_no_chunk_covers_is_a_hole_not_a_write(rings, make):
    """Async only: P6 can fall far enough behind that the new chunk
    starts past `k_exec + 2`. There is nothing to blend, so nothing is written."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=20, n=3)
    loop = make()
    step = loop.step()
    assert step.k == 12 and not step.written and step.reason == "uncovered"
    assert rings["body"].get(12) is None and rings["hand"].get(12) is None


# --- step 4/5: the seed and the clamp ---------------------------------------


def test_the_seed_is_the_written_target_with_the_largest_k_below(rings, make):
    """Not the measurement: seeding from measurement makes tracking
    error an input to the next target."""
    solver = FakeSolver()
    rings["targets"].write(11, 5.0, np.full(BODY, 0.3), intervene=0)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make(solver=solver)
    loop.step()
    assert np.allclose(solver.calls[-1]["seed"], 0.3)


def test_the_velocity_clamp_spans_the_seeds_tick_distance_not_one_tick(rings, cfg, hw, make):
    """After a hole, or after a new chunk snapped the cursor
    back, the seed can be several ticks away — and the arm has that many ticks
    to cover the delta. Clamping to one tick would quietly slew-limit every
    re-solve to a third of the motion it was asked for."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3, x=1e6)   # the fake will return junk
    loop = make(solver=FakeSolver())
    step = loop.step()
    assert step.k == 12 and step.written
    span = (12 - 0) * cfg.dt                    # seed is homing's k = 0
    written = rings["body"].get(12).payload[:ARM]
    assert np.allclose(written, hw.qdot_max_body[:ARM] * span)
    assert not np.allclose(written, hw.qdot_max_body[:ARM] * cfg.dt)


def test_the_solver_is_given_one_action_tick_of_budget(rings, cfg, make):
    solver = FakeSolver()
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    make(solver=solver).step()
    assert solver.calls[-1]["budget"] == pytest.approx(cfg.dt)


def test_the_ik_residual_is_recorded_with_the_target_it_produced(rings, make):
    """The IK returns `tracking_error` precisely so an unreachable target reads
    as an arm that went as far as it could, rather than as a silently dropped
    tick. P6 discarded it, so the 2026-08-20 rollouts recorded 82 mm of
    under-reach as joint angles that looked perfectly ordinary."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make(solver=FakeSolver(err=0.0826))

    assert loop.step().k == 12

    assert rings["body"]._rec["err"][12 % rings["body"].n] == pytest.approx(0.0826)


def test_a_tick_no_ik_produced_records_undefined_not_zero(rings, cfg):
    """Homing and the exit ramp are written by P0 and have no solve behind
    them. Recording 0.0 would claim perfect tracking for a tick nothing
    tracked."""
    rings["targets"].write(-5, T0, np.zeros(BODY), intervene=0)
    assert np.isnan(rings["body"]._rec["err"][-5 % rings["body"].n])


def test_an_abandoned_solve_writes_nothing(rings, make):
    """Never publish a slot believed unsolved. The tick becomes a
    hole, which P7 skips and P5's window tolerates."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make(solver=FakeSolver(ok=False))
    step = loop.step()
    assert step.k == 12 and not step.written and step.reason == "abandoned"
    assert rings["body"].get(12) is None
    assert rings["hand"].get(12) is None
    assert loop.step().k == 13, "an abandoned tick does not stall the cursor"


# --- step 4: the trunk the IK solves in -------------------------------------


def push_waist(rings, waist3, *, t=T0):
    """One P1-shaped state sample. Only the waist columns matter here."""
    q = np.zeros(schema.N_JOINTS)
    q[[12, 13, 14]] = waist3
    rings["state"].write(t_recv=t, q=q, dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=t)


def test_the_ik_solves_against_the_MEASURED_trunk(rings, make):
    """The whole point. The chunk asks for an upright trunk; the robot's is not.

    On this machine the bow rests against a physical obstruction 0.039 rad
    forward of the zero it is commanded to, permanently — so a solve against the
    commanded waist places the arm in a frame tilted 2.25 deg from the real one,
    and every flange lands 8-11 mm out with nothing reporting it.
    """
    measured = np.array([-0.0013, 0.0385, 0.0])
    push_waist(rings, measured)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakeSolver()
    loop = make(solver=solver)

    assert loop.step().written
    assert np.allclose(solver.calls[-1]["waist"], measured)


def test_the_wire_is_commanded_P6s_OWN_waist_never_the_measurement(rings, make):
    """The measurement is the IK's frame, never the command.

    An eef chunk carries no waist, so under the pure arm IK P6
    commands `OWN_WAIST` — upright. Writing the measurement instead would publish
    it at kp=300 — the one change that must NOT happen here: commanding the
    trunk to where it already is removes the restoring torque holding it,
    leaving nothing pulling it back.
    """
    push_waist(rings, np.array([-0.0013, 0.0385, 0.0]))
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = make()

    step = loop.step()
    assert step.written
    written = rings["body"].get(step.k).payload[schema.ARM_DOF:]
    assert np.array_equal(written, np.zeros(schema.WAIST_DOF))


def test_the_seed_is_untouched_by_the_measured_trunk(rings, make):
    """Changing the IK's FRAME must not change where it is seeded FROM.

    The seed stays the written target below this tick — never the
    measurement, which would make tracking error an input to the next target and
    let it accumulate. The trunk is the opposite case, and the two must not be
    conflated just because both now come from the same process.
    """
    seed_q = np.full(BODY, 0.25)
    push_waist(rings, np.array([0.0, 0.0385, 0.0]))
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakeSolver()
    loop = make(solver=solver, seed_q=seed_q)

    assert loop.step().written
    assert np.array_equal(solver.calls[-1]["seed"], seed_q[:ARM])


def test_an_empty_state_ring_falls_back_to_the_commanded_waist(rings, make):
    """Unreachable in a run — CHECK proves the stream before ACTIVE — but it is
    what lets `step()` be driven without a state writer. The fallback is the
    COMMANDED waist, which for an eef chunk is P6's own upright one."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakeSolver()
    loop = make(solver=solver)

    assert loop.step().written
    assert np.allclose(solver.calls[-1]["waist"], OWN_WAIST)


def test_a_non_finite_measurement_never_reaches_the_solver(rings, make):
    """The one path where a measurement becomes a commanded joint angle.

    A NaN trunk places the whole model at NaN and the QP returns NaN;
    `validate_chunk` is upstream of here, and P6's velocity clamp does not stop
    it either — `np.clip(nan, lo, hi)` is `nan`. Nothing between this line and
    `rt/lowcmd` would catch it.
    """
    push_waist(rings, np.full(schema.WAIST_DOF, np.nan))
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakeSolver()
    loop = make(solver=solver)

    assert loop.step().written
    assert np.isfinite(solver.calls[-1]["waist"]).all()
    assert np.allclose(solver.calls[-1]["waist"], OWN_WAIST)


def test_the_target_records_the_trunk_it_was_solved_in(rings, make):
    """`waist_ik` is the frame, `q[14:17]` is the command, and both are stored.

    They are permanently different on this robot, so a record carrying only the
    second cannot be read at all: `FK(q[:14], q[14:17])` is not the `pose` stored
    beside it, and the 8-11 mm of trunk droop shows up as IK error that no amount
    of solver tuning can remove.
    """
    measured = np.array([-0.0013, 0.0385, 0.0])
    push_waist(rings, measured)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = make()

    step = loop.step()
    assert step.written
    row = rings["body"].full(step.k)
    assert np.allclose(row["waist_ik"], measured)
    assert np.array_equal(row["q"][schema.ARM_DOF:], np.zeros(schema.WAIST_DOF))


def test_the_recorded_trunk_is_the_ONE_the_solver_was_given(rings, make):
    """One read of the state ring, used twice — not two reads a solve apart.

    The record's whole value is that it names the frame the QP was actually
    asked about. A second `waist_now()` at write time would be a different trunk
    with nothing saying so, and the entry would close against a solve that never
    happened.
    """
    push_waist(rings, np.array([-0.0013, 0.0385, 0.0]))
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakeSolver()
    loop = make(solver=solver)

    step = loop.step()
    assert step.written
    # The trunk moves between the solve and the write. The record must still
    # name the one the solver saw.
    push_waist(rings, np.array([0.02, 0.09, 0.0]), t=T0 + 1.0)
    assert np.array_equal(rings["body"].full(step.k)["waist_ik"],
                          solver.calls[-1]["waist"])


def test_a_tick_with_no_ik_behind_it_records_no_trunk(rings, arena, cfg, hw):
    """P0's ramps are joint-space: there is no frame anything was pinned in, so
    the column is NaN rather than an upright trunk that was never solved for.
    Same convention as `err` and `pose`, and for the same reason — a reader can
    tell "P0 wrote this" by looking."""
    body = rings["body"]
    body.write(0, 1.0, np.zeros(schema.BODY_DOF), intervene=0)
    assert np.isnan(body.full(0)["waist_ik"]).all()
    assert np.isnan(body.full(0)["err"])


# --- step 6: the epoch ------------------------------------------------------


def test_sync_restamps_the_epoch_on_every_chunk(rings, cfg, make, clk):
    """Clock time passes during inference while `k` does not, so
    without the re-stamp every deadline after the first chunk is early by the
    accumulated inference latency."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=cfg.P)
    loop = make()
    first = loop.step()
    assert first.deadline == pytest.approx(clk.t + 2 * cfg.dt)

    clk.advance(0.6)                            # 600 ms of inference
    rings["control"].k_exec = 13                # P7 drained the chunk
    publish(rings, 2, k_first=14, n=cfg.P)
    second = loop.step()
    assert second.k == 15
    assert second.deadline == pytest.approx(clk.t + 2 * cfg.dt)
    assert rings["body"].get(15).deadline == pytest.approx(clk.t + 2 * cfg.dt)


def test_sync_stamps_with_the_captured_k_exec_not_a_fresh_read(rings, cfg, make, clk):
    """The epoch, stated as a hard requirement there. If the stamp re-read
    `k_exec` after a solve during which P7 advanced, `deadline(k_captured + 2)`
    would come out *before* the stamp instant."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    control = rings["control"]
    loop = make(solver=FakeSolver(on_solve=lambda: setattr(control, "k_exec", 14)))
    step = loop.step()
    assert step.k == 12 and step.written
    assert step.deadline == pytest.approx(clk.t + 2 * cfg.dt)
    assert loop.k_base == 10


def test_async_stamps_the_epoch_exactly_once(rings, cfg, hw, make, clk):
    """Async has one epoch forever, `k_base = 0` — which is why P5's
    whole window is one epoch for free."""
    acfg = cfgmod.replace(cfg, sync=False)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make(config=acfg)
    t_stamp = clk.t
    first = loop.step()
    assert (loop.k_base, loop.t_base) == (0, t_stamp)
    assert first.deadline == pytest.approx(t_stamp + 12 * cfg.dt)

    clk.advance(0.6)
    rings["control"].k_exec = 13
    publish(rings, 2, k_first=14, n=3)
    second = loop.step()
    assert (loop.k_base, loop.t_base) == (0, t_stamp), "re-stamped in async"
    assert second.deadline == pytest.approx(t_stamp + 15 * cfg.dt)


def test_every_written_deadline_re_derives_from_the_epoch(rings, cfg, make, clk):
    """Re-derive every ring entry from the stamp in force
    when it was written."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make()
    for _ in range(3):
        loop.step()
    for k in (12, 13):
        assert rings["body"].get(k).deadline == pytest.approx(loop.deadline(k))


# --- step 7: the write gate -------------------------------------------------


@pytest.mark.parametrize("bump,gated", [(0, False), (2, True), (3, True), (4, False)])
def test_the_write_gate_refuses_k_exec_and_k_exec_minus_one(rings, make, bump, gated):
    """The write gate. `k_exec` is re-read at write time because the question
    is whether the value can perturb the segment P7 is mid-interpolation on.

    `bump = 4` is the write gate's tail: a tick below `k_exec − 1` passes the gate and
    is written. It is harmless because P7 only ever selects a *future* deadline,
    so the stale slot is never consumed."""
    control = rings["control"]
    control.k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make(solver=FakeSolver(
        on_solve=lambda: setattr(control, "k_exec", 10 + bump)))
    step = loop.step()
    assert step.k == 12
    assert step.written is not gated
    assert (rings["body"].get(12) is None) is gated
    assert (rings["hand"].get(12) is None) is gated


def test_a_gated_tick_still_carries_its_epoch_forward(rings, cfg, make, clk):
    """Step 6 happens before step 7, so the chunk's re-stamp is not lost when
    the gate happens to close on its first tick."""
    control = rings["control"]
    control.k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    loop = make(solver=FakeSolver(on_solve=lambda: setattr(control, "k_exec", 12)))
    assert loop.step().reason == "gated"
    assert (loop.k_base, loop.t_base) == (10, clk.t)


# --- both rings, or neither -------------------------------------------------


def test_both_rings_get_the_same_k_and_the_same_deadline(rings, make):
    """P8 depends on this: it publishes no cursor and
    infers its segment from deadlines alone."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3, hand=0.7)
    loop = make()
    loop.step()
    body, hand = rings["body"].get(12), rings["hand"].get(12)
    assert body is not None and hand is not None
    assert body.k == hand.k == 12
    assert body.deadline == hand.deadline


def test_the_hand_and_the_joints_come_from_the_same_contributor(rings, cfg, make):
    """Same contributors, same weights, same commit point, so a
    crossfade never executes one chunk's grip against another's pose (or, in
    joint mode, another's joints or waist)."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3, x=1.0, hand=0.1)
    publish(rings, 2, k_first=11, n=3, x=2.0, hand=0.9)
    loop = make(config=cfgmod.replace(cfg, blend="newest"))
    loop.step()
    assert np.allclose(rings["hand"].get(12).payload, 0.9)

    rings["control"].k_exec = 20
    publish(rings, 3, k_first=21, n=3, arm_q=0.1, hand=0.1, waist=0.1)
    publish(rings, 4, k_first=21, n=3, arm_q=0.4, hand=0.9, waist=0.3)
    assert loop.step().k == 22
    body = rings["body"].get(22).payload
    assert np.allclose(rings["hand"].get(22).payload, 0.9)
    assert np.allclose(body[:ARM], 0.4) and np.allclose(body[ARM:], 0.3)


# --- The waist is P6's ------------------------------------------


def test_an_eef_chunk_cannot_command_the_waist(rings):
    """The ring refuses it: a waist on an eef chunk would be a trunk command
    nobody is allowed to give."""
    with pytest.raises(ValueError, match="no waist"):
        stage(rings, 1, k_first=11, n=3, waist=0.2)


def test_under_the_arm_ik_the_waist_is_commanded_upright_and_solved_measured(
        rings, make):
    """The eef contract's rule in one tick: waist = 0 on the wire, the arm solved in
    the trunk the robot is actually in."""
    measured = np.array([0.01, 0.04, 0.0])
    push_waist(rings, measured)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3)
    solver = FakeSolver()
    make(solver=solver).step()
    assert np.allclose(solver.calls[-1]["waist"], measured)
    assert np.array_equal(rings["body"].get(12).payload[ARM:], OWN_WAIST)


# --- joint mode: the IK, and only the IK, is skipped -------------


def test_a_joint_chunk_skips_the_ik_and_writes_its_joints(rings, make):
    solver = FakeSolver()
    push_waist(rings, np.array([0.0, 0.04, 0.0]))
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3, arm_q=0.05, hand=2.0)
    loop = make(solver=solver)
    step = loop.step()
    assert step.k == 12 and step.written
    assert solver.calls == [], "joint mode must not call the IK"
    row = rings["body"].full(12)
    assert np.allclose(row["q"][:ARM], 0.05)
    assert np.allclose(row["q"][ARM:], 0.0)                # the chunk's waist_q
    assert np.isnan(row["err"]), "no solve, so no residual"
    assert np.isnan(row["pose"]).all(), "no Cartesian target behind it"
    assert np.allclose(row["waist_ik"], [0.0, 0.04, 0.0])  # the frame, measured
    assert np.allclose(rings["hand"].get(12).payload, 2.0)


def test_a_joint_chunk_commands_its_waist(rings, make):
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3, arm_q=0.05, waist=0.02)
    make().step()
    assert np.allclose(rings["body"].get(12).payload[ARM:], 0.02)


def test_joint_mode_keeps_the_epoch_cursor_and_clamp(rings, cfg, hw, make, clk):
    """Everything but the IK is the eef path: the cursor at `k_exec + 2`, the
    the epoch stamp, and the velocity clamp against the seed — so a joint target
    two ticks from home cannot ask for more than two ticks of motion."""
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=3, arm_q=1e3)
    loop = make()
    step = loop.step()
    assert step.k == 12 and step.written
    assert step.deadline == pytest.approx(clk.t + 2 * cfg.dt)
    written = rings["body"].get(12).payload[:ARM]
    assert np.allclose(written, hw.qdot_max_body[:ARM] * 12 * cfg.dt)
    assert loop.step().k == 13


# --- The process wrapper ----------------------------------------------------


def test_run_attaches_builds_the_real_solver_and_exits_outside_ACTIVE(arena, cfg, hw):
    """The only test that touches the process entry point. It exists because a
    typo in `run()` — a wrong layout, a wrong ring name, a solver constructed
    from the wrong config — is invisible to every test above and fatal to the
    run. P6 exists only during ACTIVE, so a session of anything else is
    the cheapest way to reach the end of the loop."""
    rings = open_all(arena, cfg)
    rings["control"].session = schema.READY
    run(arena.shm.name, cfg, hw)
