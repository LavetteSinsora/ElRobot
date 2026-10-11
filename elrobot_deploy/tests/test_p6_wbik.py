"""P6's waist-mode path — `cfg.ik` (docs/infra_logic/ik_solvers.md).

The shape under test is deliberately small. When a chunk lands P6 asks the
solver for a TRUNK PLAN — one commanded waist per tick, from passes A and B —
and then runs the per-tick loop it has always run, with the planned waist in
place of the measured one. There is no whole-chunk solve, no second write path,
and no pass that can move the trunk after a tick has been published.

So what these tests hold is that the whole-body path is the deployed path plus
one substitution:

* the plan is built once per chunk, never per tick;
* the planned waist is what the arm is solved in AND what is commanded;
* every way the plan can be absent — flag off, short chunk, planner failure —
  falls back to the measured trunk rather than to anything new;
* the epoch is stamped on the FIRST tick, so a chunk's motion starts as soon as
  the plan exists rather than after the whole chunk is solved.

Same fake-solver discipline as `test_p6_target.py`, whose helpers this reuses:
what is under test is the SCHEDULING, not the kinematics. The solver itself is
exercised in `test_wbik.py`, where a wrong joint angle is the point.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.processes.p6_target import Target

from .test_p6_target import FakeClock, FakeSolver, publish  # shared helpers

ARM, BODY = schema.ARM_DOF, schema.BODY_DOF


@pytest.fixture
def clk():
    """Re-declared rather than imported: a fixture is registered by the module
    it is DEFINED in, so importing the function from `test_p6_target` brings the
    helper across and not the fixture. `FakeClock` is what actually matters — an
    epoch stamp is a time, and the only way to assert `deadline = now + 2·dt`
    exactly is to own `now`."""
    return FakeClock()


class FakePlanner(FakeSolver):
    """`FakeSolver` plus `plan_chunk`, which is the whole new interface.

    The plan is `bow = 0.1 * tick index`, so a test can read a published waist
    and say which tick of which chunk it came from.
    """

    def __init__(self, *, plan_fails=False, on_plan=None, **kw):
        super().__init__(**kw)
        self.plans = []
        self.plan_fails = plan_fails
        self.on_plan = on_plan

    def plan_chunk(self, poses, waist_meas, seed_q14, *, budget=None,
                   slow_start=None):
        self.plans.append({"poses": np.array(poses),
                           "waist_meas": np.array(waist_meas),
                           "seed": np.array(seed_q14), "budget": budget,
                           "n": len(poses)})
        if self.on_plan is not None:
            self.on_plan()
        if self.plan_fails:
            raise RuntimeError("planner exploded")
        out = np.tile(np.asarray(waist_meas, np.float64), (len(poses), 1))
        out[:, 1] = 0.1 * np.arange(len(poses))
        return out


@pytest.fixture
def wcfg(cfg):
    # `waist_qp`: the original whole-body path these tests were written for
    # (`ik = "waist"` meant it until the solver set; it now means
    # `waist_planner`).
    return dataclasses.replace(cfg, ik="waist_qp")


def _seeded(rings, clk, k=0, q=None):
    """Homing's `k = 0` written — what gives the first solve a seed."""
    rings["targets"].write(k, clk.t, np.zeros(BODY) if q is None else q, intervene=0)


# --- The plan ---------------------------------------------------------------


def test_the_plan_is_built_once_per_chunk_not_once_per_tick(rings, hw, wcfg,
                                                            clk):
    """The planner is the only whole-chunk work left, and it must stay that
    way: per-tick planning would put pass A and pass B back on the hot path."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner()
    loop = Target(rings, wcfg, hw, solver, now=clk)

    for _ in range(8):
        loop.step()
    assert len(solver.plans) == 1
    assert solver.plans[0]["n"] == 7          # 12..18, the covered ticks
    assert len(solver.calls) > 1              # but many per-tick solves


def test_ticks_stream_one_per_iteration(rings, hw, wcfg, clk):
    """One tick per `step()`, exactly as without the flag. The chunk is NOT
    published in a burst — that is what let the epoch be stamped late."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, wcfg, hw, FakePlanner(), now=clk)

    assert loop.step().k == 12
    assert rings["targets"].get(12) is not None
    assert rings["targets"].get(13) is None, "wrote more than one tick"
    assert loop.step().k == 13
    assert rings["targets"].get(13) is not None


def test_the_epoch_is_stamped_on_the_first_tick(rings, hw, wcfg, clk):
    """The latency claim, pinned.

    The planner costs 400 ms of fake clock here. The first tick's deadline is
    still `2·dt` after the plan returns — not after the whole chunk is solved,
    which is what the four-pass solver used to make it.
    """
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner(on_plan=lambda: clk.advance(0.400))
    loop = Target(rings, wcfg, hw, solver, now=clk)
    loop.step()

    first = rings["targets"].get(12)
    assert first.deadline == pytest.approx(clk.t + 2 * wcfg.dt)


def test_deadlines_stay_one_dt_apart(rings, hw, wcfg, clk):
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, wcfg, hw, FakePlanner(), now=clk)
    # The servo walks the chunk out as P6 solves it. Without this the
    # solve-ahead bound stops P6 at `k_exec + lead`, which is the point of it.
    for _ in range(7):
        loop.step()
        rings["control"].k_exec += 1
    prev = rings["targets"].get(12).deadline
    for k in range(13, 19):
        d = rings["targets"].get(k).deadline
        assert d == pytest.approx(prev + wcfg.dt), k
        prev = d


# --- The planned waist ------------------------------------------------------


def test_the_planned_waist_is_solved_in_and_commanded(rings, hw, wcfg, clk):
    """Both halves, because they are different questions.

    The arm must be solved in the frame the trunk will actually be in, and the
    trunk must be commanded there — a record where those disagree is the exact
    failure `waist_ik` exists to expose.
    """
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner()
    loop = Target(rings, wcfg, hw, solver, now=clk)
    for _ in range(3):
        loop.step()

    for i, k in enumerate((12, 13, 14)):
        assert solver.calls[i]["waist"][1] == pytest.approx(0.1 * i)
        row = rings["targets"].get(k)
        assert row.payload[ARM:][1] == pytest.approx(0.1 * i)


def test_the_lead_bound_stops_the_solver_running_away(rings, hw, wcfg, clk):
    """`solve_lead_ticks`. P6 solves ~8x faster than P7 executes, so unbounded
    it solves the whole chunk against one stale observation of the trunk."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, dataclasses.replace(wcfg, solve_lead_ticks=3), hw,
                  FakePlanner(), now=clk)
    for _ in range(10):
        loop.step()

    # k_exec is 10 and the anchor is 12, so 12 and 13 are reachable; 14 is
    # k_exec + 4 and must wait for the servo.
    assert rings["targets"].get(13) is not None
    assert rings["targets"].get(14) is None
    assert loop.step().reason == "idle"

    rings["control"].k_exec = 11            # the servo advances
    loop.step()
    assert rings["targets"].get(14) is not None


def test_no_lead_bound_solves_the_whole_chunk(rings, hw, wcfg, clk):
    """0 restores the behaviour every run before this had."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, dataclasses.replace(wcfg, solve_lead_ticks=0), hw,
                  FakePlanner(), now=clk)
    for _ in range(8):
        loop.step()
    for k in range(12, 19):
        assert rings["targets"].get(k) is not None, k


def test_the_bound_holds_without_the_flag_too(rings, hw, cfg, clk):
    """It is a property of the LOOP, not of the whole-body path: the deployed
    solver benefits from a fresh measurement for the same reason."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, dataclasses.replace(cfg, solve_lead_ticks=2), hw,
                  FakeSolver(), now=clk)
    for _ in range(6):
        loop.step()
    assert rings["targets"].get(12) is not None
    assert rings["targets"].get(13) is None


def _measured(rings, hw, clk, yaw=0.0, bow=0.0):
    """Put a measured trunk on the state ring."""
    q = np.zeros(schema.N_JOINTS)
    q[np.asarray(hw.joints_waist)] = [yaw, bow, 0.0]
    rings["state"].write(t_recv=clk.t, q=q, dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=clk.t)


def test_the_solve_frame_carries_the_trunk_s_real_position(rings, hw, wcfg,
                                                           clk):
    """The measured error between commanded and achieved trunk must survive
    into the frame the arm is solved in.

    Measured on hardware: the bow settles `21.4 + 32.5*v` mrad short of its
    command, which is 9 mm at the flange at `rate_waist`. Solving at the
    planned angle would discard exactly that, and it is seven times the
    residual the tier exists to remove.
    """
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    _measured(rings, hw, clk, bow=-0.030)     # the trunk is 30 mrad behind
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner()
    loop = Target(rings, wcfg, hw, solver, now=clk)
    for _ in range(3):
        loop.step()

    # plan[i] = 0.1*i, k_exec = 10 is below the chunk so the reference is
    # plan[0] = 0. Each solve therefore sees the real trunk plus the planned
    # increment, NOT the planned angle.
    for i in range(3):
        assert solver.calls[i]["waist"][1] == pytest.approx(-0.030 + 0.1 * i)
    # ...while the COMMAND stays the plan, unshifted.
    for i, k in enumerate((12, 13, 14)):
        assert rings["targets"].get(k).payload[ARM:][1] == \
            pytest.approx(0.1 * i)


def test_a_stationary_plan_solves_exactly_where_waist_now_would(rings, hw,
                                                                wcfg, clk):
    """The reduction that makes this safe: no trunk motion, no difference.

    A flat plan must produce the deployed behaviour exactly, so the whole-body
    path cannot perturb a chunk it had nothing to contribute to.
    """
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    _measured(rings, hw, clk, bow=0.043)
    publish(rings, 1, k_first=11, n=8)

    flat = FakePlanner()
    flat.plan_chunk = lambda poses, w, s, **kw: np.tile(
        np.asarray(w, np.float64), (len(poses), 1))
    loop = Target(rings, wcfg, hw, flat, now=clk)
    loop.step()
    assert flat.calls[0]["waist"][1] == pytest.approx(0.043)


def test_the_reference_follows_the_servo_through_the_chunk(rings, hw, wcfg,
                                                           clk):
    """The increment is from where the servo IS, not from the chunk's start.

    Otherwise the planned motion is added twice: once by the trunk actually
    moving (which the measurement sees) and once by the plan.
    """
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    _measured(rings, hw, clk, bow=0.0)
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, wcfg, hw, FakePlanner(), now=clk)
    loop.step()                                    # writes 12

    # P7 has reached 13 and the trunk has followed the plan to 0.1.
    rings["control"].k_exec = 13
    _measured(rings, hw, clk, bow=0.1)
    frame = loop.solve_frame(15, np.zeros(3), loop._planned_waist(15))
    # plan[15] - plan[13] = 0.3 - 0.1 = 0.2, on top of a measured 0.1
    assert frame[1] == pytest.approx(0.1 + 0.2)


def test_without_the_flag_the_measured_trunk_is_used(rings, hw, cfg, clk):
    """`ik = "arm"` must not reach the planner at all."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner()
    loop = Target(rings, cfg, hw, solver, now=clk)
    assert not loop.wbik
    loop.step()
    assert solver.plans == []
    assert rings["targets"].get(12).payload[ARM:][1] == pytest.approx(0.0)


def test_a_planner_failure_falls_back_to_the_measured_trunk(rings, hw, wcfg,
                                                            clk):
    """A solver that cannot plan one chunk must not take the robot down."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner(plan_fails=True)
    loop = Target(rings, wcfg, hw, solver, now=clk)
    step = loop.step()

    assert step.written, "a failed plan must not lose the tick"
    assert rings["targets"].get(12).payload[ARM:][1] == pytest.approx(0.0)


def test_a_tick_past_the_plan_uses_the_measurement(rings, hw, wcfg, clk):
    """A short chunk plans as far as it goes; ticks past it are not invented."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=5)               # covers 12..15
    loop = Target(rings, wcfg, hw, FakePlanner(), now=clk)
    for _ in range(6):
        loop.step()
    assert rings["targets"].get(15) is not None
    assert rings["targets"].get(16) is None
    assert loop._planned_waist(16) is None


def test_the_plan_is_rebuilt_on_the_next_chunk(rings, hw, wcfg, clk):
    """Each chunk gets its own plan, anchored at its own first tick."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner()
    loop = Target(rings, wcfg, hw, solver, now=clk)
    loop.step()

    rings["control"].k_exec = 20
    publish(rings, 2, k_first=21, n=8)
    loop.step()
    assert len(solver.plans) == 2
    assert loop._plan[0] == 22
    assert loop._planned_waist(22)[1] == pytest.approx(0.0)


# --- The invariants the deployed path already had ---------------------------


def test_the_measured_trunk_is_what_the_plan_starts_from(rings, hw, wcfg, clk):
    """The bow rests on a stop, so a plan that starts anywhere but the
    measurement starts in a frame the robot is not in."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    q = np.zeros(schema.N_JOINTS)
    q[np.asarray(hw.joints_waist)] = [0.0, 0.043, 0.0]
    rings["state"].write(t_recv=clk.t, q=q, dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=clk.t)
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner()
    Target(rings, wcfg, hw, solver, now=clk).step()

    assert solver.plans[0]["waist_meas"][1] == pytest.approx(0.043)


def test_the_seed_is_the_written_target_below_the_anchor(rings, hw, wcfg, clk):
    """The seed chain, unchanged: the plan seeds from the written target, never the
    measurement."""
    _seeded(rings, clk, q=np.full(BODY, 0.25))
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner()
    Target(rings, wcfg, hw, solver, now=clk).step()
    assert solver.plans[0]["seed"] == pytest.approx(np.full(ARM, 0.25))


def test_both_rings_or_neither_for_every_tick(rings, hw, wcfg, clk):
    """The both-or-neither rule, per tick — now a property of the record's single row."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, wcfg, hw, FakePlanner(), now=clk)
    for _ in range(8):
        loop.step()
    for k in range(11, 20):
        b, h = rings["body"].get(k), rings["hand"].get(k)
        assert (b is None) == (h is None), k
        if b is not None:
            assert b.deadline == h.deadline, k


def test_the_write_gate_still_applies(rings, hw, wcfg, clk):
    """A tick P7 is already interpolating is refused."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = FakePlanner(
        on_solve=lambda: setattr(rings["control"], "k_exec", 12))
    loop = Target(rings, wcfg, hw, solver, now=clk)
    step = loop.step()
    assert not step.written and step.reason == "gated"
    assert rings["targets"].get(12) is None


def test_an_abandoned_tick_writes_nothing(rings, hw, wcfg, clk):
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    loop = Target(rings, wcfg, hw, FakePlanner(ok=False), now=clk)
    step = loop.step()
    assert not step.written and step.reason == "abandoned"
    assert rings["targets"].get(12) is None


def test_an_uncovered_anchor_is_a_hole_not_a_crash(rings, hw, wcfg, clk):
    """The new-chunk branch anchors at `k_exec + 2` without testing coverage."""
    _seeded(rings, clk)
    rings["control"].k_exec = 100
    publish(rings, 1, k_first=11, n=8)              # nowhere near 102
    loop = Target(rings, wcfg, hw, FakePlanner(), now=clk)
    step = loop.step()
    assert not step.written and step.reason == "uncovered"


# --- The lookahead rows (`cfg.plan_lookahead_ticks`) -------------------------
#
# The trunk plan is the only place in P6 that benefits from seeing past the
# chunk it is solving: pass A picks a bow for the whole span, and a span that
# stops at the replan boundary lets it commit to a branch the NEXT chunk cannot
# continue (on one recorded rollout that branch pinned L_sh_roll at its +2.25 limit
# and cost 18.8 mm). The rows it looks at are decoded actions the chunk will
# never run, so the two things these tests hold are that the planner sees them
# and that nothing else does.


def publish_with_plan(rings, chunk_id, k_first, n, extra, *, x=0.0, plan_x=9.0):
    """A chunk of `n` actions carrying `extra` resident rows it will not run."""
    def poses(m, value):
        p = np.zeros((m, schema.N_HANDS, schema.POSE_DIM))
        p[..., 0] = value
        p[..., 3] = 1.0                     # identity quaternion, wxyz
        return p
    rings["chunks"].write(
        chunk_id=chunk_id, k_first_action=k_first, n_actions=n, n_plan=n + extra,
        pose=np.concatenate([poses(n, x), poses(extra, plan_x)]),
        hand=np.zeros((n + extra, schema.N_HANDS, schema.HAND_DIM)),
        t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
        t_cpu_processed=0.0, t_chunk_received=0.0, t_chunk_written=0.1)
    rings["control"].newest_chunk_id = chunk_id


def test_the_planner_sees_the_lookahead_rows(rings, hw, wcfg, clk):
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish_with_plan(rings, 1, k_first=11, n=5, extra=3, x=1.0, plan_x=9.0)
    solver = FakePlanner()
    loop = Target(rings, dataclasses.replace(wcfg, plan_lookahead_ticks=3),
                  hw, solver, now=clk)
    loop.step()

    plan = solver.plans[0]
    assert plan["n"] == 4 + 3                    # 12..15 executed, then 16..18
    assert plan["poses"][3][0, 0] == 1.0         # the last EXECUTED row
    assert plan["poses"][4][0, 0] == 9.0         # the first lookahead row


def test_the_rows_are_resident_but_the_knob_is_what_reads_them(rings, hw, wcfg,
                                                               clk):
    """A writer that publishes plan rows must not change P6's behaviour on its
    own — the knob off is the deployed path, unaltered."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish_with_plan(rings, 1, k_first=11, n=5, extra=3, x=1.0, plan_x=9.0)
    solver = FakePlanner()
    loop = Target(rings, wcfg, hw, solver, now=clk)      # plan_lookahead = 0
    loop.step()
    assert solver.plans[0]["n"] == 4


def test_the_lookahead_does_not_buy_the_planner_more_time(rings, hw, wcfg, clk):
    """The budget is wall-clock until the chunk's last EXECUTED tick. Sizing it
    to the extended span would spend time P7 does not have."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish_with_plan(rings, 1, k_first=11, n=5, extra=3)
    solver = FakePlanner()
    loop = Target(rings, dataclasses.replace(wcfg, plan_lookahead_ticks=3),
                  hw, solver, now=clk)
    loop.step()
    assert solver.plans[0]["budget"] == pytest.approx((4 - 1) * wcfg.dt)


def test_no_lookahead_tick_is_ever_published(rings, hw, wcfg, clk):
    """The rows are planning context. If one reached the wire it would be an
    action the policy never scheduled, at a tick the next chunk owns."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish_with_plan(rings, 1, k_first=11, n=5, extra=3)
    loop = Target(rings, dataclasses.replace(wcfg, plan_lookahead_ticks=3),
                  hw, FakePlanner(), now=clk)
    for _ in range(9):
        loop.step()
    assert rings["targets"].get(15) is not None       # the last executed tick
    assert rings["targets"].get(16) is None           # the first planned one


def test_the_planned_waist_still_lines_up_with_the_executed_ticks(rings, hw,
                                                                  wcfg, clk):
    """`FakePlanner` returns `bow = 0.1 * index`, so an off-by-one between the
    extended plan and the ticks it is consumed at is visible as a wrong bow."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish_with_plan(rings, 1, k_first=11, n=5, extra=3)
    loop = Target(rings, dataclasses.replace(wcfg, plan_lookahead_ticks=3),
                  hw, FakePlanner(), now=clk)
    for _ in range(4):
        loop.step()
    for i, k in enumerate(range(12, 16)):
        assert rings["body"].get(k).payload[ARM + 1] == pytest.approx(0.1 * i)



# --- joint mode under a trunk planner ---------------------------


def test_a_joint_chunk_is_not_planned_and_commands_its_own_waist(
        rings, hw, wcfg, clk):
    """A trunk planner needs Cartesian targets to reach; a joint chunk has
    none. So nothing is planned, the IK is not called, and the waist on the
    wire is the chunk's `waist_q` (required in joint mode; zeros here)."""
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8, arm_q=0.05)
    solver = FakePlanner()
    loop = Target(rings, wcfg, hw, solver, now=clk)
    assert loop.step().written
    assert solver.plans == [] and solver.calls == []
    assert loop._plan is None
    assert np.allclose(rings["targets"].get(12).payload[ARM:], 0.0)


def test_a_joint_chunk_with_a_waist_is_not_replanned(rings, hw, wcfg, clk):
    _seeded(rings, clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8, arm_q=0.05, waist=0.03)
    solver = FakePlanner()
    loop = Target(rings, wcfg, hw, solver, now=clk)
    assert loop.step().written
    assert solver.plans == []
    assert np.allclose(rings["targets"].get(12).payload[ARM:], 0.03)
