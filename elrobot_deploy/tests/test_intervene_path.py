"""The intervention as it runs through the arena.

`test_intervene.py` proves the arithmetic on a laptop. This file proves the
PLUMBING: that the counter reaches P6, P7 and P8; that a takeover stops the arm
without wiping anything; that the operator's rows become ordinary targets on the
ordinary tick line; and that a handback puts the policy back in charge.

The failures being hunted here are all of one kind — a target executed by the
wrong authority, or at the wrong instant — and every one of them produces a
perfectly plausible joint angle. None would show up as an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1d_deploy.runtime.teleop import intervention as iv
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.processes.p6_target import Target
from g1d_deploy.runtime.processes.p7_body import BodyServo
from g1d_deploy.runtime.processes.p8_hand import HandServo
from g1d_deploy.runtime.launcher.startup_checks import _CHECK_WINDOW_S
from g1d_deploy.runtime.processes.p11_pico import PicoLoop
from tests.test_p6_target import (BODY, T0, FakeClock, FakeSolver,
                                            publish)

IDENT = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


# --- helpers ----------------------------------------------------------------


class FakeOut:
    """Records what reached the wire."""

    def __init__(self):
        self.q, self.dq, self.cmd = [], [], []

    def publish_body(self, q, tau, dq):
        self.q.append(np.asarray(q).copy())
        self.dq.append(np.asarray(dq).copy())

    def publish(self, cmd):
        self.cmd.append(np.asarray(cmd).copy())

    def connect(self, *a):
        pass

    def close(self):
        pass


class ZeroComp:
    def tau(self, q):
        return np.zeros(schema.BODY_DOF)


def write_state(rings, *, q=0.0, flange=None, grip=0.0, t=T0):
    """One P1 sample, so `_measured()` and the gravity term have something."""
    if flange is None:
        flange = np.tile(IDENT, (schema.N_HANDS, 1))
        flange[:, 0] = 0.4
    rings["state"].write(
        t_recv=t, q=np.full(schema.N_JOINTS, q), dq=np.zeros(schema.N_JOINTS),
        flange=flange,
        grip_meas=np.full(schema.N_HANDS * schema.HAND_DIM, grip), t_grip_recv=t)


@pytest.fixture
def servo(rings, hw):
    out = FakeOut()
    s = BodyServo(ring=rings["body"], control=rings["control"],
                  state=rings["state"], out=out, gravity=ZeroComp(), hw=hw,
                  q_cmd_last=np.zeros(schema.BODY_DOF), t0=T0)
    return s, out


def drive(servo, t):
    """One servo iteration to completion — the advance branch publishes
    nothing and falls straight into the next."""
    for _ in range(4):
        if servo.step(t):
            return


# --- P7 and P8: the flag filter and the preempt ------------------------------


def test_a_policy_target_is_invisible_once_the_counter_moves(rings, servo):
    """The core of the intervention's takeover: nothing is wiped, the slots simply stop
    belonging to the world P7 is in."""
    s, out = servo
    write_state(rings)
    rings["targets"].write(5, T0 + 0.5, np.full(BODY, 1.0), intervene=0)
    drive(s, T0 + 0.1)
    assert s.t_segment_end == T0 + 0.5, "latched while the counters agreed"

    rings["control"].intervene = 1
    drive(s, T0 + 0.2)
    assert s.stalled, "the target belongs to a world that no longer exists"
    assert rings["body"].next_after(T0 + 0.2, 1) is None


def test_the_preempt_abandons_a_segment_already_in_flight(rings, servo):
    """Without it the advance branch is gated on `t_now > t_segment_end`, so a
    policy target latched microseconds before the trigger is driven to
    completion — up to a whole action tick of policy motion after a human asked
    for it to stop."""
    s, out = servo
    write_state(rings)
    rings["targets"].write(5, T0 + 1.0, np.full(BODY, 1.0), intervene=0)
    # at the SERVO rate: P7's last-defence slew clamp bounds each publish to
    # `qdot_max / body_hz`, so a test that jumped time would measure the clamp
    # rather than the interpolation.
    for i in range(250):
        drive(s, T0 + 0.1 + i * 0.002)
    halfway = out.q[-1].copy()
    assert 0.2 < float(halfway[0]) < 0.9, "genuinely mid-segment"

    rings["control"].intervene = 1
    drive(s, T0 + 0.602)
    assert out.q[-1] == pytest.approx(halfway), "frozen where the preempt found it"
    assert out.dq[-1] == pytest.approx(np.zeros(BODY)), "and told to expect nothing"


def test_the_preempt_does_not_jump_to_the_target_it_is_cancelling(rings, servo):
    """`stalled = True` is the load-bearing line. Without it the advance branch
    takes its else-clause, sets `q_segment_start = q_segment_end` — the very
    target being cancelled — and the command JUMPS there before interpolating on
    from it."""
    s, out = servo
    write_state(rings)
    rings["targets"].write(5, T0 + 1.0, np.full(BODY, 10.0), intervene=0)
    drive(s, T0 + 0.1)
    rings["control"].intervene = 1
    for i in range(20):
        drive(s, T0 + 0.2 + i * 0.01)
    assert np.max(np.abs(out.q[-1])) < 0.2, "never went anywhere near the target"


def test_a_stream_pause_cannot_resurrect_an_orphaned_policy_target(rings, servo):
    """THE FAILURE THE FILTER EXISTS FOR.

    At a takeover the ring holds up to P future-dated policy targets — ~830 ms
    of committed motion. Operator rows overwrite those slots one at a time as
    they arrive, and while they keep arriving the smallest future deadline is
    always one of theirs. The moment the stream pauses, the un-overwritten
    policy target at the FAR END of the horizon becomes the smallest future
    deadline, and without the filter the arm lurches to a pose nobody asked for.
    """
    s, out = servo
    write_state(rings)
    for k in range(2, 27):                   # a whole policy horizon, ~830 ms
        rings["targets"].write(k, T0 + k * 0.033, np.full(BODY, 5.0), intervene=0)
    rings["control"].intervene = 1
    # the operator drives for a moment, overwriting the near slots only
    for i, k in enumerate(range(4, 8)):
        rings["targets"].write(k, T0 + 0.10 + i * 0.033, np.full(BODY, 0.1),
                            intervene=1)
    for i in range(12):
        drive(s, T0 + 0.09 + i * 0.02)
    # ... and then the headset goes quiet. Far-future policy targets remain.
    assert rings["body"].get(26).deadline > T0 + 0.8
    for i in range(20):
        drive(s, T0 + 0.35 + i * 0.02)
    assert s.stalled, "held, rather than reaching for the orphan at k=26"
    assert np.max(np.abs(out.q[-1])) < 0.2


def test_p8_filters_and_preempts_for_the_same_reason_p7_does(rings, hw):
    """The two rings carry the same `k` and the same deadline, so if they
    disagreed about which side of a takeover a slot belongs to, the jaw would go
    on executing the policy's aperture while the arm followed the operator —
    which is precisely when a gripped object is dropped."""
    out = FakeOut()
    s = HandServo(ring=rings["hand"], out=out, hw=hw,
                  cmd_last=np.zeros((schema.N_HANDS, schema.HAND_DIM)), t0=T0,
                  control=rings["control"])
    cmd = np.zeros((schema.N_HANDS, schema.HAND_DIM))
    cmd[:, 0] = 4.0
    rings["targets"].write(5, T0 + 1.0, np.zeros(schema.BODY_DOF), cmd, intervene=0)
    drive(s, T0 + 0.1)
    drive(s, T0 + 0.5)
    halfway = out.cmd[-1].copy()

    rings["control"].intervene = 1
    for i in range(10):
        drive(s, T0 + 0.6 + i * 0.05)
    assert out.cmd[-1] == pytest.approx(halfway), "the jaw held where it was"


def test_the_exit_ramp_still_executes_after_an_intervention(rings, servo):
    """P0's ramps stamp the LIVE counter, not a zero. "Outside an intervention"
    is an EVEN counter, not necessarily a zero one — after one takeover it is 2,
    and a ramp stamped 0 would be filtered out and the arm would never come
    down."""
    s, out = servo
    write_state(rings)
    rings["control"].intervene = 2
    rings["targets"].write(40, T0 + 0.5, np.full(BODY, 0.3), intervene=2)
    drive(s, T0 + 0.1)
    assert not s.stalled and s.t_segment_end == T0 + 0.5


# --- P6: the guided branch ---------------------------------------------------


@pytest.fixture
def target(rings, cfg, hw):
    clk = FakeClock()
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    loop = Target(rings, cfg, hw, FakeSolver(), now=clk)
    return loop, clk


def guide(rings, counter, deadline, *, x=0.0, jaw=0.0, waist=0.0):
    pose = np.tile(IDENT, (schema.N_HANDS, 1))
    pose[:, 0] = x
    hand = np.zeros((schema.N_HANDS, schema.HAND_DIM))
    hand[:, 0] = jaw
    return rings["intervene"].write(
        intervene=counter, deadline=deadline, pose=pose, hand=hand,
        waist=np.full(schema.WAIST_DOF, waist),
        engaged=np.ones(schema.N_HANDS, dtype=np.uint64),
        trigger=np.ones(schema.N_HANDS), axis_y=np.zeros(schema.N_HANDS),
        direction=np.zeros(schema.N_HANDS, dtype=np.int64),
        t_device=0.0, t_recv=deadline)


def test_the_operators_row_becomes_an_ordinary_target(rings, target, cfg):
    loop, clk = target
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    guide(rings, 1, T0 + 0.5, x=0.7, jaw=2.0, waist=0.1)
    step = loop.step()
    assert (step.k, step.written, step.deadline) == (12, True, T0 + 0.5)
    body, hand = rings["body"].get(12), rings["hand"].get(12)
    assert body.deadline == hand.deadline == T0 + 0.5, "both rings, one deadline"
    assert hand.payload[0, 0] == 2.0
    assert body.payload[schema.ARM_DOF:] == pytest.approx(np.full(3, 0.1))


def test_the_deadline_comes_from_the_operator_and_not_from_the_epoch(
        rings, target):
    """An operator's instant is already an instant; there is no epoch behind
    it, and deriving one would replay their motion at the tick rate instead of
    the rate they made it."""
    loop, clk = target
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    guide(rings, 1, T0 + 12.345)
    assert loop.step().deadline == T0 + 12.345
    assert loop.t_base is None, "no epoch was stamped"


def test_rows_are_decimated_to_the_action_rate(rings, target, cfg):
    """P11 publishes at ~90 Hz. One target per row would run `k` at three times
    the action rate, and `k` is not a free counter — the epoch is affine in it,
    P5 extrapolates off that, and `(k - seed.k)*dt` in the velocity clamp would
    licence three times the joint speed the policy path allows."""
    loop, clk = target
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    written, seen = [], []
    for i in range(30):                      # 30 rows at 11 ms = 330 ms
        guide(rings, 1, T0 + 0.05 + i * 0.011, x=i * 0.01)
        step = loop.step()
        seen.append(step.reason)
        if step.written:
            written.append((step.k, step.deadline))

    assert seen.count("decimated") > 0
    ks = [k for k, _ in written]
    assert ks == list(range(12, 12 + len(ks))), "k advanced one per committed row"
    gaps = np.diff([d for _, d in written])
    # A row period is the granularity available, so no gap is nearer `dt` than
    # that — but the grid reference bounds how far EITHER way a single one goes.
    assert np.all(gaps >= cfg.dt - 0.011 - 1e-9)
    assert np.all(gaps <= cfg.dt + 0.011 + 1e-9)
    # Far fewer targets than rows: the operator is resampled onto the action
    # line, which is the whole point. One row in three, since the rows are three
    # to an action period.
    assert len(written) == pytest.approx(10, abs=1)


def test_the_decimation_does_not_drift_off_the_action_rate(rings, target, cfg):
    """The 90 Hz case, which is the one the robot runs: three PICO frames is
    33.333 ms and `dt` is 33.333 ms, so the boundary test lands exactly on the
    knife edge.

    Referencing the last row PUBLISHED rather than the grid, this dithered
    between taking the third row and the fourth and carried each slip forward —
    measured at 22.9-24.2 Hz on every intervention of one recorded rollout, with
    segments alternating 33.4 and 44.8 ms. P7 turns segment length straight into
    its velocity feedforward, so that is a 25% judder at every step and not
    merely a slower arm.

    The jitter is what makes the failure reachable, and it is the device's own:
    a PERFECT 90 Hz grid clears `last + dt` on the third row every time and even
    the row reference holds 30 Hz. The measured stream does not — device gaps
    ran a median 11.11 ms with a p95 of 11.17 — so the third row falls a
    microsecond short about as often as not, and the reference never recovers
    the slot it then skips.
    """
    loop, clk = target
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    period = 1.0 / 90.0
    rng = np.random.default_rng(0)
    t, written = T0 + 0.05, []
    for i in range(270):                     # 3 s of PICO at 90 Hz
        guide(rings, 1, t, x=i * 0.001)
        t += period + float(rng.normal(0.0, 4e-5))
        step = loop.step()
        if step.written:
            written.append(step.deadline)

    d = np.asarray(written)
    rate = (len(d) - 1) / (d[-1] - d[0])
    assert rate == pytest.approx(1.0 / cfg.dt, abs=0.2), "30 Hz, not 23"
    gaps = np.diff(d)
    assert np.all(gaps == pytest.approx(cfg.dt, abs=1e-3)), "and no alternation"
    # The published deadlines stay on the grid they started on: the reference
    # advancing by `dt` is what makes an error impossible to accumulate.
    assert d[-1] - d[0] == pytest.approx((len(d) - 1) * cfg.dt, abs=1e-3)


def test_a_paused_guide_stream_does_not_fire_the_backlog_it_appears_to_owe(
        rings, target, cfg):
    """A grid reference is only meaningful while the stream is running. After a
    gap of many action periods the grid owes a burst of targets that nothing
    asked for; the operator would get them back-to-back at the loop rate, all in
    the past. It re-founds on the row actually published instead."""
    loop, clk = target
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    guide(rings, 1, T0 + 0.05)
    assert loop.step().written

    guide(rings, 1, T0 + 1.05)               # a second of nothing
    assert loop.step().written
    # The very next row is one PICO frame later — still decimated away, which it
    # would not be if the grid had kept counting through the pause.
    guide(rings, 1, T0 + 1.05 + 1.0 / 90.0)
    step = loop.step()
    assert (step.written, step.reason) == (False, "decimated")


def test_the_first_operator_solve_is_seeded_from_the_policys_last_posture(
        rings, cfg, hw):
    """Nothing is wiped at a takeover, so `latest_below` still finds it — which
    is why this path needs no fallback seed, and why the IK's null space does not
    jump at the seam."""
    clk = FakeClock()
    solver = FakeSolver()
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    loop = Target(rings, cfg, hw, solver, now=clk)
    rings["control"].k_exec = 10
    policy_posture = np.full(BODY, 0.42)
    rings["targets"].write(11, T0 + 0.1, policy_posture, intervene=0)

    rings["control"].intervene = 1
    guide(rings, 1, T0 + 0.5)
    loop.step()
    assert solver.calls[-1]["seed"] == pytest.approx(
        policy_posture[:schema.ARM_DOF])


def test_a_takeover_wipes_nothing(rings, target):
    """The economy of stamping: the slots become unreachable rather than absent,
    so there is no window in which the ring looks empty to a live P7 — and the
    seed above survives."""
    loop, clk = target
    rings["control"].k_exec = 10
    rings["targets"].write(11, T0 + 0.1, np.full(BODY, 0.42), intervene=0)
    rings["control"].intervene = 1
    guide(rings, 1, T0 + 0.5)
    loop.step()
    assert rings["body"].get(11) is not None
    assert rings["body"].get(11).payload == pytest.approx(np.full(BODY, 0.42))


def test_a_row_already_solved_is_not_solved_again(rings, target):
    """Every row is an absolute pose from a fixed anchor, so re-writing one would
    claim the operator held still for a sample they moved through."""
    loop, clk = target
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    guide(rings, 1, T0 + 0.5)
    assert loop.step().written
    assert loop.step().reason == "idle"
    assert loop.step().reason == "idle"


def test_the_velocity_clamp_applies_to_an_operators_target_too(rings, cfg, hw):
    """`_commit` is shared verbatim, so an operator's target is bounded by the
    same ceiling a policy target is — with no second opinion anywhere."""
    clk = FakeClock()
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    loop = Target(rings, cfg, hw, FakeSolver(), now=clk)
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    guide(rings, 1, T0 + 0.5, x=1000.0)       # the fake solver adds pose[0,0]
    step = loop.step()
    q = rings["body"].get(step.k).payload[:schema.ARM_DOF]
    span = (step.k - 0) * cfg.dt
    assert np.all(np.abs(q) <= hw.qdot_max_body[:schema.ARM_DOF] * span + 1e-9)


def test_the_write_gate_still_guards_the_tick_p7_is_on(rings, cfg, hw):
    loop_solver = FakeSolver()
    clk = FakeClock()
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    loop = Target(rings, cfg, hw, loop_solver, now=clk)
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    loop_solver.on_solve = lambda: setattr(
        type(rings["control"]), "_", None) or rings["control"].__setattr__(
            "k_exec", 12)
    guide(rings, 1, T0 + 0.5)
    assert loop.step().reason == "gated"


def test_the_handback_drops_the_cursor_so_the_policy_re_anchors(rings, target):
    """The policy path can only resume through its new-chunk branch — the one
    that re-anchors on `k_exec` and re-stamps the epoch — and never by walking a
    cursor from the world it just left."""
    loop, clk = target
    rings["control"].k_exec = 10
    rings["control"].intervene = 1
    guide(rings, 1, T0 + 0.5)
    loop.step()
    assert loop.k_to_solve == 12

    rings["control"].intervene = 2
    rings["control"].k_exec = 30
    publish(rings, 1, k_first=31, n=5)
    step = loop.step()
    assert step.k == 32, "k_exec + 2, exactly as an ordinary first chunk"


def test_async_re_founds_its_epoch_after_a_handback(rings, cfg, hw):
    """The bug the intervention exposed. Async stamps its epoch once ever, which is right
    while `k` and the clock advance together; an intervention breaks that, and
    every policy deadline afterwards lands in the PAST — P7 discards all of them
    and the arm holds forever, silently, in async only."""
    from g1d_deploy.config_loading import resolved as cfgmod
    acfg = cfgmod.replace(cfg, sync=False)
    clk = FakeClock()
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    loop = Target(rings, acfg, hw, FakeSolver(), now=clk)

    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=4)
    loop.step()
    assert (loop.k_base, loop.t_base) == (0, T0), "one epoch, k_base = 0"

    # a long intervention: P6 keeps writing targets, one per action tick, so the
    # ring stays populated and `k` keeps advancing — which is what makes the
    # policy's seed still reachable on the way back.
    rings["control"].intervene = 1
    for i in range(20):
        clk.advance(cfg.dt)
        guide(rings, 1, clk.t + 0.05, x=0.01)
        loop.step()
    clk.advance(1.0)
    rings["control"].intervene = 2
    rings["control"].k_exec = loop.k_to_solve
    publish(rings, 2, k_first=rings["control"].k_exec + 1, n=4)
    step = loop.step()
    assert step.written, step.reason
    assert step.deadline == pytest.approx(clk.t + 2 * cfg.dt), \
        "re-founded on the live cursor, not extrapolated from a dead one"


def test_an_ordinary_async_run_still_stamps_its_epoch_exactly_once(
        rings, cfg, hw):
    """The re-founding must fire at a handback and NOWHERE else: async chunks
    overlap by design and their tick line has to stay continuous, which is what
    the window extrapolation relies on."""
    from g1d_deploy.config_loading import resolved as cfgmod
    acfg = cfgmod.replace(cfg, sync=False)
    clk = FakeClock()
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    loop = Target(rings, acfg, hw, FakeSolver(), now=clk)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=4)
    loop.step()
    clk.advance(0.6)
    rings["control"].k_exec = 13
    publish(rings, 2, k_first=14, n=4)
    loop.step()
    assert (loop.k_base, loop.t_base) == (0, T0), "re-stamped in async"


# --- P11 ---------------------------------------------------------------------


class FakeSource:
    """A scripted PICO. `feed` is what the next `read()` returns."""

    def __init__(self):
        self.sample = None

    def feed(self, *, t_device, left=None, right=None, trigger=(0.0, 0.0),
             axis=(0.0, 0.0)):
        def sdk(p):
            p = IDENT if p is None else np.asarray(p, dtype=np.float64)
            return [p[0], p[1], p[2], p[4], p[5], p[6], p[3]]   # wxyz -> xyzw
        self.sample = {"t_device": t_device, "pose": (sdk(left), sdk(right)),
                       "trigger": trigger, "axis": axis}

    def read(self):
        s, self.sample = self.sample, None
        return s

    def open(self):
        pass

    def close(self):
        pass


@pytest.fixture
def pico(rings, cfg, hw):
    src = FakeSource()
    write_state(rings)
    return PicoLoop(rings, cfg, hw, src), src, rings


def test_a_squeeze_opens_an_intervention_and_the_row_lands_first(pico):
    """The publish order: `control.intervene` is every reader's only signal that the
    world changed, so the row a reader switches onto must already be readable."""
    loop, src, rings = pico
    src.feed(t_device=0.0, trigger=(0.0, 0.0))
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    assert loop.step(T0 + 0.011) == 1
    assert rings["control"].intervene == 1
    row = rings["intervene"].newest(1)
    assert row is not None and row.index == 1
    assert list(rings["intervene"].meta()["engaged"]) == [1, 0]


def test_the_arm_that_is_not_engaged_holds_where_it_was(pico):
    """What makes single-arm correction work: grab the left, fix what went
    wrong, let go, and the right arm never moved."""
    loop, src, rings = pico
    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    loop.step(T0 + 0.011)
    held = rings["intervene"].newest(1).pose[1].copy()
    moved = IDENT.copy()
    moved[0] = 5.0
    for i in range(2, 8):
        src.feed(t_device=i * 0.011, left=moved, trigger=(0.9, 0.0))
        loop.step(T0 + i * 0.011)
    row = rings["intervene"].newest(1)
    assert row.pose[1] == pytest.approx(held), "the right arm never moved"
    assert row.pose[0][0] != pytest.approx(held[0]), "the left one did"


def test_letting_go_of_the_last_trigger_hands_back(pico):
    loop, src, rings = pico
    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.9))
    loop.step(T0 + 0.011)
    src.feed(t_device=0.022, trigger=(0.0, 0.9))
    assert loop.step(T0 + 0.022) == 0, "one hand still driving"
    src.feed(t_device=0.033, trigger=(0.0, 0.0))
    assert loop.step(T0 + 0.033) == -1
    assert rings["control"].intervene == 2, "even again"


def test_a_silent_stream_holds_and_then_hands_back(pico, cfg):
    """Until the budget runs out the arm HOLDS — P11 writes no rows, P6 has
    nothing new, P7 stalls — which is the correct response and costs no code.
    Past it the run goes back to the policy deliberately, rather than waiting on
    a headset that is not coming back."""
    loop, src, rings = pico
    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    loop.step(T0 + 0.011)
    before = rings["intervene"].meta()["newest"]

    t = T0 + 0.011
    for _ in range(200):                      # nothing fed: every read is None
        t += 0.01
        if loop.step(t) == -1:
            break
    assert rings["control"].intervene == 2
    assert t - (T0 + 0.011) == pytest.approx(cfg.intervene_stale_budget_s,
                                             abs=0.02)
    assert rings["intervene"].meta()["newest"] == before, "no rows while silent"


def test_a_poll_that_finds_nothing_new_is_not_a_dead_stream(pico):
    """This polls at about twice the publish rate, so roughly half of all
    iterations see the same device timestamp. Treating that as a dead stream is
    the single mistake that would make every takeover release itself on the
    following poll."""
    loop, src, rings = pico
    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    loop.step(T0 + 0.011)
    for i in range(10):                       # ten empty polls, 1 ms apart
        assert loop.step(T0 + 0.012 + i * 0.001) == 0
    assert rings["control"].intervene == 1, "still driving"


def test_the_session_gate_stops_a_trigger_from_doing_anything(pico):
    """A squeeze during HOME or READY must not take over — but the poll still
    runs, because that is the window CHECK measures the rate in."""
    loop, src, rings = pico
    src.feed(t_device=0.0, trigger=(1.0, 1.0))
    loop.step(T0, active=False)
    src.feed(t_device=0.011, trigger=(1.0, 1.0))
    assert loop.step(T0 + 0.011, active=False) == 0
    assert rings["control"].intervene == 0
    assert int(rings["intervene"].meta()["samples"]) == 2, "polls still counted"


def test_the_jaw_anchors_on_the_commanded_aperture_not_the_measured_one(
        rings, cfg, hw):
    """0 rad is fully closed, so a jaw squeezing an object sits ABOVE its
    command and the difference IS the grip force. Anchoring on the measurement
    gives that up, and the robot drops what it was holding at the very moment a
    human took over to save it."""
    src = FakeSource()
    write_state(rings, grip=1.4)              # the jaw is held open by an object
    cmd = np.zeros((schema.N_HANDS, schema.HAND_DIM))
    cmd[:, 0] = 1.0                           # ... while being commanded shut
    rings["targets"].write(0, T0, np.zeros(schema.BODY_DOF), cmd, intervene=0)
    rings["control"].k_exec = 0
    loop = PicoLoop(rings, cfg, hw, src)

    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    loop.step(T0 + 0.011)
    assert rings["intervene"].newest(1).hand[0, 0] == pytest.approx(1.0)


def test_the_arm_anchors_on_the_measurement(rings, cfg, hw):
    """The deliberate asymmetry against the jaw: the arm's anchor is where it
    actually is, which costs one small step down at the takeover and buys an
    anchor that is unambiguous."""
    src = FakeSource()
    flange = np.tile(IDENT, (schema.N_HANDS, 1))
    flange[:, 0] = 0.37
    write_state(rings, flange=flange)
    loop = PicoLoop(rings, cfg, hw, src)
    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    loop.step(T0 + 0.011)
    assert rings["intervene"].newest(1).pose[0][0] == pytest.approx(0.37)


def test_a_takeover_that_cannot_read_the_arms_is_refused(rings, cfg, hw):
    """Refused rather than guessed at, and refusing leaves the console disarmed
    so the operator's squeeze does nothing visible and they squeeze again."""
    src = FakeSource()
    loop = PicoLoop(rings, cfg, hw, src)      # no state written at all
    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    assert loop.step(T0 + 0.011) == 0
    assert rings["control"].intervene == 0


def test_release_leaves_the_counter_even(pico):
    """What lets it stay monotonic and single-writer for the life of the
    session: a P0 that stored a zero would be a second writer of a word a live
    P11 also writes."""
    loop, src, rings = pico
    src.feed(t_device=0.0)
    loop.step(T0)
    src.feed(t_device=0.011, trigger=(0.9, 0.0))
    loop.step(T0 + 0.011)
    assert rings["control"].intervene == 1
    loop.release()
    assert rings["control"].intervene == 2
    loop.release()
    assert rings["control"].intervene == 2, "idempotent"


# --- end to end --------------------------------------------------------------


def test_policy_then_operator_then_policy(rings, cfg, hw):
    """The whole shape of the intervention in one test: the policy drives, a human takes over
    and the arm follows THEM, and the policy gets it back."""
    clk = FakeClock()
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    p6 = Target(rings, cfg, hw, FakeSolver(), now=clk)
    out = FakeOut()
    p7 = BodyServo(ring=rings["body"], control=rings["control"],
                   state=rings["state"], out=out, gravity=ZeroComp(), hw=hw,
                   q_cmd_last=np.zeros(schema.BODY_DOF), t0=T0)
    write_state(rings)

    publish(rings, 1, k_first=1, n=cfg.H, x=0.05)
    for _ in range(6):
        p6.step()
    assert rings["body"].full(2) is not None
    assert rings["body"].full(2)["intervene"] == 0

    # the human takes over
    rings["control"].intervene = 1
    drive(p7, clk.t + 0.001)
    assert p7.stalled, "the arm stopped the instant the counter moved"

    for i in range(6):
        guide(rings, 1, clk.t + 0.05 + i * cfg.dt, x=0.9)
        p6.step()
    k = p6.k_to_solve
    assert rings["body"].full(k)["intervene"] == 1
    assert rings["hand"].full(k)["intervene"] == 1

    # and hands it back
    rings["control"].intervene = 2
    rings["control"].k_exec = k
    clk.advance(0.2)
    publish(rings, 2, k_first=k + 1, n=cfg.H, x=0.05)
    step = p6.step()
    assert step.written and step.k == k + 2
    assert rings["body"].full(step.k)["intervene"] == 2
    drive(p7, clk.t + 0.001)
    assert not p7.stalled, "the policy is driving again"


# --- P5: the handback --------------------------------------------------------


from tests.test_p5_policy import (  # noqa: E402
    StubPolicy, _arena_for, _fill_epoch, _loop)


def test_p5_does_not_infer_while_a_human_is_driving(cfg, hw):
    """The chunks would not be executed — P6 is in its guided branch for as long
    as the counter is odd — and producing one anyway leaves a chunk in the ring
    whose observation predates the correction, which P6 would then adopt at the
    handback."""
    with _arena_for(cfg, hw) as rings:
        loop, policy = _loop(cfg, rings)
        rings["control"].session = schema.ACTIVE
        rings["control"].intervene = 1
        # hand back on the first poll, so `_wait_out` terminates
        loop.clock.sleep_until = (
            lambda t, **_: setattr(rings["control"], "intervene", 2) or t)
        assert loop.step() is None
        assert policy.calls == [], "no inference at all"


def test_the_handback_re_founds_the_tick_line_on_the_live_cursor(cfg, hw):
    """THE DEADLOCK THIS AVOIDS. The ordinary schedule extrapolates a
    trigger instant off a WRITTEN body target, and there are none on the line P6
    is about to anchor to. Waiting for one is mutual: P6 cannot write until a
    chunk covers its cursor, and no chunk will until P5 runs."""
    with _arena_for(cfg, hw) as rings:
        loop, policy = _loop(cfg, rings)
        rings["control"].session = schema.ACTIVE
        loop.first_inference()
        rings["control"].intervene = 1
        loop.clock.sleep_until = (
            lambda t, **_: setattr(rings["control"], "intervene", 2) or t)
        loop.step()                                  # notices, waits it out
        rings["control"].k_exec = 137                # where the operator left it

        chunk_id = loop.step()
        assert chunk_id is not None
        slot = rings["chunks"]._rec[chunk_id % rings["chunks"].n]
        assert int(slot["k_first_action"]) == 138
        assert (loop.k_start_read, loop.k_next_inference) == (139, 137 + cfg.P)


def test_the_cursor_is_read_after_the_inference_and_not_before(cfg, hw):
    """At the instant the operator lets go, P7 is still draining the last
    `intervene_lead_s` of their targets and `k_exec` is still moving. An
    inference takes ~120 ms; by the time this reads it, it has been still."""
    with _arena_for(cfg, hw) as rings:
        loop, policy = _loop(cfg, rings)
        rings["control"].k_exec = 400
        original = policy.infer
        moved = []

        def infer(*a, **kw):
            rings["control"].k_exec = 500            # P7 drains during produce
            moved.append(True)
            return original(*a, **kw)

        policy.infer = infer
        chunk_id = loop.first_inference()
        slot = rings["chunks"]._rec[chunk_id % rings["chunks"].n]
        assert moved and int(slot["k_first_action"]) == 501


def test_a_chunk_inferred_across_a_takeover_is_dropped(cfg, hw):
    """Its observation predates the human's correction, and the tick line it was
    scheduled against may not be the one anybody is walking."""
    with _arena_for(cfg, hw) as rings:
        loop, policy = _loop(cfg, rings)
        _fill_epoch(rings["body"], range(2, cfg.P + 1), cfg)
        before = rings["control"].newest_chunk_id
        original = policy.infer

        def infer(*a, **kw):
            rings["control"].intervene = 1           # squeezed mid-inference
            return original(*a, **kw)

        policy.infer = infer
        assert loop.step() is None
        assert rings["control"].newest_chunk_id == before, "nothing published"


def test_a_whole_intervention_inside_one_sleep_still_re_founds_the_line(cfg, hw):
    """The deadlock the "changed", rather than "is intervening", test closes.

    P5's precise sleep can be most of a replan period, so a short takeover and
    its handback can both happen inside one. P5 would wake to an EVEN counter,
    see nothing wrong, and publish against cursors from before the takeover —
    which P6, anchored at `k_exec + 2`, never finds covering. It writes nothing,
    and P5's own window scan then waits for a target P6 will never write.
    """
    with _arena_for(cfg, hw) as rings:
        loop, policy = _loop(cfg, rings)
        _fill_epoch(rings["body"], range(2, cfg.P + 1), cfg)
        original = policy.infer

        def infer(*a, **kw):
            rings["control"].intervene = 2           # both edges, unobserved
            rings["control"].k_exec = 77
            return original(*a, **kw)

        policy.infer = infer
        assert loop.step() is None, "the stale chunk was dropped"

        chunk_id = loop.step()                       # ... and the line re-founded
        slot = rings["chunks"]._rec[chunk_id % rings["chunks"].n]
        assert int(slot["k_first_action"]) == 78


def test_the_first_inference_of_an_ordinary_run_is_unchanged(cfg, hw):
    """`k_exec` is 0 at the start of a rollout — homing's ramp ends at tick 0 and
    P7 has latched it — so the same method that serves the handback
    produces exactly what the first inference always produced."""
    with _arena_for(cfg, hw) as rings:
        loop, _ = _loop(cfg, rings)
        chunk_id = loop.first_inference()
        slot = rings["chunks"]._rec[chunk_id % rings["chunks"].n]
        assert (chunk_id, int(slot["k_first_action"])) == (1, 1)
        assert (loop.k_start_read, loop.k_next_inference) == (2, cfg.P)


def test_waiting_out_an_intervention_still_notices_the_run_ending(cfg, hw):
    """Without the session test a P5 parked here would never see P0 move the
    session on, and EXITING would hang on a process waiting for a headset."""
    with _arena_for(cfg, hw) as rings:
        loop, _ = _loop(cfg, rings)
        rings["control"].intervene = 1
        rings["control"].session = schema.EXITING
        loop._wait_out()                             # returns rather than hanging


# --- P0 and P9 ---------------------------------------------------------------


from g1d_deploy.config_loading import resolved as cfgmod                        # noqa: E402
from g1d_deploy.runtime.teleop import pico_service                           # noqa: E402
from g1d_deploy.config_loading.errors import ConfigError                     # noqa: E402
from g1d_deploy.runtime.launcher import Launcher         # noqa: E402
from g1d_deploy.runtime.processes.p9_record import Recorder           # noqa: E402


def _arm_qp(cfg):
    """`--intervene` runs `arm_qp` (`validate.check_intervene_ik`); the launcher
    refuses anything else, including the `arm_planner` default."""
    return cfgmod.replace(cfg, ik="arm_qp")


def test_the_recorder_keeps_the_operators_own_stream(rings, cfg, hw, tmp_path):
    """The target rings already carry `intervene`, so WHO was driving each tick
    is recoverable without this — but only as a label. This is the raw input
    behind it, which is what makes a session replayable offline and a mapping
    complaint diagnosable rather than a matter of opinion."""
    rec = Recorder(tmp_path, cfg, rings)
    for i in range(5):
        guide(rings, 1, T0 + i * 0.011, x=i * 0.1)
        rec.sweep()
    rec.close(hw)
    rows = np.fromfile(rec.intervene.path, dtype=rec.intervene.dtype)
    assert len(rows) == 5
    assert list(rows["index"]) == [1, 2, 3, 4, 5]
    assert rows["pose"][-1][0][0] == pytest.approx(0.4)
    assert set(rows["intervene"]) == {1}


def test_a_run_that_never_intervenes_records_an_empty_operator_stream(
        rings, cfg, hw, tmp_path):
    rec = Recorder(tmp_path, cfg, rings)
    rings["targets"].write(1, T0, np.zeros(BODY), intervene=0)
    rec.sweep()
    rec.close(hw)
    assert len(np.fromfile(rec.intervene.path, dtype=rec.intervene.dtype)) == 0


def test_every_recorded_target_says_which_authority_wrote_it(rings, cfg, hw,
                                                             tmp_path):
    """THE DAGGER PAIR. Segmenting a session into "the policy drove this" and
    "the human drove this" is what the whole feature is collected for, and it
    falls out of the counter already being in the slot."""
    rec = Recorder(tmp_path, cfg, rings)
    rings["targets"].write(1, T0 + 0.1, np.zeros(BODY), intervene=0)
    rings["targets"].write(2, T0 + 0.2, np.zeros(BODY), intervene=1)
    rings["targets"].write(3, T0 + 0.3, np.zeros(BODY), intervene=2)
    rec.sweep()
    rec.close(hw)
    rows = np.fromfile(rec.targets.path, dtype=rec.targets.dtype)
    by_k = {int(r["k"]): int(r["intervene"]) for r in rows}
    assert (by_k[1], by_k[2], by_k[3]) == (0, 1, 2)


def test_without_the_flag_p11_is_never_started(cfg, hw):
    """Off means off: `control.intervene` stays 0 for the life of the run, so
    `is_intervening` is never true and every process takes the path it took
    before any of this existed."""
    assert Launcher(cfg, hw, log=lambda *_: None).pico is False


def test_check_refuses_a_trigger_that_is_already_held(rings, cfg, hw, derived):
    """The one that catches a real mistake: a controller resting on a bench with
    something against it. A takeover needs a rising edge, so it could not have
    started an intervention — but it would not have WORKED when squeezed either,
    and the honest place to say so is before the robot has moved."""
    lch = Launcher(_arm_qp(cfg), hw, log=lambda *_: None, pico=True)
    lch.rings, lch.derived = rings, derived
    rings["intervene"].note(samples=10, fresh=1,
                            trigger=np.array([1.0, 0.0]))

    class Clock:
        t = T0

        def now(self):
            return self.t

        def sleep_until(self, t, **_):
            self.t = t
            rings["intervene"].note(
                samples=10 + int(hw.pico_poll_hz * _CHECK_WINDOW_S))
            return t

    lch.clock = Clock()
    with pytest.raises(ConfigError, match="left trigger is fully pressed"):
        lch._verify_pico()


class _Child:
    """A P11 stand-in. The CHECK waits ask one thing: is it still there?"""

    def __init__(self, alive: bool = True):
        self.alive = alive

    def is_alive(self):
        return self.alive


def _pico_launcher(cfg, hw, rings, derived, logged, *, alive=True):
    lch = Launcher(_arm_qp(cfg), hw, log=logged.append, pico=True)
    lch.rings, lch.derived = rings, derived
    lch.children.handles["p11_pico"] = _Child(alive)
    return lch


class _PicoClock:
    """Polls at `pico_poll_hz`, and connects the headset at `connect_at`.

    `samples` has to climb with real time rather than by a fixed step: the
    rate gate divides it by the measured window, and the fresh wait then polls
    at `_POLL_S` for as long as the headset takes.
    """

    def __init__(self, rings, hw, *, connect_at=None):
        self.t, self.rings, self.hw = T0, rings, hw
        self.connect_at = connect_at

    def now(self):
        return self.t

    def sleep_until(self, t, **_):
        self.t = t
        self.rings["intervene"].note(
            samples=10 + int(self.hw.pico_poll_hz * (t - T0)))
        if self.connect_at is not None and t - T0 >= self.connect_at:
            self.rings["intervene"].note(fresh=1)
        return t


def test_check_waits_for_a_headset_that_is_not_streaming_yet(
        rings, cfg, hw, derived):
    """THE POINT OF THE WAIT. The headset dials in to the PC Service over WiFi
    — nothing on this box dials out to it — so it can connect long after the
    box is up. This used to be a refusal, which meant the headset had to be on
    the operator's head before the command was typed and stay there through
    P5's ~40 s compile, or the run died at CHECK and they started over."""
    logged: list[str] = []
    lch = _pico_launcher(cfg, hw, rings, derived, logged)
    rings["intervene"].note(samples=10, fresh=0, trigger=np.array([0.0, 0.0]))
    lch.clock = _PicoClock(rings, hw, connect_at=12.0)

    lch._verify_pico()

    notes = [m for m in logged if "still waiting on p11_pico" in m]
    assert notes, "a wait with no timeout must say what it is waiting on"
    assert "63901" in notes[0], "and where to point the headset"
    assert logged[-1] == "[P0] p11_pico: streaming poses"


def test_the_two_pico_waits_are_told_apart_in_the_console(
        rings, cfg, hw, derived):
    """One child, waited on twice for different things. Two identical
    "producing" lines would hide that the second wait is a different question:
    P11 polling proves the SDK is alive, not that any pose is arriving."""
    logged: list[str] = []
    lch = _pico_launcher(cfg, hw, rings, derived, logged)
    rings["intervene"].note(samples=10, fresh=1, trigger=np.array([0.0, 0.0]))
    lch.clock = _PicoClock(rings, hw)

    lch._verify_pico()

    assert "[P0] p11_pico: polling" in logged
    assert "[P0] p11_pico: streaming poses" in logged


def test_check_refuses_a_dead_p11_instead_of_waiting_for_it_forever(
        rings, cfg, hw, derived):
    """The gap `6b33fc2` closed for P1 and the cameras and left open here: a
    P11 that raised in `xrt.init()` — a missing native library, a service that
    went away — used to leave CHECK polling a header nobody would write again,
    terminal silent, with no timeout to end it."""
    logged: list[str] = []
    lch = _pico_launcher(cfg, hw, rings, derived, logged, alive=False)
    rings["intervene"].note(samples=10, fresh=0, trigger=np.array([0.0, 0.0]))
    lch.clock = _PicoClock(rings, hw)

    with pytest.raises(ConfigError, match="p11_pico exited before"):
        lch._verify_pico()


def test_the_service_autostart_is_warned_never_fatal(cfg, hw, monkeypatch):
    """Like the board imager: the autostart saves the operator a terminal, it
    is not the gate. `_verify_pico` is — it asks whether poses are ARRIVING,
    which no amount of process bookkeeping can substitute for."""
    logged: list[str] = []
    lch = Launcher(_arm_qp(cfg), hw, log=logged.append, pico=True)

    def boom(**_):
        raise pico_service.PicoServiceError("no launcher at /nope")

    monkeypatch.setattr(pico_service, "ensure_running", boom)
    lch._ensure_pico_service()          # must NOT raise

    assert any("autostart FAILED" in m for m in logged)
    assert any("continuing anyway" in m for m in logged)


def test_a_service_that_is_already_up_is_left_alone(monkeypatch):
    """Idempotent, so it is safe on every run. The running service holds 63901;
    a second one would only fight the first for it."""
    monkeypatch.setattr(pico_service, "is_running", lambda: True)

    def never(*a, **k):
        raise AssertionError("must not start a second service")

    monkeypatch.setattr(pico_service.subprocess, "Popen", never)
    assert pico_service.ensure_running() is True

