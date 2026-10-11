"""P7 — the 500 Hz servo loop, and the stall in particular.

Every test here drives the real `BodyServo` against the real arena rings with a
`FakeLowCmd` and an injected time, because the things that go wrong in a servo
loop go wrong *quietly*: a stall that absorbs tracking error still publishes
plausible numbers, and a segment restarted from an expired deadline still
converges — it just does it in one violent step that the slew clamp then hides.
Nothing here sleeps.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from elrobot_deploy.runtime.processes import p7_body
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.processes.p7_body import BodyServo
from elrobot_deploy.runtime.drivers.arm_command import FakeLowCmd, ZeroComp

PERIOD = 1.0 / 500.0


class RecordingComp:
    """A gravity comp that records the measured vector it was handed.

    the stall has two halves — the command freezes, the compensation does not — and
    only the second one needs to observe the measurement.
    """

    def __init__(self):
        self.seen: list[np.ndarray] = []

    def tau(self, q_measured29):
        self.seen.append(np.asarray(q_measured29, dtype=np.float64).copy())
        return np.zeros(schema.BODY_DOF)


def push_state(rings, t: float, q29) -> None:
    """One P1-shaped state sample. Only `q` matters to P7."""
    rings["state"].write(t_recv=t, q=np.asarray(q29, dtype=np.float64),
                         dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=t)


def make(rings, hw, *, q0=None, t0=0.0, gravity=None):
    out = FakeLowCmd(hw)
    out.connect(np.zeros(len(hw.joints_leg)))
    q0 = np.zeros(schema.BODY_DOF) if q0 is None else np.asarray(q0, float)
    servo = BodyServo(ring=rings["body"], control=rings["control"],
                      state=rings["state"], out=out,
                      gravity=ZeroComp() if gravity is None else gravity,
                      hw=hw, q_cmd_last=q0, t0=t0)
    return servo, out


def tick(servo, t: float) -> None:
    """One servo frame: step until something is published.

    The segment-advance branch neither publishes nor waits — it falls straight
    into the next iteration, which publishes microseconds later against the new
    segment. Two calls is the worst case, because an advance always
    installs a `t_segment_end` strictly greater than `t`.
    """
    for _ in range(3):
        if servo.step(t):
            return
    raise AssertionError("step() never published")


# --- interpolation ----------------------------------------------------------


def test_interpolation_lands_exactly_on_the_target_at_the_deadline(rings, hw):
    """u = 1 must be bit-exact, not nearly. The endpoint is the only sample the
    next segment inherits as its start, so a residue here accumulates."""
    q_end = np.full(schema.BODY_DOF, 0.02)
    rings["targets"].write(0, 1.0, q_end, intervene=0)
    servo, out = make(rings, hw, t0=0.0)

    for i in range(501):                              # t = 0.000 .. 1.000
        tick(servo, i * PERIOD)

    assert np.array_equal(out.q_body[-1], q_end)
    assert rings["control"].k_exec == 0


def test_published_values_never_overshoot_the_segment(rings, hw):
    """The upper end of the `u` clamp, stated observably.

    `u > 1` is unreachable through `step` — the branch guard sends any `t_now`
    past the deadline to the segment advance — so the assertion that carries
    meaning is that nothing published inside a segment ever leaves the interval
    the segment spans.
    """
    q_end = np.full(schema.BODY_DOF, 0.02)
    rings["targets"].write(0, 1.0, q_end, intervene=0)
    servo, out = make(rings, hw, t0=0.0)
    for i in range(501):
        tick(servo, i * PERIOD)

    q = out.q_body
    assert q.min() >= 0.0
    assert q.max() <= q_end.max()
    assert np.all(np.diff(q, axis=0) >= 0.0)          # monotone, no overshoot


def test_u_clamps_at_zero_on_an_inverted_span(rings, hw):
    """The lower end of the clamp, on the state that actually produces it.

    A segment whose start is later than its end collapses `span` to `EPS_SPAN`,
    and `(t_now - t_segment_start) / 1e-9` is then a number in the hundreds of
    millions. Without the clamp the commanded posture would be that number times
    the joint delta; with it, the loop holds the segment start.
    """
    q_start = np.full(schema.BODY_DOF, 0.10)
    q_end = np.full(schema.BODY_DOF, 0.20)
    servo, out = make(rings, hw, q0=q_start, t0=0.0)
    servo.seg.t_start, servo.seg.x_start = 1.0, q_start
    servo.seg.t_end, servo.seg.x_end = 0.5, q_end
    servo.seg.stalled = False

    assert servo.step(0.4) is True
    assert np.array_equal(out.q_body[-1], q_start)
    assert np.isfinite(out.q_body[-1]).all()


# --- The stall ------------------------------------------------------


def test_a_stall_freezes_the_command_and_never_absorbs_tracking_error(rings, hw):
    """The decision the stall overturns: while stalled P7 republishes `q_cmd_last`,
    NOT the measured `q`.

    The measured posture here sags a millimetre of radian per iteration, which
    is what a real arm does under a stall. If the loop published the
    measurement, that value would become the next `q_cmd_last` and the command
    would walk after the sag — plausible numbers, a drifting robot, and nothing
    reporting it. So: 400 stalled iterations, and the command must not move by
    one ulp.
    """
    q_cmd = np.full(schema.BODY_DOF, 0.30)
    grav = RecordingComp()
    servo, out = make(rings, hw, q0=q_cmd, t0=0.0, gravity=grav)

    measured = np.zeros(schema.N_JOINTS)
    for i in range(400):
        measured = measured + 0.001                   # the arm sags, steadily
        push_state(rings, i * PERIOD, measured)
        tick(servo, i * PERIOD)

    assert servo.stalled is True
    assert len(out.published) == 400
    assert np.array_equal(out.q_body, np.tile(q_cmd, (400, 1)))
    assert np.array_equal(servo.q_cmd_last, q_cmd)
    assert np.array_equal(servo.q_segment_start, q_cmd)

    # The other half of the stall: gravity compensation still tracks the measurement.
    assert len(grav.seen) == 400
    assert np.allclose(grav.seen[-1], 0.400)
    assert not np.allclose(grav.seen[0], grav.seen[-1])


def test_a_stall_pushes_the_segment_start_forward_every_iteration(rings, hw):
    servo, _ = make(rings, hw, t0=0.0)
    for i in range(1, 51):
        tick(servo, i * PERIOD)
        assert servo.t_segment_start == pytest.approx(i * PERIOD)


def test_leaving_a_stall_spreads_the_motion_over_the_time_that_remains(rings, hw):
    """The stall's own example, executed.

    `deadline(25) = 0.8`, `deadline(26) = 1.4`, and `target(26)` does not exist
    until `t = 1.3` because 600 ms of inference sits between the two ticks.
    Interpolating from `t_segment_start = 0.8` the moment the target appears
    demands 84% of the joint delta in the first 2 ms frame; pushing the start
    forward during the stall makes the segment `[1.298, 1.4]` and spreads the
    same delta over the ~100 ms that actually remain.

    The joint delta is 0.05 rad, so the two answers differ by a factor of forty
    and the slew clamp (0.003 rad on the tightest joint) would have hidden the
    wrong one as a long ramp instead of a step.
    """
    q_a = np.zeros(schema.BODY_DOF)
    q_b = np.full(schema.BODY_DOF, 0.05)
    rings["targets"].write(25, 0.8, q_a, intervene=0)
    servo, out = make(rings, hw, q0=q_a, t0=0.0)

    k_exec_during_stall = []
    landed_at = None
    for i in range(701):                              # t = 0.000 .. 1.400
        t = i * PERIOD
        if t >= 1.3 and landed_at is None:
            rings["targets"].write(26, 1.4, q_b, intervene=0)         # the target appears
            landed_at = len(out.published)
        tick(servo, t)
        if landed_at is None and t > 0.8:
            k_exec_during_stall.append(rings["control"].k_exec)

    # k_exec named tick 25 for the whole stall and only moved on the advance.
    assert k_exec_during_stall
    assert set(k_exec_during_stall) == {25}
    assert rings["control"].k_exec == 26

    q = out.q_body
    steps = np.abs(np.diff(q, axis=0)).max(axis=1)
    first_step = steps[landed_at - 1]

    # [1.298, 1.400] is 51 frames, so ~1/51 of the delta per frame.
    assert first_step == pytest.approx(0.05 * PERIOD / 0.102, rel=0.05)
    # The bug this exists to catch: restarting from t_segment_start = 0.8 would
    # command (1.300 - 0.8) / 0.6 = 83% of the delta in one 2 ms frame.
    assert first_step < 0.05 * 0.10
    # ...and the motion still completes, to within the one frame the 500 Hz
    # grid leaves between the last publish and the deadline.
    assert np.allclose(q[-1], q_b, atol=first_step * 1.01)


def test_a_stall_leaves_k_exec_alone_and_it_never_decreases(rings, hw):
    """`k_exec` monotonicity: `k_exec` is non-decreasing and changes only in the advance branch.

    The ring below has a hole at tick 1 — an IK solve that overran its budget
    — and a gap after tick 2 that stalls the loop. Neither may move the
    cursor backwards or move it at all while stalled.
    """
    q = np.full(schema.BODY_DOF, 0.01)
    rings["targets"].write(0, 0.10, q * 1, intervene=0)
    rings["targets"].write(2, 0.30, q * 2, intervene=0)               # tick 1 was never solved
    servo, out = make(rings, hw, t0=0.0)

    seen = []
    for i in range(251):                              # t = 0.000 .. 0.500
        n_before = len(out.published)
        k_before = rings["control"].k_exec
        tick(servo, i * PERIOD)
        seen.append(rings["control"].k_exec)
        if servo.stalled and n_before < len(out.published):
            assert rings["control"].k_exec == k_before

    assert seen == sorted(seen)
    assert seen[-1] == 2
    assert servo.stalled is True


# --- The one clamp ----------------------------------------------------------


def test_the_slew_clamp_binds_at_qdot_max_over_body_hz(rings, hw):
    """The slew-clamp rule, and the only clamp in this loop.

    A one-frame segment to a posture 5 rad away is not a situation the system
    should ever produce — P6 clamps against its seed and homing derives its
    duration from the same ceiling — so the assertion is that when it happens
    anyway the published step is exactly the per-frame limit, per joint, and
    not one part in a thousand more.
    """
    limit = np.asarray(hw.qdot_max_body) / hw.body_hz
    rings["targets"].write(0, PERIOD, np.full(schema.BODY_DOF, 5.0), intervene=0)
    servo, out = make(rings, hw, t0=0.0)

    tick(servo, 0.0)
    tick(servo, PERIOD)

    assert np.array_equal(out.q_body[0], np.zeros(schema.BODY_DOF))
    assert np.array_equal(out.q_body[1], limit)
    assert np.all(np.abs(np.diff(out.q_body, axis=0)) <= limit + 1e-12)


def test_no_published_step_ever_exceeds_the_limit(rings, hw):
    limit = np.asarray(hw.qdot_max_body) / hw.body_hz
    rings["targets"].write(0, 0.05, np.full(schema.BODY_DOF, 1.0), intervene=0)
    rings["targets"].write(1, 0.06, np.full(schema.BODY_DOF, -1.0), intervene=0)
    servo, out = make(rings, hw, t0=0.0)
    for i in range(101):
        tick(servo, i * PERIOD)
    assert np.all(np.abs(np.diff(out.q_body, axis=0)) <= limit + 1e-12)


# --- The advance branch -----------------------------------------------------


def test_the_segment_advance_branch_publishes_nothing(rings, hw):
    """It does not publish and does not wait; the next iteration does both."""
    rings["targets"].write(0, 1.0, np.full(schema.BODY_DOF, 0.01), intervene=0)
    servo, out = make(rings, hw, t0=0.0)

    assert servo.step(0.0) is False                   # advance
    assert out.published == []
    assert rings["control"].k_exec == 0

    assert servo.step(0.0) is True                    # publish
    assert len(out.published) == 1


def test_the_stall_branch_publishes_and_waits(rings, hw):
    servo, out = make(rings, hw, t0=0.0)
    assert servo.step(0.1) is True
    assert len(out.published) == 1


# --- The velocity feedforward -----------------------------------------------


def test_dq_is_the_segment_slope(rings, hw):
    """The published velocity is the derivative of the published position.

    Not an approximation of it and not a filtered version of it: the loop walks
    a straight line between two targets, so the slope is exact and constant for
    the whole segment. That is the number the firmware's `kd` term needs in
    order to stop reading intended motion as error.
    """
    q_end = np.full(schema.BODY_DOF, 0.02)
    rings["targets"].write(0, 1.0, q_end, intervene=0)
    servo, out = make(rings, hw, t0=0.0)
    for i in range(400):
        tick(servo, i * PERIOD)

    # q_start is zero and the segment spans [0, 1] s, so the slope is q_end / 1.
    assert np.allclose(out.dq, np.tile(q_end / 1.0, (len(out.dq), 1)))

    # And it really is the position's derivative: integrating it over the
    # segment reproduces the travel the loop actually commanded.
    travelled = out.q_body[-1] - out.q_body[0]
    predicted = out.dq[-1] * (len(out.q_body) - 1) * PERIOD
    assert np.allclose(travelled, predicted, atol=1e-9)


def test_dq_is_zero_while_stalled(rings, hw):
    """A stall is a hold, so the intended velocity is nothing.

    The failure this rules out is subtle and one-directional: carry the expired
    segment's slope into the stall and the damper is told to expect travel that
    the frozen `q_cmd_last` is not commanding, so the joint is pushed off the
    hold by a feedforward alone — the ratchet the stall exists to prevent, arriving
    through the other channel.
    """
    servo, out = make(rings, hw, q0=np.full(schema.BODY_DOF, 0.1), t0=0.0)
    for i in range(50):
        tick(servo, i * PERIOD)
    assert servo.stalled
    assert np.array_equal(out.dq, np.zeros_like(out.dq))


def test_dq_is_zero_where_the_position_clamp_holds_an_endpoint(rings, hw):
    """An inverted span pins `q_t`; the velocity must be pinned with it.

    Same state as `test_u_clamps_at_zero_on_an_inverted_span`. There `span`
    collapses to `EPS_SPAN`, which makes `delta / span` a number in the
    hundreds of millions — so this is also the case that says the velocity is
    taken from `u_raw` rather than from the clamped `u`, and that a feedforward
    is never allowed to be a number the machine cannot do.
    """
    q_start = np.full(schema.BODY_DOF, 0.10)
    servo, out = make(rings, hw, q0=q_start, t0=0.0)
    servo.seg.t_start, servo.seg.x_start = 1.0, q_start
    servo.seg.t_end = 0.5
    servo.seg.x_end = np.full(schema.BODY_DOF, 0.20)
    servo.seg.stalled = False

    assert servo.step(0.4) is True
    assert np.array_equal(out.q_body[-1], q_start)
    assert np.array_equal(out.dq[-1], np.zeros(schema.BODY_DOF))


def test_dq_never_exceeds_the_velocity_ceiling(rings, hw):
    """The velocity channel is bounded by the same number the positions are.

    A velocity feedforward is a torque request, so an unbounded one is an
    unbounded torque. `qdot_max_body` is the ceiling the slew clamp is derived
    from, and using the same one is what stops the two channels from
    disagreeing about what this machine may do.
    """
    q_start = np.zeros(schema.BODY_DOF)
    servo, out = make(rings, hw, q0=q_start, t0=0.0)
    servo.seg.t_start, servo.seg.x_start = 0.0, q_start
    # A whole radian in one servo period: far past any real target.
    servo.seg.t_end = PERIOD
    servo.seg.x_end = np.ones(schema.BODY_DOF)
    servo.seg.stalled = False

    assert servo.step(PERIOD / 2) is True
    assert np.all(np.abs(out.dq[-1]) <= hw.qdot_max_body + 1e-12)
    assert np.all(out.dq[-1] > 0.0), "the ceiling bounds it, it does not zero it"


# --- what P7 is not ---------------------------------------------------------


def test_p7_never_damps(rings, hw):
    """The damp is P0's. Keeping it out of the 500 Hz loop is an
    explicit design decision, so it is worth a test rather than a comment."""
    rings["targets"].write(0, 0.05, np.full(schema.BODY_DOF, 0.01), intervene=0)
    servo, out = make(rings, hw, t0=0.0)
    for i in range(200):
        tick(servo, i * PERIOD)
    assert out.damped is False

    src = pathlib.Path(p7_body.__file__).read_text()
    assert ".damp(" not in src


def test_p7_imports_nothing_from_the_ik_stack():
    """P7's import rule, checked the only way that means anything: transitively.

    A source scan catches a module-scope `import mink`; it does not catch
    `elrobot_deploy.runtime.ik.arm_qp` growing a mujoco dependency three levels down. Importing the
    module in a clean interpreter and reading `sys.modules` does.
    """
    probe = ("import sys; import elrobot_deploy.runtime.processes.p7_body as m; "
             "print(sorted({n.split('.')[0] for n in sys.modules} & "
             "{'mujoco','mink','dm_control','casadi','qpsolvers'}))")
    root = pathlib.Path(p7_body.__file__).resolve().parents[3]
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=root)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "[]", res.stdout

    # ...and "at module scope or otherwise": every `import` in the file,
    # including the ones inside functions, parsed rather than grepped so that
    # the module's own prose about P7's import rule is not mistaken for a violation.
    banned = {"mujoco", "mink", "dm_control", "ik", "kin"}
    tree = ast.parse(pathlib.Path(p7_body.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [f"{node.module or ''}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            parts = set(name.lstrip(".").split("."))
            assert not (parts & banned), f"P7 imports {name!r}"


# --- P0's homing gate -------------------------------------------


class LatchAfter:
    """A `ready` stand-in that records how much had been published when it was
    set. The claim is an ORDERING, so the count is the assertion."""

    def __init__(self, out):
        self.out = out
        self.published_at = None

    def set(self) -> None:
        self.published_at = len(self.out.published)


def test_run_sets_ready_only_once_a_command_is_on_the_wire(arena, rings, cfg, hw):
    """P0 waits on this latch before it writes the homing ramp.

    It must be set after the first PUBLISH and not after `connect`, because
    everything expensive in this process — the DDS handshake, pinocchio's URDF
    — sits between the two. A gate that opened on `connect` would open on a P7
    with a second of loading still to go, and P0 would resume writing 90 ticks
    into a ring no interpolator was reading. That is the 2026-08-26 bug: the arm
    stood still for 2.77 s and then covered the whole ramp in ~150 ms.

    `session` is STOPPED before the call, so the servo loop exits immediately
    and only the pre-loop publish runs — which is exactly the code under test.
    """
    push_state(rings, 0.0, np.zeros(schema.N_JOINTS))
    rings["control"].session = schema.STOPPED

    out = FakeLowCmd(hw)
    ready = LatchAfter(out)
    p7_body.run(arena.shm.name, cfg, hw, out=out, gravity=ZeroComp(),
                ready=ready)

    assert ready.published_at is not None, "P7 never set the latch"
    assert ready.published_at >= 1, "the latch opened before P7 published"


def test_run_without_a_latch_is_unchanged(arena, rings, cfg, hw):
    """`ready` is optional: P7 is spawned by tools and tests that have no gate,
    and it must not learn to require one. It is also the ONLY thing this process
    is told about the session sequence — it sets the latch and never reads it,
    so "P7 knows nothing" still holds."""
    push_state(rings, 0.0, np.zeros(schema.N_JOINTS))
    rings["control"].session = schema.STOPPED

    out = FakeLowCmd(hw)
    p7_body.run(arena.shm.name, cfg, hw, out=out, gravity=ZeroComp())

    assert len(out.published) >= 1 and out.closed


# --- bringing the wire up ---------------------------------------------------


def test_the_wire_refuses_rather_than_hanging_when_lowstate_is_silent(
        monkeypatch, hw):
    """`connect` waits for one `rt/lowstate` sample to read `mode_machine`, and
    that wait must be bounded.

    It was not. The bare `sub.Read()` is `take_one()` with no wait bound and does
    not return on a silent topic — measured 2026-09-02 against a robot that had
    stopped publishing: still blocked after 6 s, where `Read(0.5)` returned None
    in 0.50 s.

    This is the DAMP's path. P0 reaches `_damp` from `finally` on every exit
    including the failure exits, so a robot that goes quiet used to hang P0 in
    EXITING — the one state it must never hang in, having already failed and
    being on its way to the only thing that ends a `kp`-held setpoint.
    """
    pytest.importorskip("cyclonedds")          # the message types are real
    from elrobot_deploy.runtime.drivers.dds import channel
    from elrobot_deploy.runtime.drivers import arm_command as lowcmd

    reads = []

    class _SilentSub:
        def __init__(self, *a, **k):
            pass

        def Init(self):
            pass

        def Read(self, timeout=None):
            reads.append(timeout)
            return None                 # the topic is silent, forever

        def Close(self):
            pass

    monkeypatch.setattr(channel, "ChannelFactoryInitialize", lambda *a, **k: None)
    monkeypatch.setattr(channel, "ChannelPublisher", lambda *a, **k: None)
    monkeypatch.setattr(channel, "ChannelSubscriber", _SilentSub)
    monkeypatch.setattr(lowcmd.DdsLowCmd, "MODE_MACHINE_TIMEOUT_S", 0.3)

    wire = lowcmd.DdsLowCmd(hw)
    with pytest.raises(lowcmd.LowCmdError, match="mode_machine"):
        wire.connect(np.zeros(len(hw.joints_leg)))

    assert reads, "connect must actually poll the topic"
    # EVERY read is bounded. One bare `Read()` anywhere in the loop is the bug.
    assert all(t is not None for t in reads)
