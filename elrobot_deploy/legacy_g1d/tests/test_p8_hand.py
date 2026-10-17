"""P8 — P7's loop over the hand column, at 200 Hz, minus everything.

The tests mirror `test_p7_body.py` deliberately. P8's docstring says "identical in
structure", and the way that claim decays is by one of the two loops acquiring a
fix the other never gets — so the stall, the interpolation and the advance
branch are asserted here in the same shapes, and the *differences* (no cursor,
no torque, no gravity comp, no slew clamp) are asserted as absences.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from elrobot_deploy.runtime.processes import p8_hand
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.processes.p8_hand import HandServo
from elrobot_deploy.runtime.drivers.gripper import FakeHandCmd

PERIOD = 1.0 / 200.0
SHAPE = (schema.N_HANDS, schema.HAND_DIM)


def make(rings, hw, *, cmd0=None, t0=0.0):
    out = FakeHandCmd(hw)
    out.connect()
    cmd0 = np.zeros(SHAPE) if cmd0 is None else np.asarray(cmd0, float)
    servo = HandServo(ring=rings["hand"], out=out, hw=hw, cmd_last=cmd0,
                      t0=t0, control=rings["control"])
    return servo, out


def tick(servo, t: float) -> None:
    for _ in range(3):
        if servo.step(t):
            return
    raise AssertionError("step() never published")


# --- interpolation ----------------------------------------------------------


def test_interpolation_lands_exactly_on_the_target_at_the_deadline(rings, hw):
    cmd_end = np.full(SHAPE, 0.5)
    rings["targets"].write(0, 1.0, np.zeros(schema.BODY_DOF), cmd_end, intervene=0)
    servo, out = make(rings, hw, t0=0.0)

    for i in range(201):                              # t = 0.000 .. 1.000
        tick(servo, i * PERIOD)

    assert out.cmd.shape == (201, *SHAPE)
    assert np.array_equal(out.cmd[-1], cmd_end)


def test_published_values_never_overshoot_the_segment(rings, hw):
    cmd_end = np.full(SHAPE, 0.5)
    rings["targets"].write(0, 1.0, np.zeros(schema.BODY_DOF), cmd_end, intervene=0)
    servo, out = make(rings, hw, t0=0.0)
    for i in range(201):
        tick(servo, i * PERIOD)

    c = out.cmd
    assert c.min() >= 0.0
    assert c.max() <= 0.5
    assert np.all(np.diff(c, axis=0) >= 0.0)


def test_u_clamps_at_zero_on_an_inverted_span(rings, hw):
    """Same collapse as P7's: an end before its start floors `span` at
    `EPS_SPAN`, and only the clamp keeps the published command finite and in
    range. It matters more here — the number goes to a jaw, not through a
    per-joint velocity ceiling that would have caught it."""
    a, b = np.full(SHAPE, 1.2), np.full(SHAPE, 5.4)
    servo, out = make(rings, hw, cmd0=a, t0=0.0)
    servo.seg.t_start, servo.seg.x_start = 1.0, a
    servo.seg.t_end, servo.seg.x_end = 0.5, b
    servo.seg.stalled = False

    assert servo.step(0.4) is True
    assert np.array_equal(out.cmd[-1], a)


# --- The stall --------------------------------------------------------------


def test_a_stall_republishes_the_last_command_frozen(rings, hw):
    """A stall must not creep toward the segment it was heading for, and must
    not creep anywhere else either. 400 iterations, bit-identical."""
    cmd = np.full(SHAPE, 3.0)
    servo, out = make(rings, hw, cmd0=cmd, t0=0.0)

    for i in range(400):
        tick(servo, i * PERIOD)

    assert servo.stalled is True
    assert out.cmd.shape == (400, *SHAPE)
    assert np.array_equal(out.cmd, np.tile(cmd, (400, 1, 1)))
    assert np.array_equal(servo.cmd_last, cmd)


def test_a_stall_pushes_the_segment_start_forward_every_iteration(rings, hw):
    servo, _ = make(rings, hw, t0=0.0)
    for i in range(1, 51):
        tick(servo, i * PERIOD)
        assert servo.t_segment_start == pytest.approx(i * PERIOD)


def test_leaving_a_stall_spreads_the_motion_over_the_time_that_remains(rings, hw):
    """P8's own sentence, executed: "a jaw commanded to close across a 600 ms
    inference gap must not be asked to do it in the 90 ms after the target
    lands".

    Same timeline as the stall — `deadline(25) = 0.8`, `deadline(26) = 1.4`, the
    target lands at 1.3 — and the same two candidate answers, four decades
    apart. The hand has no velocity ceiling behind it (see the module
    docstring), so the wrong answer reaches the actuator unfiltered.
    """
    a = np.full(SHAPE, 5.4)                           # open
    b = np.full(SHAPE, 1.2)                           # closed
    rings["targets"].write(25, 0.8, np.zeros(schema.BODY_DOF), a, intervene=0)
    servo, out = make(rings, hw, cmd0=a, t0=0.0)

    landed_at = None
    for i in range(281):                              # t = 0.000 .. 1.405
        t = i * PERIOD
        if t >= 1.3 and landed_at is None:
            rings["targets"].write(26, 1.4, np.zeros(schema.BODY_DOF), b, intervene=0)
            landed_at = len(out.published)
        tick(servo, t)

    c = out.cmd.reshape(len(out.published), -1)
    steps = np.abs(np.diff(c, axis=0)).max(axis=1)
    first_step = steps[landed_at - 1]
    travel = 5.4 - 1.2

    # [1.295, 1.400] at 200 Hz: ~1/21 of the travel per frame.
    assert first_step == pytest.approx(travel * PERIOD / 0.105, rel=0.05)
    # Restarting from t_segment_start = 0.8 would have commanded
    # (1.30 - 0.8) / 0.6 = 83% of the travel in one 5 ms frame.
    assert first_step < travel * 0.10
    # ...and the jaw still arrives, to within the one frame the 200 Hz grid
    # leaves between the last publish and the deadline.
    assert np.allclose(c[-1], 1.2, atol=first_step * 1.01)


# --- The range clamp ---------------------------------------------


def test_a_command_past_the_top_of_travel_is_clamped(rings, hw):
    """A policy once commanded 5.4304 rad against a Dex-1 stop at 5.4, and nothing
    in the system said so. The driver's own clamp does not cover this: it bounds
    each STEP against the measured position, not the target."""
    over = np.full(SHAPE, hw.hand_max + 0.03)
    rings["targets"].write(0, 1.0, np.zeros(schema.BODY_DOF), over, intervene=0)
    servo, out = make(rings, hw, t0=0.0)

    for i in range(201):
        tick(servo, i * PERIOD)

    assert out.cmd.max() == pytest.approx(hw.hand_max)
    assert np.all(out.cmd <= hw.hand_max + 1e-12)


def test_a_command_below_the_closed_stop_is_clamped(rings, hw):
    rings["targets"].write(0, 1.0, np.zeros(schema.BODY_DOF), np.full(SHAPE, hw.hand_min - 1.0), intervene=0)
    servo, out = make(rings, hw, cmd0=np.full(SHAPE, 2.0), t0=0.0)

    for i in range(201):
        tick(servo, i * PERIOD)

    assert out.cmd.min() == pytest.approx(hw.hand_min)


def test_the_clamp_does_not_touch_a_command_inside_travel(rings, hw):
    """It is a bound, not a rescale: everything reachable must pass through
    unchanged, or the gripper's whole trajectory is quietly distorted."""
    inside = np.full(SHAPE, 0.5 * (hw.hand_min + hw.hand_max))
    rings["targets"].write(0, 1.0, np.zeros(schema.BODY_DOF), inside, intervene=0)
    servo, out = make(rings, hw, t0=0.0)

    for i in range(201):
        tick(servo, i * PERIOD)

    assert np.array_equal(out.cmd[-1], inside)


def test_a_measured_aperture_outside_travel_is_clamped_before_it_seeds(rings, hw):
    """`cmd_last` comes from `grip_meas`. A jaw resting a hair past its
    own reported stop would otherwise make the FIRST published command — the
    frozen republish of a stall — out of range."""
    servo, out = make(rings, hw, cmd0=np.full(SHAPE, hw.hand_max + 0.5), t0=0.0)

    tick(servo, 0.0)                                  # stalled: republish

    assert out.cmd[-1].max() == pytest.approx(hw.hand_max)


def test_the_segment_advance_branch_publishes_nothing(rings, hw):
    rings["targets"].write(0, 1.0, np.zeros(schema.BODY_DOF), np.full(SHAPE, 0.1), intervene=0)
    servo, out = make(rings, hw, t0=0.0)

    assert servo.step(0.0) is False
    assert out.published == []
    assert servo.step(0.0) is True
    assert len(out.published) == 1


# --- what P8 is not ---------------------------------------------------------


def test_p8_publishes_no_cursor(rings, hw):
    """There is ONE `k_exec` and P7 writes it. P8 sits in the same
    segment by construction — both rings carry the same `k` and the same
    deadline for every tick — so a second cursor would only be a second
    way to be wrong about the same fact."""
    rings["targets"].write(7, 0.5, np.zeros(schema.BODY_DOF), np.full(SHAPE, 0.2), intervene=0)
    rings["targets"].write(8, 1.0, np.zeros(schema.BODY_DOF), np.full(SHAPE, 0.4), intervene=0)
    servo, _ = make(rings, hw, t0=0.0)
    for i in range(250):
        tick(servo, i * PERIOD)

    assert rings["control"].k_exec == 0                # untouched

    src = pathlib.Path(p8_hand.__file__).read_text()
    tree = ast.parse(src)
    assert "k_exec" not in {node.attr for node in ast.walk(tree)
                            if isinstance(node, ast.Attribute)}


def test_p8_has_no_torque_no_gravity_comp_and_no_ik(rings, hw):
    """P8's list of differences, as absences. `FakeHandCmd.publish` takes one
    argument; a torque term could not be delivered even if the loop computed
    one."""
    banned = {"mujoco", "mink", "ik", "kin", "pinocchio", "lowcmd"}
    tree = ast.parse(pathlib.Path(p8_hand.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [f"{node.module or ''}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            parts = set(name.lstrip(".").split("."))
            assert not (parts & banned), f"P8 imports {name!r}"


def test_the_wire_puts_dim_zero_of_each_hand_on_the_gripper_motors(rings, hw):
    """The (2, 6) contract meets one single-motor topic per hand.

    `rt/dex1/{left,right}/cmd` are the Dex-1's, one server per gripper, and only
    dim 0 of each hand block is on them. The other five dims are carried, not
    dropped, so that the day the Brainco mapping is settled the change is in the
    binding and not in the loop.
    """
    from elrobot_deploy.runtime.drivers import gripper as wire

    out = FakeHandCmd(hw)
    out.connect()
    cmd = np.arange(12, dtype=np.float64).reshape(SHAPE)
    out.publish(cmd)

    assert np.array_equal(out.cmd[-1], cmd)           # all six survive
    assert wire.JAW_MOTOR_ID == 0                     # one motor per topic
    assert len(hw.topic_hand_cmd) == SHAPE[0]         # one topic per hand

    with pytest.raises(ValueError):
        out.publish(np.zeros(6))                     # a (6,) is not a command


# --- P0's homing gate -------------------------------------------


class LatchAfter:
    """A `ready` stand-in that records how much had been published when set."""

    def __init__(self, out):
        self.out = out
        self.published_at = None

    def set(self) -> None:
        self.published_at = len(self.out.published)


def push_state(rings, t: float, grip=0.0) -> None:
    """One P1-shaped state sample. Only `grip_meas` matters to P8."""
    rings["state"].write(t_recv=t, q=np.zeros(schema.N_JOINTS),
                         dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.full(schema.GRIP_DIM, grip),
                         t_grip_recv=t)


def test_run_sets_ready_only_once_a_command_is_on_the_wire(arena, rings, cfg, hw):
    """P7's gate, and P8 is gated for the same reason rather than for symmetry:
    the homing ramp writes BOTH rings, so a P0 that waited only on P7 would
    still hand a late P8 a ring full of apertures whose deadlines had passed."""
    push_state(rings, 0.0, grip=5.4)
    rings["control"].session = schema.STOPPED

    out = FakeHandCmd(hw)
    ready = LatchAfter(out)
    p8_hand.run(arena.shm.name, cfg, hw, out=out, ready=ready)

    assert ready.published_at is not None, "P8 never set the latch"
    assert ready.published_at >= 1, "the latch opened before P8 published"
    # And what it published is the MEASURED aperture, not a default.
    assert np.allclose(out.published[0], 5.4)


def test_run_without_a_latch_is_unchanged(arena, rings, cfg, hw):
    """`ready` is optional, exactly as in P7."""
    push_state(rings, 0.0)
    rings["control"].session = schema.STOPPED

    out = FakeHandCmd(hw)
    p8_hand.run(arena.shm.name, cfg, hw, out=out)

    assert len(out.published) >= 1 and out.closed
