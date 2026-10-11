"""`elrobot_deploy.runtime.teleop.intervention` — the arithmetic of a takeover.

Pure functions only: no arena, no clock, no SDK, no robot. Everything here runs
on a laptop, which is the point of keeping the geometry out of P11.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.runtime.teleop import intervention as iv
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.robot.kinematics import mat_from_quat

#: Realistic offsets, so the numbers in these tests are the numbers a
#: run would see: the Dex-1 fingertip midpoint is 18.5 cm out of the flange, and
#: an operator's virtual gripper sits 5 cm ahead of the controller.
TCP = np.array([0.184759, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
HAND_TCP = np.array([0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def quat(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * a)])


def pose(xyz, q) -> np.ndarray:
    return np.concatenate([np.asarray(xyz, dtype=np.float64), q])


# --- The mapping ------------------------------------------------------------


@pytest.mark.parametrize("robot_q, hand_q", [
    (quat([0, 0, 1], 0.0), quat([0, 0, 1], 0.0)),          # both neutral
    (quat([0, 1, 0], np.pi / 2), quat([0, 0, 1], 0.0)),    # gripper points DOWN
    (quat([0, 0, 1], np.pi / 2), quat([0, 1, 0], -0.9)),   # yawed, hand tilted
    (quat([1, 2, -3], 1.1), quat([-2, 1, 0.5], 2.3)),      # nothing aligned
])
def test_ten_cm_out_of_the_operators_jaws_is_ten_cm_out_of_the_robots(
        robot_q, hand_q):
    """THE REQUIREMENT, and the one test the whole mapping exists for.

    Whatever direction either gripper happens to be pointing, a displacement
    along the operator's own out-of-the-jaws axis becomes the same displacement
    along the ROBOT's own out-of-the-jaws axis. That is what makes the mapping
    need no frame calibration: the XR world's orientation cancels between the two
    anchors, so there is no `R_align`, no headset yaw, and nothing that depends
    on where the operator is standing.
    """
    ctrl = pose([0.3, 0.1, 1.2], hand_q)
    flange = pose([0.4, 0.2, 0.9], robot_q)
    a = iv.Anchor.freeze(ctrl, flange, hand_tcp=HAND_TCP)

    # push the controller 10 cm along the OPERATOR's virtual TCP +X
    moved = ctrl.copy()
    moved[:3] = ctrl[:3] + mat_from_quat(a.xr[3:])[:, 0] * 0.10

    now = a.follow(moved)
    in_robot_tool = mat_from_quat(a.robot[3:]).T @ (now[:3] - a.robot[:3])
    assert in_robot_tool == pytest.approx([0.10, 0.0, 0.0], abs=1e-9)


def test_a_zero_delta_commands_exactly_the_pose_that_was_anchored():
    """The property that makes the clutch free: releasing and re-squeezing
    re-freezes both halves, so the first sample after any engage asks for no
    motion at all — however far the operator's hand has drifted meanwhile."""
    ctrl = pose([0.3, 0.1, 1.2], quat([1, 1, 0], 0.7))
    flange = pose([0.4, 0.2, 0.9], quat([0, 1, 1], 1.3))
    a = iv.Anchor.freeze(ctrl, flange, hand_tcp=HAND_TCP)
    assert a.follow(ctrl) == pytest.approx(flange, abs=1e-12)


def test_the_rotation_is_tool_local_not_world_local():
    """Roll your wrist about the controller's own long axis and the gripper rolls
    about ITS own long axis — not about a base axis, and not about the room."""
    hand_q = quat([0, 0, 1], np.pi / 2)          # operator yawed 90 deg
    robot_q = quat([0, 1, 0], np.pi / 2)         # gripper pointing down
    ctrl, flange = pose([0.3, 0, 1.2], hand_q), pose([0.4, 0, 0.9], robot_q)
    a = iv.Anchor.freeze(ctrl, flange, hand_tcp=HAND_TCP)

    twisted = ctrl.copy()
    twisted[3:] = iv.quat_mul(ctrl[3:], quat([1, 0, 0], 0.4))   # about tool +X
    out = a.follow(twisted)

    # the commanded gripper's own +X must have moved by 0.4 rad about its own +X,
    # which means its +X axis is UNCHANGED and the other two have rotated.
    r0, r1 = mat_from_quat(flange[3:]), mat_from_quat(out[3:])
    assert r1[:, 0] == pytest.approx(r0[:, 0], abs=1e-9)
    assert float(np.arccos(np.clip(r1[:, 1] @ r0[:, 1], -1, 1))) == pytest.approx(
        0.4, abs=1e-9)


def test_a_rotation_pivots_at_the_flange():
    """The robot side is the FLANGE (2026-09-28): turning the wrist across the
    tool with the hand held still moves the flange only by what the operator's
    own 5 cm tool tip did, so the Dex-1's fingertips 18.5 cm further out swing
    through the chord of a quarter turn on that arm as well."""
    ctrl, flange = pose([0.3, 0, 1.2], quat([0, 0, 1], 0.0)), \
        pose([0.4, 0, 0.9], quat([0, 0, 1], 0.0))
    a = iv.Anchor.freeze(ctrl, flange, hand_tcp=HAND_TCP)
    tw = ctrl.copy()
    tw[3:] = iv.quat_mul(ctrl[3:], quat([0, 0, 1], np.pi / 2))
    out = a.follow(tw)
    assert np.linalg.norm(out[:3] - flange[:3]) == pytest.approx(
        0.05 * np.sqrt(2), abs=1e-6)
    tip0 = iv.pose_mul(flange, TCP)[:3]
    tip = iv.pose_mul(out, TCP)[:3]
    assert np.linalg.norm(tip - tip0) > 0.26


def test_scale_multiplies_the_translation_and_never_the_rotation():
    """A half-scale correction is half as far, not half as turned: scaling the
    rotation too would make a full turn of the wrist a partial turn of the
    gripper, and the two would then disagree about which way is up."""
    ctrl, flange = pose([0.3, 0, 1.2], quat([0, 1, 0], 0.3)), \
        pose([0.4, 0, 0.9], quat([1, 0, 0], 0.8))
    a = iv.Anchor.freeze(ctrl, flange, hand_tcp=HAND_TCP)
    moved = ctrl.copy()
    moved[:3] += [0.2, 0.0, 0.0]
    moved[3:] = iv.quat_mul(quat([0, 0, 1], 0.5), ctrl[3:])

    full, half = a.follow(moved, 1.0), a.follow(moved, 0.5)
    assert half[3:] == pytest.approx(full[3:], abs=1e-12)
    d_full = full[:3] - a.robot[:3]
    d_half = half[:3] - a.robot[:3]
    assert d_half == pytest.approx(d_full * 0.5, abs=1e-12)


def test_follow_returns_the_flange():
    ctrl, flange = pose([0.3, 0, 1.2], quat([0, 0, 1], 0.0)), \
        pose([0.4, 0, 0.9], quat([0, 0, 1], 0.0))
    a = iv.Anchor.freeze(ctrl, flange, hand_tcp=HAND_TCP)
    moved = ctrl.copy()
    moved[:3] += [0.1, 0.0, 0.0]
    assert a.follow(moved)[:3] == pytest.approx([0.5, 0.0, 0.9], abs=1e-9)


# --- The console ------------------------------------------------------------


def test_the_level_is_the_clutches():
    """The intervention opens when the first hand squeezes and closes
    when the last one lets go, so "somebody is driving" and "some hand is
    engaged" are the same proposition and cannot disagree."""
    c = iv.Console()
    edge, grabbed = c.update([0.0, 0.0])
    assert (edge, list(grabbed), c.active) == (0, [False, False], False)

    edge, grabbed = c.update([0.9, 0.0])
    assert (edge, list(grabbed), c.active) == (1, [True, False], True)

    edge, grabbed = c.update([0.9, 0.9])          # second hand joins
    assert (edge, list(grabbed), c.active) == (0, [False, True], True)

    edge, _ = c.update([0.0, 0.9])                # first lets go, still driving
    assert (edge, c.active) == (0, True)
    assert list(c.engaged) == [False, True]

    edge, _ = c.update([0.0, 0.0])                # last lets go
    assert (edge, c.active) == (-1, False)


def test_a_trigger_already_held_at_construction_cannot_take_over():
    """A controller resting on a bench with something against the trigger, or an
    operator mid-squeeze at a rollout boundary. Entry needs a release first."""
    c = iv.Console()
    for _ in range(5):
        edge, grabbed = c.update([1.0, 1.0])
        assert (edge, list(grabbed), c.active) == (0, [False, False], False)
    c.update([0.0, 0.0])                          # arms on the LEVEL, not an edge
    assert c.update([1.0, 1.0])[0] == 1


def test_the_trigger_has_hysteresis():
    """A finger resting near a single threshold would engage and disengage at the
    poll rate, and every one of those edges re-anchors the arm."""
    c = iv.Console()
    c.update([0.0, 0.0])
    c.update([iv.Console.PRESS, 0.0])
    assert c.engaged[0]
    c.update([0.5, 0.0])                          # between RELEASE and PRESS
    assert c.engaged[0], "let go inside the hysteresis band"
    c.update([iv.Console.RELEASE - 1e-9, 0.0])
    assert not c.engaged[0]


def test_release_all_speaks_the_same_edge_vocabulary_as_update():
    """So P11 has one handback path rather than two."""
    c = iv.Console()
    assert c.release_all() == 0
    c.update([0.0, 0.0])
    c.update([0.9, 0.0])
    assert c.release_all() == -1
    assert not c.active and not c._armed.any()


def test_a_nan_trigger_reads_as_released():
    """A garbled lever holds the arm, which is the same answer a lifted finger
    gives — never an exception on the poll path."""
    c = iv.Console()
    c.update([0.0, 0.0])
    c.update([0.9, 0.9])
    c.update([np.nan, np.nan])
    assert not c.active


# --- The gripper ------------------------------------------------------------


def gripper(**kw) -> iv.Gripper:
    kw.setdefault("rate", 3.0)
    return iv.Gripper(lo=0.0, hi=5.4, **kw)


def test_the_stick_integrates_while_it_is_held():
    g = gripper()
    g.engage(3.0)
    g.latch(0.0)                                  # arm it
    for _ in range(5):
        cmd = g.step(-1.0, 0.1)                   # down = open, up_closes
    assert cmd == pytest.approx(3.0 + 5 * 0.1 * 3.0, abs=1e-9)
    held = g.step(0.0, 0.1)
    assert held == pytest.approx(cmd), "neutral holds where it got to"


def test_up_closes_by_default_and_zero_is_closed():
    g = gripper()
    g.engage(3.0)
    g.latch(0.0)
    assert g.step(1.0, 0.1) < 3.0


def test_the_offset_is_clamped_not_the_target():
    """The one that decides whether it feels broken. Clamping only the target
    lets the offset wind past the stop, and then pushing back does nothing at all
    for as long as it takes to unwind."""
    g = gripper()
    g.engage(0.2)
    g.latch(0.0)
    for _ in range(50):                           # hold "close" for five seconds
        g.step(1.0, 0.1)
    assert g.offset == pytest.approx(-0.2)        # clamped to the travel there is
    # Reversal costs one sample in the dead zone — the latch is three-state, so a
    # stick thrown from one stop to the other passes through neutral. At 90 Hz
    # that is 11 ms, against the seconds an unwinding offset would have cost.
    assert g.step(-1.0, 0.05) == pytest.approx(0.0)
    assert g.step(-1.0, 0.05) > 0.0, "and moves on the one after"


def test_the_full_travel_is_reachable_from_any_anchor():
    """Engaging mid-squeeze must not cost travel — which a proportional map
    would have halved."""
    for anchor in (0.0, 0.3, 2.7, 5.4):
        g = gripper()
        g.engage(anchor)
        g.latch(0.0)
        for _ in range(100):
            lo = g.step(1.0, 0.1)
        for _ in range(200):
            hi = g.step(-1.0, 0.1)
        assert (lo, hi) == pytest.approx((0.0, 5.4), abs=1e-9)


def test_the_squeeze_margin_bounds_the_command_below_the_measured_jaw():
    """THE FORCE BOUND. elrobot_deploy publishes jaw position straight to the wire at
    kp = 10 with no leash, and a held stick is an integrator — so without this an
    operator can wind the command to 0 against an object stopped at 1.4 rad and
    ask for 14 N·m."""
    g = gripper(squeeze_margin=0.6)
    g.engage(1.4)
    g.latch(0.0)
    for _ in range(50):
        cmd = g.step(1.0, 0.1, measured=1.4)      # the jaw is stopped by an object
    assert cmd == pytest.approx(0.8, abs=1e-9)

    loose = gripper(squeeze_margin=np.inf)
    loose.engage(1.4)
    loose.latch(0.0)
    for _ in range(50):
        cmd = loose.step(1.0, 0.1, measured=1.4)
    assert cmd == pytest.approx(0.0), "the bound is what stops it, nothing else"


def test_the_squeeze_bound_yields_as_the_jaw_actually_closes():
    """A floor relative to the MEASUREMENT rather than a limit on the offset, so
    it tracks the jaw down and never becomes a dead zone."""
    g = gripper(squeeze_margin=0.6)
    g.engage(4.0)
    g.latch(0.0)
    for measured in (4.0, 3.0, 2.0, 1.0):
        cmd = g.step(1.0, 0.05, measured=measured)
        assert cmd >= measured - 0.6 - 1e-9


def test_a_deflected_stick_at_the_squeeze_does_not_drive_the_jaw():
    """The stick is a VELOCITY control, so unlike the trigger it must see neutral
    once after each engage. A thumb already pushing would otherwise move the jaw
    the instant the operator took hold."""
    g = gripper()
    g.engage(3.0)
    for _ in range(10):
        assert g.step(1.0, 0.1) == pytest.approx(3.0)
    g.step(0.0, 0.1)                              # neutral arms it
    assert g.step(1.0, 0.1) < 3.0


def test_the_stick_has_a_dead_zone_and_hysteresis_in_both_directions():
    g = gripper()
    g.engage(3.0)
    g.latch(0.0)
    assert g.latch(0.4) == 0, "inside the dead zone"
    assert g.latch(iv.Gripper.PRESS) == 1
    assert g.latch(0.3) == 1, "held inside the hysteresis band"
    assert g.latch(iv.Gripper.RELEASE - 1e-9) == 0
    assert g.latch(-iv.Gripper.PRESS) == -1
    assert g.latch(-0.3) == -1
    assert g.latch(0.0) == 0


def test_hold_freezes_where_the_operator_left_it():
    g = gripper()
    g.engage(3.0)
    g.latch(0.0)
    g.step(-1.0, 0.2)
    assert g.hold() == pytest.approx(3.6)
    assert g.hold() == pytest.approx(3.6), "and does not drift on repetition"


# --- The clock --------------------------------------------------------------


def test_the_lead_is_what_keeps_targets_out_of_the_past():
    """Without it the margin on every target is `L_1 - L_n` — the first
    sample's total latency minus this one's — which is a coin flip, so about half
    of all targets are published already expired and P7 discards them."""
    p = iv.Pace(0.035)
    assert p.due(100.0, 1000.0) == pytest.approx(1000.035)
    # a later sample whose transport was 20 ms slower still lands in the future
    assert p.due(100.020, 1000.040) == pytest.approx(1000.055)


def test_the_pace_is_the_devices_and_not_the_arrivals():
    """The headset publishes over WiFi, and a network hop delivers in BURSTS —
    two samples a millisecond apart carrying 22 ms of hand motion between them.
    Arrival-timed deadlines would ask the arm to perform that 22 ms of travel in
    1 ms, and P7 would faithfully attempt it."""
    p = iv.Pace(0.035)
    p.due(0.0, 1000.0)
    a = p.due(0.011, 1000.030)                    # burst: 1 ms apart on arrival
    b = p.due(0.022, 1000.031)
    assert b - a == pytest.approx(0.011), "device spacing survived intact"


def test_a_deadline_that_would_land_in_the_past_slides_the_whole_map():
    """The backstop that makes the map unable to die. Without it a first sample
    that happened to be fast, or minutes of clock drift, walks every deadline
    into the past and the arm stops responding for the rest of the intervention
    with nothing reporting why."""
    p = iv.Pace(0.035)
    p.due(0.0, 1000.0)                            # anchored: due = 1000.035
    late = p.due(0.010, 1000.100)                 # 55 ms later than the map says
    assert late == pytest.approx(1000.100 + iv.Pace.MIN_MARGIN_S)
    # and the margin is bought back permanently, not re-paid every sample
    assert p.due(0.021, 1000.111) == pytest.approx(late + 0.011)


def test_the_slide_does_not_fire_on_ordinary_jitter():
    """It must be much smaller than the lead. Sliding whenever the margin fell
    short of the FULL lead would fire on every sample slower than the first —
    half of them at steady state — and the lag would ratchet up to the worst
    transport delay ever seen."""
    p = iv.Pace(0.035)
    p.due(0.0, 1000.0)
    before = p.t_anchor_local
    for i in range(1, 40):                        # +-10 ms of arrival jitter
        p.due(i * 0.011, 1000.0 + i * 0.011 + (0.010 if i % 2 else -0.010))
    assert p.t_anchor_local == pytest.approx(before)


def test_reset_forgets_the_map_so_the_next_takeover_re_anchors():
    p = iv.Pace(0.035)
    p.due(0.0, 1000.0)
    p.reset()
    assert p.due(500.0, 2000.0) == pytest.approx(2000.035)


# --- The counter ------------------------------------------------------------


@pytest.mark.parametrize("counter, driving", [
    (0, False), (1, True), (2, False), (3, True), (98, False), (99, True)])
def test_parity_is_the_level(counter, driving):
    assert iv.is_intervening(counter) is driving


def test_a_degenerate_quaternion_is_refused_rather_than_normalised_to_something():
    """The one caller is P11's vendor boundary, where this means the SDK returned
    nothing and the right answer is to drop the sample."""
    with pytest.raises(ValueError):
        iv.quat_normalise([0.0, 0.0, 0.0, 0.0])


def test_pose_mul_and_pose_inv_are_inverses():
    rng = np.random.default_rng(7)
    for _ in range(20):
        a = np.concatenate([rng.normal(size=3),
                            iv.quat_normalise(rng.normal(size=4))])
        assert iv.pose_mul(a, iv.pose_inv(a)) == pytest.approx(
            iv.IDENTITY_POSE, abs=1e-12) or iv.pose_mul(
            a, iv.pose_inv(a))[:3] == pytest.approx([0, 0, 0], abs=1e-12)
        assert len(a) == schema.POSE_DIM


# --- The two tool offsets ----------------------------------------------------
#
# These are the only two numbers the mapping depends on, they are measured
# rather than chosen, and getting either wrong produces an
# arm that moves confidently in the wrong direction rather than an error. So
# the shipped defaults are pinned here.


def test_the_robot_tool_offset_is_exp_004s_measured_one():
    """`FLANGE_TO_TCP_X = 0.184759`, identity rotation — measured off a
    MuJoCo model of the gripper at three arm poses, decomposing as
    0.041500 (wrist_yaw -> dex1_base_link) + 0.143259 (-> far-tip midpoint).

    It is the Dex-1's, so it lives in the robot file (`[end_effector] tcp`)
    and reaches the IK as `Hardware.tcp`; the model constant is the same
    number."""
    from elrobot_deploy.robot import g1d_model as g1d

    tcp = cfgmod.load_hardware("g1d_fisheye_dex1").tcp
    assert tcp.shape == (schema.POSE_DIM,) and tcp.dtype == np.float64
    assert tcp[:3] == pytest.approx([0.184759, 0.0, 0.0])
    assert tcp[3:] == pytest.approx([1.0, 0.0, 0.0, 0.0]), "identity rotation"
    assert tcp[0] == g1d.FLANGE_TO_TCP_X


def test_the_operator_tool_offset_is_per_hand_and_not_identity():
    """**Both halves of this matter, and both are measured.**

    The rotation is 162 degrees from identity, so a default of identity would be
    wrong by almost a half-turn — and the two hands differ from each other by
    12.5 degrees, so one shared value would be wrong for at least one hand by
    that much. Neither failure raises anything; both are an arm that moves
    confidently in a direction the operator did not ask for.
    """
    h = np.asarray(cfgmod.DeployConfig.intervene_hand_tcp, dtype=np.float64)
    assert h.shape == (schema.N_HANDS, schema.POSE_DIM), "per hand"
    assert np.linalg.norm(h[:, 3:], axis=1) == pytest.approx(1.0), "unit quats"

    def angle(a, b):
        r = mat_from_quat(a) @ mat_from_quat(b).T
        return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))

    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert angle(h[0, 3:], ident) == pytest.approx(162.2, abs=0.5)
    assert angle(h[0, 3:], h[1, 3:]) == pytest.approx(12.5, abs=0.5)
    # ~18.4 cm out of the controller, both hands
    assert np.linalg.norm(h[:, :3], axis=1) == pytest.approx(0.1838, abs=0.001)


def test_each_hand_gets_its_own_offset():
    """P11 indexes `hand_tcp` per hand, and the cost of not doing so is measured
    here rather than asserted in the abstract.

    The tool ROTATION does not merely decide where a wrist twist pivots — it sets
    the direction a straight push travels, because `R_A` is the frame the
    operator's displacement is expressed in before `R_C` puts it back out. So
    pushing the controller 10 cm straight along XR +x drives the arm:

        left offset      [ 0.75, -9.93, -0.87] cm
        right offset     [-0.71, -9.95,  0.73] cm      12.4 deg apart, 2.17 cm
        identity         [10.00,  0.00,  0.00] cm      85.7 deg off

    Identity is not "slightly uncalibrated"; it is very nearly perpendicular, and
    it would show up as an arm that moves confidently in the wrong direction with
    nothing anywhere reporting an error.
    """
    h = np.asarray(cfgmod.DeployConfig.intervene_hand_tcp, dtype=np.float64)
    ctrl = pose([0.3, 0.1, 1.2], quat([0, 0, 1], 0.0))
    flange = pose([0.4, 0.2, 0.9], quat([0, 0, 1], 0.0))
    moved = ctrl.copy()
    moved[:3] += [0.10, 0.0, 0.0]

    def travel(hand_tcp):
        a = iv.Anchor.freeze(ctrl, flange, hand_tcp=hand_tcp)
        return a.follow(moved)[:3] - flange[:3]

    def between(u, v):
        return float(np.degrees(np.arccos(np.clip(
            u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))

    left, right = travel(h[0]), travel(h[1])
    assert between(left, right) == pytest.approx(12.4, abs=0.5)
    assert np.linalg.norm(left - right) == pytest.approx(0.0217, abs=0.001)
    assert between(left, travel(iv.IDENTITY_POSE)) == pytest.approx(85.7, abs=1.0)
