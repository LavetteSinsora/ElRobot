"""The teleop arithmetic (`runtime/teleop/intervention.py`): clutch, room frame, gripper, pace."""

from __future__ import annotations

import math

import numpy as np
import pytest

from elrobot_deploy.robot.transforms import mat_from_quat, quat_angle
from elrobot_deploy.runtime.teleop.intervention import (
    Console,
    Gripper,
    HeadingUndefined,
    Pace,
    RoomFrame,
    apply_delta,
    is_intervening,
)


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def pose(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


#: The user's grip, in ARKit's world (X right, Y up, -Z ahead when the session started facing
#: forward): the phone's top (device -X) points forward (-Z), its screen (+Z) faces left (-X),
#: so device +Y points up. Columns = the device axes in world coordinates.
GRIP = np.c_[[0, 0, 1], [0, 1, 0], [-1, 0, 0]].astype(float)


def room(**kw):
    kw = {"phone_forward": (-1, 0, 0), "base_forward": (0, 1, 0), "base_up": (0, 0, 1)} | kw
    return RoomFrame(kw.pop("phone_forward"), kw.pop("base_forward"), kw.pop("base_up"), **kw)


def test_the_grip_matrix_is_a_rotation():
    assert np.allclose(GRIP.T @ GRIP, np.eye(3)) and np.linalg.det(GRIP) == pytest.approx(1)
    assert np.allclose(GRIP @ [-1, 0, 0], [0, 0, -1]), "top forward"
    assert np.allclose(GRIP @ [0, 0, 1], [-1, 0, 0]), "screen left"
    assert np.allclose(GRIP @ [0, 1, 0], [0, 1, 0]), "device +Y up"


@pytest.mark.parametrize("move, expect", [
    ((0, 0, -0.1), (0, 0.1, 0)),       # forward in the room -> base +y (forward)
    ((-0.1, 0, 0), (-0.1, 0, 0)),      # left -> base -x (left)
    ((0, 0.1, 0), (0, 0, 0.1)),        # up -> base +z
])
def test_moving_the_phone_moves_the_tcp_the_same_way(move, expect):
    rf = room()
    rf.freeze(pose(GRIP, [0.3, 1.2, -0.5]))
    d = rf.delta(pose(GRIP, np.add([0.3, 1.2, -0.5], move)))
    assert np.allclose(d[:3], expect, atol=1e-12)
    assert quat_angle(d[3:]) == pytest.approx(0.0, abs=1e-9)


def test_forward_is_wherever_the_operator_faces_at_the_press():
    """The ARKit world's heading is arbitrary (where the session started): turn the operator
    and the phone 90 deg left together and "forward" is still the robot's forward."""
    turn = rot([0, 1, 0], 90)
    rf = room()
    rf.freeze(pose(turn @ GRIP, [0, 0, 0]))
    ahead = turn @ np.array([0, 0, -0.1])
    assert np.allclose(rf.delta(pose(turn @ GRIP, ahead))[:3], [0, 0.1, 0], atol=1e-12)
    assert np.allclose(rf.axes(pose(turn @ GRIP, ahead)), [0.1, 0, 0], atol=1e-12)


def test_the_heading_ignores_how_the_phone_is_tilted_at_the_press():
    tilted = rot([1, 0, 0], -30) @ GRIP          # top pitched 30 deg down, still facing ahead
    rf = room()
    rf.freeze(pose(tilted, [0, 0, 0]))
    assert np.allclose(rf.delta(pose(tilted, [0, 0, -0.1]))[:3], [0, 0.1, 0], atol=1e-12)


def test_a_phone_pointing_straight_up_has_no_heading():
    up = rot([1, 0, 0], 90) @ GRIP               # the top now points up
    assert np.allclose(up @ [-1, 0, 0], [0, 1, 0])
    with pytest.raises(HeadingUndefined):
        room().freeze(pose(up, [0, 0, 0]))


def test_rotations_are_room_frame():
    rf = room()
    rf.freeze(pose(GRIP, [0, 0, 0]))
    # Tilting the top down (about the operator's left axis, world -X... i.e. right-hand about
    # +X by -20 deg) pitches the TCP about the base's left axis (-x) the same way.
    d = rf.delta(pose(rot([1, 0, 0], -20) @ GRIP, [0, 0, 0]))
    R = mat_from_quat(d[3:])
    assert np.allclose(R, rot([1, 0, 0], -20), atol=1e-9)
    assert np.allclose(R @ [0, 1, 0], [0, math.cos(math.radians(20)),
                                       -math.sin(math.radians(20))], atol=1e-9), \
        "forward tips down"
    # Turning left (about up) yaws the TCP left about base +z.
    d = rf.delta(pose(rot([0, 1, 0], 30) @ GRIP, [0, 0, 0]))
    assert np.allclose(mat_from_quat(d[3:]), rot([0, 0, 1], 30), atol=1e-9)


def test_scale_applies_to_translation_only():
    rf = room(scale=0.5)
    rf.freeze(pose(GRIP, [0, 0, 0]))
    d = rf.delta(pose(rot([0, 1, 0], 30) @ GRIP, [0, 0, -0.2]))
    assert np.allclose(d[:3], [0, 0.1, 0], atol=1e-12)
    assert quat_angle(d[3:]) == pytest.approx(math.radians(30))


def test_apply_delta_moves_and_rotates_about_the_tcp():
    tcp = np.r_[0.1, 0.2, 0.3, 1.0, 0, 0, 0]
    q90 = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    out = apply_delta(tcp, np.r_[0.01, 0, 0, q90])
    assert np.allclose(out[:3], [0.11, 0.2, 0.3])
    assert np.allclose(mat_from_quat(out[3:]), rot([0, 0, 1], 90))


# --- the clutch -----------------------------------------------------------------------------


def test_a_button_held_at_startup_does_not_engage_until_seen_released():
    c = Console()
    assert c.update(True) == 0 and not c.engaged
    assert c.update(False) == 0
    assert c.update(True) == 1 and c.engaged
    assert c.update(True) == 0
    assert c.update(False) == -1 and not c.engaged


def test_a_forced_release_needs_a_fresh_press():
    c = Console()
    c.update(False)
    c.update(True)
    assert c.release() == -1
    assert c.update(True) == 0 and not c.engaged, "still held: no re-engage"
    c.update(False)
    assert c.update(True) == 1


def test_parity():
    assert not is_intervening(0) and is_intervening(1) and not is_intervening(2)


# --- the gripper ----------------------------------------------------------------------------


def test_hold_to_close_on_device_time_and_open_on_release():
    g = Gripper(rate=1.0, lo=0.05, hi=2.4)
    g.engage(0.05)
    assert g.step(True, 0.05) == pytest.approx(0.10)
    assert g.step(True, 0.05) == pytest.approx(0.15)
    assert g.step(False, 0.025) == pytest.approx(0.125)
    assert g.step(True, 10.0) == pytest.approx(0.125 + Gripper.MAX_DT), "a gap is capped"


def test_the_squeeze_bound_is_stored_so_release_opens_at_once():
    g = Gripper(rate=1.0, lo=0.05, hi=2.4, squeeze_margin=0.1)
    g.engage(0.5)
    for _ in range(50):                          # held closed on an object stopped at 0.6
        g.step(True, 0.1, measured=0.6)
    assert g.cmd == pytest.approx(0.7)
    assert g.step(False, 0.1, measured=0.6) == pytest.approx(0.6), "opens on the first step"


def test_engage_continues_from_the_command_and_clips():
    g = Gripper(rate=1.0, lo=0.05, hi=2.4)
    g.engage(9.0)
    assert g.cmd == 2.4


# --- the clock ------------------------------------------------------------------------------


def test_pace_replays_at_device_pace_with_the_lead():
    p = Pace(0.04)
    assert p.due(5.0, 100.0) == pytest.approx(100.04)
    assert p.due(5.1, 100.07) == pytest.approx(100.14), "arrival jitter does not matter"


def test_pace_slides_rather_than_publishing_the_past():
    p = Pace(0.04)
    p.due(5.0, 100.0)
    due = p.due(5.01, 100.2)                     # 190 ms late
    assert due == pytest.approx(100.2 + Pace.MIN_MARGIN_S)
    assert p.slid > 0.1
    assert p.due(5.02, 100.201) == pytest.approx(due + 0.01), "the slide is kept"
    p.reset()
    assert p.due(6.0, 200.0) == pytest.approx(200.04)
