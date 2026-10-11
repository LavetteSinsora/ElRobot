import math

import pytest

from elrobot import move as M
from elrobot.calibration import JointCal
from elrobot.poses import POSES
from elrobot.st3215 import registers as R
from elrobot.st3215.bus import Bus

from fake_bus import FakeSerial, FakeServo, SimClock


def cal(j, sign=1, zero=2048, lo=900, hi=3200):
    return JointCal(j, f"j{j}", sign, zero, 0, lo, hi, "test")


def rig(start=None, servo_kw=None):
    servo_kw = servo_kw or {}
    clock = SimClock()
    start = start or {j: 2048 for j in range(1, 9)}
    servos = [FakeServo(j, position=start[j], stops=(0, 4095), clock=clock, **servo_kw.get(j, {}))
              for j in range(1, 9)]
    ser = FakeSerial(servos)
    cals = {j: cal(j) for j in range(1, 9)}
    cals[8] = cal(8, sign=-1, zero=2861, lo=1459, hi=2827)
    mover = M.Mover(Bus(serial_port=ser, timeout=0.01), cals, clock=clock.now, sleep=clock.sleep,
                    log=lambda *a: None)
    return mover, ser


def test_moves_to_rest_and_clamps_the_gripper_off_its_stop():
    mover, ser = rig()
    now, target = mover.plan(POSES["rest"])
    assert target[2] == 2048 - round(math.radians(90) / (2 * math.pi / 4096))
    assert target[8] == 2827  # 0 deg (fully open) would push the stop; clamped 3 deg short
    final = mover.move(target)
    assert all(abs(final[j] - target[j]) <= 3 for j in final)


def test_goal_equals_present_before_torque_on():
    mover, ser = rig()
    _, target = mover.plan(POSES["rest"])
    mover.move(target)
    goal_seen = False
    for p in ser.sent:
        if p[4] == 0x83 and p[5] == R.GOAL_POSITION.addr:
            goal_seen = True
        if p[4] == 0x83 and p[5] == R.TORQUE_ENABLE.addr and p[7] == 1:
            assert goal_seen
            break


def test_runaway_when_torque_comes_on_releases_everything():
    mover, ser = rig(servo_kw={3: {"runaway": +1}})
    _, target = mover.plan(POSES["rest"])
    with pytest.raises(M.MoveAbort, match="on their own"):
        mover.move(target)
    assert all(s.get(R.TORQUE_ENABLE) == 0 for s in ser.servos.values())


def test_wrong_way_joint_is_released_and_others_hold():
    mover, ser = rig(servo_kw={4: {"wrong_way": True}})
    _, target = mover.plan(POSES["rest"])
    with pytest.raises(M.MoveAbort, match="joint 4 moved the wrong way"):
        mover.move(target)
    assert ser.servos[4].get(R.TORQUE_ENABLE) == 0
    assert ser.servos[2].get(R.TORQUE_ENABLE) == 1


def test_blocked_joint_stops_the_move():
    clock_stops = {3: {"friction": (2100, 4095, 9999)}}  # can never pass 2100
    mover, ser = rig(servo_kw=clock_stops)
    pose = dict(POSES["zero"]); pose[3] = 40
    _, target = mover.plan(pose)
    with pytest.raises(M.MoveAbort, match="joint 3 is .* behind"):
        mover.move(target)


def test_refuses_when_offsets_do_not_match():
    mover, ser = rig()
    ser.servos[5].set(R.OFFSET, 85)
    with pytest.raises(M.MoveAbort, match=r"offsets of \[5\]"):
        mover.check_ready()


def test_servo_planned_move_sends_each_target_once_with_scaled_speeds():
    mover, ser = rig()
    _, target = mover.plan(POSES["rest"])
    mover.move(target)
    goal_writes = [p for p in ser.sent if p[4] == 0x83 and p[5] == R.GOAL_POSITION.addr]
    assert len(goal_writes) == 2          # one "hold here" before torque on, one final target
    speeds = {j: ser.servos[j].get(R.GOAL_SPEED) for j in range(1, 9)}
    assert all(v >= 1 for v in speeds.values())   # 0 would mean full speed
    far = max(range(1, 9), key=lambda j: abs(target[j] - 2048))
    assert speeds[far] == M.steps(M.SPEED_DEG)


def test_stream_method_still_available():
    mover, ser = rig()
    _, target = mover.plan(POSES["rest"])
    final = mover.move_stream(target)
    assert all(abs(final[j] - target[j]) <= 3 for j in final)
