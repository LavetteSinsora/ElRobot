import json
import math

import pytest

from elrobot import find_range as F
from elrobot import precentre as PC
from elrobot import relax as RX
from elrobot.st3215 import protocol
from elrobot.st3215 import registers as R
from elrobot.st3215.bus import Bus

from fake_bus import FakeSerial, FakeServo, SimClock

GRIPPER_WIDTH = 2.2028  # rad, from the URDF
TWO_STOPS = F.Profile((150, 220, 300), "test joint with two hard stops")


@pytest.fixture
def rig(tmp_path):
    def make(stops, start, joint=8, others=(1, 2), width=GRIPPER_WIDTH, profile=TWO_STOPS,
             other_pos=1000, **kw):
        clock = SimClock()
        target = FakeServo(joint, position=start, stops=stops, clock=clock, **kw)
        servos = [target] + [FakeServo(i, position=other_pos + i, clock=clock) for i in others]
        ser = FakeSerial(servos)
        bus = Bus(serial_port=ser, timeout=0.01)
        finder = F.RangeFinder(bus, joint, width, hold=list(others), profile=profile, workdir=tmp_path,
                               clock=clock.now, sleep=clock.sleep, log=lambda *a: None)
        return finder, target, ser
    return make


def close(a, b, tol=F.STEP):
    return abs((a - b + 2048) % 4096 - 2048) <= tol


def width_for(stops):
    return (stops[1] - stops[0]) * 2 * math.pi / 4096


@pytest.mark.parametrize("stops,start", [
    ((1500, 2900), 2200),     # plain range
    ((3500, 4800), 3700),     # crosses the wrap in factory coordinates (3500 .. 4095, 0 .. 704)
    ((1500, 2900), 1510),     # starts right next to a stop
])
def test_finds_both_stops(rig, stops, start):
    finder, servo, _ = rig(stops, start, width=width_for(stops))
    lo, hi = finder.run()
    assert (lo.kind, hi.kind) == ("stop", "stop")
    assert close(lo.encoder, stops[0] % 4096) and close(hi.encoder, stops[1] % 4096)
    r = F.summarize(8, lo, hi, width_for(stops))
    assert abs(r["span_steps"] - (stops[1] - stops[0])) <= 2 * F.STEP
    assert r["problems"] == []


def test_friction_is_pushed_through_not_mistaken_for_a_stop(rig):
    """The first real gripper run stopped mid-travel. A sticky stretch that needs 22 % torque
    must be passed by escalating, and the torque must drop back to 15 % afterwards."""
    stops = (1500, 2900)
    finder, servo, _ = rig(stops, 2700, width=width_for(stops), friction=(2000, 2100, 220))
    lo, hi = finder.run()
    assert close(lo.encoder, 1500) and close(hi.encoder, 2900)
    log = finder.log_path.read_text()
    assert "stall, torque limit -> 220" in log and "clear, back to 150" in log


def test_stop_needs_the_highest_torque_level(rig):
    finder, servo, _ = rig((1500, 2900), 2200, width=width_for((1500, 2900)))
    lo, hi = finder.run()
    assert lo.torque_limit == hi.torque_limit == TWO_STOPS.torque_levels[-1]


def test_write_offset_centres_range_on_2048(rig):
    stops = (3500, 4800)
    finder, servo, _ = rig(stops, 3700, width=width_for(stops))
    lo, hi = finder.run()
    r = F.summarize(8, lo, hi, width_for(stops))
    finder.release(r["centring_offset"])
    servo.u = (stops[0] + stops[1]) / 2
    servo.step()
    assert abs(servo.get(R.PRESENT_POSITION) - 2048) <= F.STEP
    assert servo.get(R.TORQUE_ENABLE) == 0


def test_release_restores_settings_and_turns_torque_off(rig, tmp_path):
    finder, servo, ser = rig((1500, 2900), 2200)
    servo.set(R.OFFSET, 85)
    servo.set(R.PROTECTION_CURRENT, 310)
    servo.set(R.TORQUE_LIMIT, 1000)
    finder.prepare()
    assert json.loads((tmp_path / "restore.json").read_text())["offset"] == 85
    finder.search(-1)
    finder.release(None)
    assert servo.get(R.OFFSET) == 85
    assert servo.get(R.PROTECTION_CURRENT) == 310  # never touched
    assert servo.get(R.TORQUE_LIMIT) == 1000
    assert all(s.get(R.TORQUE_ENABLE) == 0 for s in ser.servos.values())
    assert not (tmp_path / "restore.json").exists()


def test_relax_cleans_up_after_a_run_that_died(rig, tmp_path):
    finder, servo, ser = rig((1500, 2900), 2200)
    servo.set(R.OFFSET, 85)
    finder.prepare()
    finder.search(-1)  # ...and the process is killed here: temporary offset, torque on
    assert servo.get(R.OFFSET) != 85 and servo.get(R.TORQUE_ENABLE) == 1
    RX.relax(finder.bus, [1, 2, 8], restore_path=tmp_path / "restore.json", log=lambda *a: None)
    assert servo.get(R.OFFSET) == 85
    assert all(s.get(R.TORQUE_ENABLE) == 0 for s in ser.servos.values())
    assert not (tmp_path / "restore.json").exists()


def test_refuses_to_start_while_a_previous_run_is_unfinished(rig, tmp_path):
    (tmp_path / "restore.json").write_text("{}")
    finder, servo, ser = rig((1500, 2900), 2200)
    with pytest.raises(F.Abort, match="relax"):
        finder.prepare()
    finder.release(None)
    assert (tmp_path / "restore.json").exists()  # left for relax to use
    assert not any(p[4] == protocol.WRITE for p in ser.sent)


def test_no_stop_stops_at_travel_cap_and_is_flagged(rig):
    finder, servo, _ = rig((-10_000, 10_000), 2000)
    lo, hi = finder.run()
    assert (lo.kind, hi.kind) == ("cap", "cap")
    r = F.summarize(8, lo, hi, GRIPPER_WIDTH)
    assert r["span_deg"] <= math.degrees(GRIPPER_WIDTH) + 15 + 2
    assert any("travel cap" in p for p in r["problems"])


def test_span_far_from_urdf_is_flagged(rig):
    finder, servo, _ = rig((1500, 2000), 1700)  # 44 deg, URDF says 126
    lo, hi = finder.run()
    r = F.summarize(8, lo, hi, GRIPPER_WIDTH)
    assert any("from the URDF" in p for p in r["problems"])


def test_goal_is_set_before_every_torque_enable(rig):
    finder, _, ser = rig((1500, 2900), 2200)
    finder.run()
    goal_written = set()
    for p in ser.sent:
        if p[4] != protocol.WRITE:
            continue
        sid, addr = p[2], p[5]
        if addr == R.GOAL_POSITION.addr:
            goal_written.add(sid)
        if addr == R.TORQUE_ENABLE.addr and p[6] == 1:
            assert sid in goal_written, f"servo {sid} torque on before any goal"
            goal_written.discard(sid)


def test_never_touches_protection_current(rig):
    finder, _, ser = rig((1500, 2900), 2200)
    finder.run()
    finder.release(None)
    assert not any(p[4] == protocol.WRITE and p[5] == R.PROTECTION_CURRENT.addr for p in ser.sent)


def test_wrong_offset_semantics_aborts_before_any_motion(rig):
    finder, servo, ser = rig((1500, 2900), 2200, offset_sign=-1)
    with pytest.raises(F.Abort, match="offset check failed"):
        finder.run()
    assert servo.get(R.TORQUE_ENABLE) == 0
    finder.release(None)
    assert servo.get(R.OFFSET) == 0


def test_refuses_nonfactory_limits_and_existing_alarm(rig):
    finder, servo, _ = rig((1500, 2900), 2200)
    servo.set(R.MIN_POSITION_LIMIT, 500)
    with pytest.raises(F.Abort, match="position limits"):
        finder.prepare()
    finder, servo, _ = rig((1500, 2900), 2200)
    servo.alarm = 1 << 5
    with pytest.raises(F.Abort, match="overload"):
        finder.prepare()


def test_naive_goal_across_the_wrap_turns_the_long_way_into_the_stop():
    """Why the search re-offsets: from 4000, a goal of 100 is only ~200 steps away through
    4095->0, but the controller goes down through 2000 and hits the stop at 3500."""
    clock = SimClock()
    servo = FakeServo(8, position=4000, stops=(3500, 4800), clock=clock)
    servo.set(R.GOAL_SPEED, 500)
    servo.set(R.GOAL_POSITION, 4000)
    servo.set(R.TORQUE_ENABLE, 1)
    servo.set(R.GOAL_POSITION, 100)
    for _ in range(200):
        clock.sleep(0.02)
        servo.step()
    assert servo.u == 3500


def test_squeezing_the_stop_at_top_torque_is_still_the_stop(rig):
    """Real gripper run: at 30 % the closed stop yielded a little, which was read as "it moved",
    and the search carried on to the travel cap."""
    stops = (1500, 2900)
    finder, _, _ = rig(stops, 2200, width=width_for(stops), give=(30, 300))
    lo, hi = finder.run()
    assert (lo.kind, hi.kind) == ("stop", "stop")
    assert close(hi.encoder, 2900, tol=2 * F.STEP)


def test_gripper_reads_its_open_stop_and_only_closes(rig):
    """The gripper rests fully open against its stop (encoder ~3300, readings rise when
    opening). It must never be pushed wider; it closes by the URDF width."""
    width_steps = round(GRIPPER_WIDTH * 4096 / (2 * math.pi))
    finder, servo, _ = rig((3300 - 1510, 3300), 3300, profile=F.PROFILES[8])
    lo, hi = finder.run()
    assert (lo.kind, hi.kind) == ("defined", "rest")
    assert close(hi.encoder, 3300, tol=2)
    assert servo.max_u == 3300  # never moved past where it rested
    r = F.summarize(8, lo, hi, GRIPPER_WIDTH)
    assert abs(r["span_steps"] - width_steps) <= F.STEP
    assert r["problems"] == []


def test_gripper_jaws_meeting_early_is_a_stop_and_flagged(rig):
    finder, servo, _ = rig((3300 - 800, 3300), 3300, profile=F.PROFILES[8])
    lo, hi = finder.run()
    assert lo.kind == "stop" and close(lo.encoder, 2500)
    assert servo.max_u == 3300
    r = F.summarize(8, lo, hi, GRIPPER_WIDTH)
    assert any("from the URDF" in p for p in r["problems"])


def test_wrong_way_motion_turns_torque_off_and_aborts(rig):
    finder, servo, _ = rig((1500, 2900), 2200, wrong_way=True)
    with pytest.raises(F.Abort, match="wrong way"):
        finder.run()
    assert servo.get(R.TORQUE_ENABLE) == 0
    assert "WRONG WAY" in finder.log_path.read_text()


def test_refuses_to_hold_a_joint_near_the_wrap(rig):
    finder, _, ser = rig((1500, 2900), 2200, others=(4,), other_pos=4090 - 4)
    with pytest.raises(F.Abort, match="precentre 4"):
        finder.prepare()
    assert not any(p[4] == protocol.WRITE and p[5] == R.TORQUE_ENABLE.addr for p in ser.sent)


def test_held_joints_get_a_gentle_speed_before_torque_on(rig):
    finder, _, ser = rig((1500, 2900), 2200)
    finder.prepare()
    held = ser.servos[1]
    assert held.get(R.TORQUE_ENABLE) == 1
    assert held.get(R.GOAL_SPEED) == F.HOLD_SPEED and held.get(R.ACCELERATION) == F.HOLD_ACCEL


def test_precentre_moves_the_wrap_half_a_turn_away():
    clock = SimClock()
    servo = FakeServo(4, position=29 + 85, clock=clock)  # raw 114, offset 85 -> reads 29
    servo.set(R.OFFSET, 85)
    servo.step()
    bus = Bus(serial_port=FakeSerial([servo]), timeout=0.01)
    old, new, pos = PC.precentre(bus, 4)
    assert (old, pos) == (85, 2048)
    assert servo.get(R.OFFSET) == new
