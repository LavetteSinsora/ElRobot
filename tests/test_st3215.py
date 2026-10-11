import pytest

from elrobot import probe
from elrobot.st3215 import protocol
from elrobot.st3215 import registers as R
from elrobot.st3215.bus import Bus, BusError

from fake_bus import FakeSerial, FakeServo


def make_bus(ids=range(1, 9), echo=False):
    ser = FakeSerial([FakeServo(i, position=1000 + 100 * i) for i in ids], echo=echo)
    return Bus(serial_port=ser, timeout=0.01), ser


def test_ping_packet_matches_datasheet_example():
    assert protocol.ping(1) == bytes.fromhex("ffff010201fb")


def test_parse_status_skips_garbage_bad_checksum_and_padding():
    good = bytes.fromhex("ffff0102" "00" "fc")
    bad = bytes.fromhex("ffff0102" "00" "00")
    buf = bytearray(b"\x00\x13" + bad + b"\xff" + good)
    status = protocol.parse_status(buf)
    assert (status.servo_id, status.error, status.params) == (1, 0, b"")
    assert buf == b""


def test_parse_status_waits_for_partial_packet():
    packet = bytes.fromhex("ffff01040000085e")[:-1]
    packet += bytes([protocol.checksum(packet[2:])])
    buf = bytearray(packet[:5])
    assert protocol.parse_status(buf) is None
    buf += packet[5:]
    assert protocol.parse_status(buf).params == b"\x00\x08"


def test_sign_magnitude_registers():
    assert R.OFFSET.encode(-5) == bytes([0x05, 0x08])
    assert R.OFFSET.decode(bytes([0x05, 0x08])) == -5
    assert R.OFFSET.decode(R.OFFSET.encode(2047)) == 2047
    with pytest.raises(ValueError):
        R.OFFSET.encode(2048)
    assert R.PRESENT_LOAD.decode((1 << 10 | 37).to_bytes(2, "little")) == -37


@pytest.mark.parametrize("echo", [False, True])
def test_single_and_group_operations(echo):
    bus, ser = make_bus(echo=echo)
    assert bus.ping(3) and not bus.ping(9)

    table = bus.read_table(4)
    assert table["model"] == R.STS3215_MODEL
    assert table["present_position"] == 1400
    assert table["max_position_limit"] == 4095

    bus.write_reg(2, R.GOAL_POSITION, 1234)
    assert ser.servos[2].get(R.GOAL_POSITION) == 1234

    assert bus.sync_read_reg(list(range(1, 9)), R.PRESENT_POSITION) == {
        i: 1000 + 100 * i for i in range(1, 9)}

    bus.sync_write_reg(R.GOAL_POSITION, {1: 111, 8: 888})
    assert ser.servos[1].get(R.GOAL_POSITION) == 111
    assert ser.servos[8].get(R.GOAL_POSITION) == 888
    # A reply-less sync write must not confuse the next transaction.
    assert bus.read_reg(5, R.PRESENT_POSITION) == 1500


def test_sync_read_names_missing_servos():
    bus, _ = make_bus(ids=[1, 2, 4])
    with pytest.raises(BusError, match=r"\[3\]"):
        bus.sync_read_reg([1, 2, 3, 4], R.PRESENT_POSITION)


def test_alarm_flags_are_recorded_not_raised():
    bus, ser = make_bus()
    ser.servos[6].alarm = 1 << 5
    assert bus.read_reg(6, R.PRESENT_POSITION) == 1600
    assert R.status_flags(bus.alarms[6]) == ["overload"]


def test_probe_is_read_only(capsys):
    bus, ser = make_bus(ids=[1, 2, 3, 4, 5, 6, 8])
    ser.servos[3].set(R.TORQUE_ENABLE, 1)
    found = probe.report(bus, list(range(1, 9)))
    out = capsys.readouterr().out
    assert found == [1, 2, 3, 4, 5, 6, 8]
    assert "MISSING: [7]" in out
    assert "torque_enable = 1 (factory 0)" in out
    assert all(p[4] in (protocol.PING, protocol.READ) for p in ser.sent)


def test_range_tracker_follows_motion_across_the_wrap():
    t = probe.RangeTracker()
    for p in [3800, 3950, 4090, 20, 300, 150, 3900, 3500, 3200]:
        t.update(p)
    assert t.ends == (3200, 300)
    assert t.span == 300 + 4096 - 3200
    assert t.crosses_wrap
    assert t.midpoint == (3200 + t.span // 2) % 4096


def test_range_tracker_without_wrap():
    t = probe.RangeTracker()
    for p in [2000, 1500, 2600, 2100]:
        t.update(p)
    assert (t.ends, t.span, t.midpoint, t.crosses_wrap) == ((1500, 2600), 1100, 2050, False)
