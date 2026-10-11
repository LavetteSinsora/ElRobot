"""Offline tests for servo_id.py against a simulated ST3215 bus (no hardware needed).

    python -m pytest test_servo_id.py      or      python test_servo_id.py
"""

import servo_id as s


class FakeServo:
    def __init__(self, sid=1, baud_code=0):
        self.mem = bytearray(0x50)
        self.mem[s.REG_MODEL:s.REG_MODEL + 2] = s.STS3215_MODEL.to_bytes(2, "little")
        self.mem[s.REG_ID] = sid
        self.mem[s.REG_BAUD] = baud_code
        self.mem[s.REG_LOCK] = 1
        self.mem[s.REG_PRESENT_POS:s.REG_PRESENT_POS + 2] = (2048).to_bytes(2, "little")
        self.mem[s.REG_VOLTAGE] = 120
        self.mem[s.REG_TEMP] = 30
        self.eeprom = bytes(self.mem[:0x28])   # what survives a power cycle

    @property
    def sid(self):
        return self.mem[s.REG_ID]

    @property
    def baud(self):
        return s.BAUD_TABLE[self.mem[s.REG_BAUD]]

    def handle(self, pkt):
        sid, instr, params = pkt[2], pkt[4], pkt[5:-1]
        if sid not in (self.sid, s.BROADCAST_ID):
            return b""
        reply_id = self.sid
        out = b""
        if instr == s.INST_READ:
            a, n = params
            out = bytes(self.mem[a:a + n])
        elif instr == s.INST_WRITE:
            a, data = params[0], params[1:]
            self.mem[a:a + len(data)] = data
            if a < 0x28 and self.mem[s.REG_LOCK] == 0:
                self.eeprom = bytes(self.mem[:0x28])
        body = bytes([reply_id, len(out) + 2, 0]) + out
        return b"\xFF\xFF" + body + bytes([s.checksum(body)])

    def power_cycle(self):
        self.mem[:0x28] = self.eeprom
        self.mem[s.REG_LOCK] = 1


class FakePort:
    """pyserial stand-in. `echo` mimics adapters that loop TX back to RX."""

    def __init__(self, servos, echo=False):
        self.servos, self.echo = servos, echo
        self.baudrate = s.TARGET_BAUD
        self.rx = b""

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, n):
        out, self.rx = self.rx[:n], self.rx[n:]
        return out

    def write(self, pkt):
        if self.echo:
            self.rx += pkt
        for sv in self.servos:
            if sv.baud == self.baudrate:
                self.rx += sv.handle(pkt)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self.rx = b""


def test_checksum_matches_feetech_example():
    # Feetech manual: ping ID 1 -> FF FF 01 02 01 FB
    assert s.build_packet(1, s.INST_PING) == bytes.fromhex("FFFF010201FB")


def test_set_id_persists_through_power_cycle():
    sv = FakeServo(sid=1)
    bus = s.Bus(FakePort([sv]), timeout=0.002)
    s.set_id(bus, 5, assume_yes=True)
    sv.power_cycle()
    assert sv.sid == 5 and sv.mem[s.REG_LOCK] == 1
    assert s.scan_ids(bus) == [5]


def test_set_id_with_echoing_adapter():
    sv = FakeServo(sid=1)
    s.set_id(s.Bus(FakePort([sv], echo=True), timeout=0.002), 3, assume_yes=True)
    assert sv.sid == 3


def test_set_id_fixes_wrong_baud():
    sv = FakeServo(sid=1, baud_code=4)       # 115200
    s.set_id(s.Bus(FakePort([sv]), timeout=0.002), 7, assume_yes=True)
    sv.power_cycle()
    assert sv.sid == 7 and sv.baud == s.TARGET_BAUD


def test_refuses_with_two_servos():
    bus = s.Bus(FakePort([FakeServo(1), FakeServo(2)]), timeout=0.002)
    try:
        s.set_id(bus, 4, assume_yes=True)
    except s.ServoError as e:
        assert "More than one" in str(e)
    else:
        raise AssertionError("should refuse")


def test_scan_daisy_chain():
    bus = s.Bus(FakePort([FakeServo(i) for i in range(1, 9)]), timeout=0.002)
    assert s.scan_ids(bus, range(0, 21)) == list(range(1, 9))
    assert "gripper" in s.describe(bus, 8)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
