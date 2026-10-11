#!/usr/bin/env python3
"""
ElRobot follower: ST3215 servo ID tool (Waveshare Bus Servo Adapter, USB mode).

Every new ST3215 ships as ID 1 at 1 Mbps. ElRobot expects IDs 1..8 at 1 Mbps,
numbered base -> gripper (NormaCore: URDF joints rev_motor_01..08, and the
auto-calibration in software/drivers/st3215/src/auto_calibrate/elrobot.rs).

Commands
  ports                      list serial ports (the Waveshare board is CH343, VID:PID 1a86:55d3)
  scan                       find every servo on the bus (all IDs, all baud rates)
  info  ID                   dump the key registers of one servo
  set-id NEW                 give the ONE servo on the bus a new ID (and force 1 Mbps)
  wizard                     guided: connect servo 1, Enter, servo 2, Enter ... 8

Wire format (Feetech SMS/STS protocol, little-endian 16-bit values):
  request : FF FF ID LEN INSTR PARAM... CHK     LEN = len(PARAM)+2
  status  : FF FF ID LEN ERR   PARAM... CHK     CHK = ~(ID+LEN+INSTR/ERR+PARAMs) & 0xFF
"""

from __future__ import annotations

import argparse
import sys
import time

# ── Protocol constants ───────────────────────────────────────────────────────
INST_PING, INST_READ, INST_WRITE = 0x01, 0x02, 0x03
BROADCAST_ID = 0xFE
MAX_ID = 253

# EEPROM / RAM registers (same table as norma-core protocol/memory.rs)
REG_MODEL = 0x00          # 2 bytes, STS3215 = 777
REG_FW = 0x02
REG_ID = 0x05
REG_BAUD = 0x06
REG_MIN_ANGLE = 0x09      # 2 bytes
REG_MAX_ANGLE = 0x0B      # 2 bytes
REG_OFFSET = 0x1F         # 2 bytes, sign-magnitude (bit 11)
REG_MODE = 0x21
REG_TORQUE_ENABLE = 0x28
REG_LOCK = 0x37           # 0 = EEPROM writes persist, 1 = locked
REG_PRESENT_POS = 0x38    # 2 bytes
REG_VOLTAGE = 0x3E        # 0.1 V
REG_TEMP = 0x3F           # deg C

STS3215_MODEL = 777

# REG_BAUD value -> bps (norma-core SUPPORTED_BAUD_RATES uses the same order)
BAUD_TABLE = {0: 1_000_000, 1: 500_000, 2: 250_000, 3: 128_000,
              4: 115_200, 5: 76_800, 6: 57_600, 7: 38_400}
TARGET_BAUD = 1_000_000

CH343_VID, CH343_PID = 0x1A86, 0x55D3

ELROBOT_JOINTS = {
    1: "base yaw",
    2: "shoulder",
    3: "elbow",
    4: "wrist pitch",
    5: "forearm roll",
    6: "wrist bend",
    7: "wrist roll",
    8: "gripper",
}


class ServoError(RuntimeError):
    pass


def checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def build_packet(servo_id: int, instr: int, params: bytes = b"") -> bytes:
    body = bytes([servo_id, len(params) + 2, instr]) + params
    return b"\xFF\xFF" + body + bytes([checksum(body)])


class Bus:
    """Minimal half-duplex Feetech bus. `port` is a pyserial-like object."""

    def __init__(self, port, timeout: float = 0.02):
        self.port = port
        self.timeout = timeout

    # ── low level ────────────────────────────────────────────────────────
    def _read_status(self, expect_id: int | None, sent: bytes) -> tuple[int, int, bytes] | None:
        """Return (id, error, params) of the first valid status packet, or None on timeout.
        An exact echo of our own request (some adapters loop TX back) is skipped."""
        buf = b""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            chunk = self.port.read(self.port.in_waiting or 1)
            if chunk:
                buf += chunk
            while True:
                start = buf.find(b"\xFF\xFF")
                if start < 0:
                    buf = buf[-1:]
                    break
                buf = buf[start:]
                # 0xFF can't be an ID; skip header padding like FF FF FF
                if len(buf) >= 3 and buf[2] == 0xFF:
                    buf = buf[1:]
                    continue
                if len(buf) < 4:
                    break
                total = 4 + buf[3]
                if len(buf) < total:
                    break
                pkt, buf = buf[:total], buf[total:]
                if pkt == sent:
                    continue
                if checksum(pkt[2:-1]) != pkt[-1]:
                    continue
                sid, err, params = pkt[2], pkt[4], pkt[5:-1]
                if expect_id is None or sid == expect_id:
                    return sid, err, params
        return None

    def _txrx(self, servo_id: int, instr: int, params: bytes = b"",
              expect_id: int | None = -1, retries: int = 2):
        pkt = build_packet(servo_id, instr, params)
        if expect_id == -1:
            expect_id = servo_id
        for _ in range(retries + 1):
            self.port.reset_input_buffer()
            self.port.write(pkt)
            self.port.flush()
            status = self._read_status(expect_id, pkt)
            if status is not None:
                return status
        return None

    # ── public API ───────────────────────────────────────────────────────
    def ping(self, servo_id: int, retries: int = 0) -> bool:
        return self._txrx(servo_id, INST_PING, retries=retries) is not None

    def read(self, servo_id: int, addr: int, length: int) -> bytes:
        st = self._txrx(servo_id, INST_READ, bytes([addr, length]))
        if st is None:
            raise ServoError(f"ID {servo_id}: no reply reading 0x{addr:02X}")
        if len(st[2]) != length:
            raise ServoError(f"ID {servo_id}: short read at 0x{addr:02X}")
        return st[2]

    def read_u8(self, sid: int, addr: int) -> int:
        return self.read(sid, addr, 1)[0]

    def read_u16(self, sid: int, addr: int) -> int:
        return int.from_bytes(self.read(sid, addr, 2), "little")

    def write(self, servo_id: int, addr: int, data: bytes, expect_id: int | None = -1) -> bool:
        """Write and wait for the ack. Returns False if no ack arrived."""
        return self._txrx(servo_id, INST_WRITE, bytes([addr]) + data, expect_id=expect_id) is not None

    def write_u8(self, sid: int, addr: int, value: int, expect_id: int | None = -1) -> bool:
        return self.write(sid, addr, bytes([value & 0xFF]), expect_id=expect_id)


# ── serial helpers ───────────────────────────────────────────────────────────
def _serial():
    try:
        import serial  # noqa: F401
        import serial.tools.list_ports  # noqa: F401
        return serial
    except ImportError:
        sys.exit("pyserial is missing: run `uv sync` in ElRobotRepo/")


def find_port() -> str:
    serial = _serial()
    ports = list(serial.tools.list_ports.comports())
    ch343 = [p for p in ports if p.vid == CH343_VID and p.pid == CH343_PID]
    if len(ch343) == 1:
        return ch343[0].device
    if len(ch343) > 1:
        sys.exit("Several CH343 adapters found; pass --port explicitly:\n  "
                 + "\n  ".join(p.device for p in ch343))
    sys.exit("No Waveshare (CH343 1a86:55d3) adapter found. Check the USB data cable and the "
             "jumper (must be on B for USB), or pass --port. Run `ports` to list devices.")


def open_bus(port_name: str, baud: int) -> Bus:
    serial = _serial()
    port = serial.Serial(port_name, baudrate=baud, timeout=0.005, write_timeout=0.1)
    return Bus(port)


def reopen(bus: Bus, baud: int) -> None:
    bus.port.baudrate = baud
    time.sleep(0.05)
    bus.port.reset_input_buffer()


# ── operations ───────────────────────────────────────────────────────────────
def scan_ids(bus: Bus, ids=range(0, MAX_ID + 1)) -> list[int]:
    return [i for i in ids if bus.ping(i)]


def scan_all_bauds(bus: Bus, ids=range(0, MAX_ID + 1), stop_at_first=False) -> dict[int, list[int]]:
    """{baud: [ids]} over every supported baud rate, 1 Mbps first."""
    found: dict[int, list[int]] = {}
    for baud in BAUD_TABLE.values():
        reopen(bus, baud)
        hits = scan_ids(bus, ids)
        if hits:
            found[baud] = hits
            if stop_at_first:
                break
    reopen(bus, TARGET_BAUD)
    return found


def describe(bus: Bus, sid: int) -> str:
    try:
        model = bus.read_u16(sid, REG_MODEL)
        pos = bus.read_u16(sid, REG_PRESENT_POS)
        volt = bus.read_u8(sid, REG_VOLTAGE) / 10
        temp = bus.read_u8(sid, REG_TEMP)
    except ServoError as e:
        return f"(read failed: {e})"
    tag = "STS3215" if model == STS3215_MODEL else f"model {model}"
    joint = ELROBOT_JOINTS.get(sid, "-")
    return f"{tag:8s} pos={pos:4d}  {volt:4.1f} V  {temp:2d} C   joint: {joint}"


def find_single_servo(bus: Bus) -> tuple[int, int]:
    """Return (baud, id) of the one servo on the bus; abort if 0 or >1."""
    # New servos are all at 1 Mbps; only fall back to the other rates if nothing answers there.
    print("Scanning bus (IDs 0-253)...")
    found = scan_all_bauds(bus, stop_at_first=True)
    total = sum(len(v) for v in found.values())
    if total == 0:
        raise ServoError("No servo answered. Is the 12 V supply on? Is the servo cable seated?")
    if total > 1:
        listing = ", ".join(f"ID {i} @ {b}" for b, ids in found.items() for i in ids)
        raise ServoError(f"More than one servo answered ({listing}). "
                         "Connect exactly ONE servo while assigning IDs.")
    baud, ids = next(iter(found.items()))
    return baud, ids[0]


def set_id(bus: Bus, new_id: int, assume_yes: bool = False) -> None:
    if not 0 <= new_id <= MAX_ID:
        raise ServoError(f"ID must be 0..{MAX_ID}")
    baud, old_id = find_single_servo(bus)
    reopen(bus, baud)
    print(f"Found servo ID {old_id} at {baud} bps  ->  {describe(bus, old_id)}")

    if old_id == new_id and baud == TARGET_BAUD:
        print(f"Already ID {new_id} at 1 Mbps; just making sure EEPROM is locked.")
        bus.write_u8(new_id, REG_LOCK, 1)
        return

    if not assume_yes:
        ans = input(f"Change ID {old_id} -> {new_id}"
                    f"{'' if baud == TARGET_BAUD else ' and baud -> 1 Mbps'}? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    # Torque off so nothing moves, then unlock EEPROM so the change survives power-off.
    bus.write_u8(old_id, REG_TORQUE_ENABLE, 0)
    if not bus.write_u8(old_id, REG_LOCK, 0):
        raise ServoError("Unlock (0x37=0) was not acknowledged")

    # The ack to an ID write may come back from either ID, so accept any.
    bus.write_u8(old_id, REG_ID, new_id, expect_id=None)
    time.sleep(0.05)
    if not bus.ping(new_id, retries=5):
        raise ServoError(f"Servo did not answer at new ID {new_id}. "
                         f"Run `scan` to see where it went.")

    if baud != TARGET_BAUD:
        bus.write_u8(new_id, REG_BAUD, 0, expect_id=None)   # 0 = 1 Mbps; ack arrives at new baud
        reopen(bus, TARGET_BAUD)
        if not bus.ping(new_id, retries=5):
            raise ServoError("Servo did not answer at 1 Mbps after the baud change; run `scan`.")

    if not bus.write_u8(new_id, REG_LOCK, 1):
        raise ServoError("Lock (0x37=1) was not acknowledged; the change may not persist")

    got = bus.read_u8(new_id, REG_ID)
    if got != new_id:
        raise ServoError(f"Verify failed: ID register reads {got}, expected {new_id}")
    print(f"OK  ID {old_id} -> {new_id}  ({ELROBOT_JOINTS.get(new_id, 'not an ElRobot joint')}), "
          f"1 Mbps, EEPROM locked.")


# ── CLI ──────────────────────────────────────────────────────────────────────
def cmd_ports(_args) -> None:
    serial = _serial()
    for p in serial.tools.list_ports.comports():
        mark = "  <- Waveshare/CH343" if (p.vid, p.pid) == (CH343_VID, CH343_PID) else ""
        vidpid = f"{p.vid:04x}:{p.pid:04x}" if p.vid is not None else "----:----"
        print(f"{p.device:32s} {vidpid}  {p.description}{mark}")


def cmd_scan(args) -> None:
    bus = open_bus(args.port or find_port(), TARGET_BAUD)
    ids = range(0, MAX_ID + 1) if args.full else range(0, 21)
    bauds = list(BAUD_TABLE.values()) if args.all_bauds else [TARGET_BAUD]
    print(f"Pinging IDs {ids.start}-{ids.stop - 1} at {', '.join(map(str, bauds))} bps ...")
    any_found = False
    for baud in bauds:
        reopen(bus, baud)
        for sid in scan_ids(bus, ids):
            any_found = True
            print(f"  ID {sid:3d} @ {baud:>7d}  {describe(bus, sid)}")
    if not any_found:
        print("  nothing answered (power on? jumper on B? try --full --all-bauds)")
    elif not args.full:
        print("(only IDs 0-20 scanned; use --full for 0-253)")


def cmd_info(args) -> None:
    bus = open_bus(args.port or find_port(), args.baud)
    sid = args.id
    if not bus.ping(sid, retries=2):
        sys.exit(f"ID {sid} does not answer at {args.baud} bps")
    off_raw = bus.read_u16(sid, REG_OFFSET)
    offset = -(off_raw & 0x7FF) if off_raw & 0x800 else off_raw & 0x7FF
    rows = [
        ("model", bus.read_u16(sid, REG_MODEL)),
        ("firmware", bus.read_u8(sid, REG_FW)),
        ("id", bus.read_u8(sid, REG_ID)),
        ("baud", BAUD_TABLE.get(bus.read_u8(sid, REG_BAUD), "?")),
        ("min angle limit", bus.read_u16(sid, REG_MIN_ANGLE)),
        ("max angle limit", bus.read_u16(sid, REG_MAX_ANGLE)),
        ("position offset", offset),
        ("mode", bus.read_u8(sid, REG_MODE)),
        ("torque enable", bus.read_u8(sid, REG_TORQUE_ENABLE)),
        ("eeprom lock", bus.read_u8(sid, REG_LOCK)),
        ("present position", bus.read_u16(sid, REG_PRESENT_POS)),
        ("voltage [V]", bus.read_u8(sid, REG_VOLTAGE) / 10),
        ("temperature [C]", bus.read_u8(sid, REG_TEMP)),
    ]
    for k, v in rows:
        print(f"  {k:18s} {v}")


def cmd_set_id(args) -> None:
    bus = open_bus(args.port or find_port(), TARGET_BAUD)
    try:
        set_id(bus, args.new_id, assume_yes=args.yes)
    except ServoError as e:
        sys.exit(f"ERROR: {e}")


def cmd_wizard(args) -> None:
    bus = open_bus(args.port or find_port(), TARGET_BAUD)
    targets = [i for i in range(args.start, 9)]
    print("ElRobot follower servo ID wizard\n"
          "  - Waveshare jumper on B (USB), 12 V supply plugged into the board.\n"
          "  - Plug in exactly ONE servo at a time (either of its two connectors).\n"
          "  - Write the number on the servo with a marker before you unplug it (manual step 3).\n")
    for new_id in targets:
        while True:
            ans = input(f"[{new_id}/8] Connect ONLY the servo for '{ELROBOT_JOINTS[new_id]}', "
                        f"then Enter (s = skip, q = quit): ").strip().lower()
            if ans == "q":
                return
            if ans == "s":
                break
            try:
                set_id(bus, new_id, assume_yes=True)
                print(f"    -> Label this servo '{new_id}' and unplug it.\n")
                break
            except ServoError as e:
                print(f"    ERROR: {e}\n    Fix it and press Enter to retry.")
    print("Done. Now daisy-chain all 8 and run:  uv run servo_setup/servo_id.py scan   (expect IDs 1-8)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial device (default: auto-detect the Waveshare CH343)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ports", help="list serial ports").set_defaults(func=cmd_ports)

    p = sub.add_parser("scan", help="list servos on the bus")
    p.add_argument("--full", action="store_true", help="scan IDs 0-253 instead of 0-20")
    p.add_argument("--all-bauds", action="store_true", help="also try the non-1 Mbps baud rates")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("info", help="dump key registers of one servo")
    p.add_argument("id", type=int)
    p.add_argument("--baud", type=int, default=TARGET_BAUD)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("set-id", help="assign a new ID to the single connected servo")
    p.add_argument("new_id", type=int)
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.set_defaults(func=cmd_set_id)

    p = sub.add_parser("wizard", help="assign IDs 1..8 one servo at a time")
    p.add_argument("--start", type=int, default=1, choices=range(1, 9), help="resume from this ID")
    p.set_defaults(func=cmd_wizard)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
