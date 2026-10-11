"""Give a joint a provisional offset so its current position reads 2048.

    uv run python -m elrobot.precentre 4

Use it when a joint rests near the 4095->0 point. Holding such a joint is dangerous: if its
reading lands on the other side of that point from its target, the servo turns the long way
round into its stop. Reading 2048 puts the point half a turn away from the current pose.
find_range later replaces this with the offset that centres the measured range.

Torque must be off. Nothing moves: only the offset register (EEPROM) is written.
"""

import argparse
import sys

import serial

from .st3215 import registers as R
from .st3215.bus import Bus, BusError

TURN = R.STEPS_PER_TURN


def precentre(bus: Bus, servo: int) -> tuple[int, int, int]:
    """Returns (old offset, new offset, position now)."""
    if bus.read_reg(servo, R.TORQUE_ENABLE):
        raise BusError(f"servo {servo} has torque on; turn it off first (elrobot.relax)")
    old = bus.read_reg(servo, R.OFFSET)
    enc = (bus.read_reg(servo, R.PRESENT_POSITION) + old) % TURN
    new = (enc - 2048) % TURN
    new = new - TURN if new > TURN // 2 else new
    new = 2047 if new == 2048 else new
    bus.write_eeprom_reg(servo, R.OFFSET, new)
    pos = bus.read_reg(servo, R.PRESENT_POSITION)
    if abs(pos - 2048) > 15:
        raise BusError(f"servo {servo}: wrote offset {new}, position reads {pos}, expected ~2048")
    return old, new, pos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", type=int, nargs="+")
    ap.add_argument("--port")
    args = ap.parse_args(argv)
    try:
        with Bus(args.port) as bus:
            for i in args.ids:
                old, new, pos = precentre(bus, i)
                print(f"servo {i}: offset {old} -> {new}, position now reads {pos}")
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
