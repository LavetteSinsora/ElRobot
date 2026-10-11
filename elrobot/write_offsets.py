"""Write the centring offsets from calibration/hand_ranges.json to the servos.

    uv run python -m elrobot.write_offsets          # show what would be written, change nothing
    uv run python -m elrobot.write_offsets --write  # write them (torque must be off everywhere)

Each offset puts the middle of that joint's measured range at reading 2048, so the joint can
never cross the 4095->0 point inside its range. Only the offset register (EEPROM) is written;
nothing moves. Power-cycle afterwards: the servo's controller only picks up a new offset
cleanly at power-up (see the 2026-10-11 runaway tests).
"""

import argparse
import json
import sys
from pathlib import Path

import serial

from .st3215 import registers as R
from .st3215.bus import Bus, BusError

HAND_RANGES = Path(__file__).resolve().parent.parent / "calibration/hand_ranges.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--port")
    args = ap.parse_args(argv)
    joints = json.loads(HAND_RANGES.read_text())["joints"]

    try:
        with Bus(args.port) as bus:
            ids = sorted(int(j) for j in joints)
            on = [i for i in ids if bus.read_reg(i, R.TORQUE_ENABLE)]
            if on:
                print(f"torque is on for {on}; run `uv run python -m elrobot.relax` first")
                return 2
            print(f"{'servo':<17}{'offset now':>10}{'new':>8}{'reads now':>11}{'will read':>11}")
            plan = []
            for i in ids:
                j = joints[str(i)]
                old, pos = bus.read_reg(i, R.OFFSET), bus.read_reg(i, R.PRESENT_POSITION)
                if old != j["offset_at_measurement"]:
                    print(f"servo {i}: offset is {old}, but the hand sweep was measured with "
                          f"{j['offset_at_measurement']}; stopping (measure again)")
                    return 2
                new = j["centring_offset"]
                will = (pos + old - new) % R.STEPS_PER_TURN
                plan.append((i, new, will))
                print(f"{i} {j['name']:<15}{old:>10}{new:>8}{pos:>11}{will:>11}")
            if not args.write:
                print("\nNothing written. Run with --write to apply.")
                return 0
            for i, new, will in plan:
                bus.write_eeprom_reg(i, R.OFFSET, new)
                got = bus.read_reg(i, R.PRESENT_POSITION)
                flag = "" if abs(got - will) <= 15 else f"  (expected {will}!)"
                print(f"servo {i}: offset {new} written, reads {got}{flag}")
            print("\nDone. Now power-cycle the 12 V supply (support the arm first), then run\n"
                  "`uv run python -m elrobot.probe` to check every joint reads near the 'will read' value.")
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
