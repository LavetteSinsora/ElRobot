"""Turn torque off on every servo and undo an unfinished find_range run.

    uv run python -m elrobot.relax

Use it when find_range was killed before cleaning up (calibration/restore.json exists), or
any time you want every joint limp. Support the arm first: the shoulder and elbow will drop.
"""

import argparse
import json
import sys

import serial

from .find_range import CALIBRATION_DIR
from .probe import JOINTS
from .st3215 import registers as R
from .st3215.bus import Bus, BusError


def relax(bus: Bus, ids: list[int], restore_path=CALIBRATION_DIR / "restore.json", log=print):
    bus.sync_write_reg(R.TORQUE_ENABLE, {i: 0 for i in ids})
    still_on = [i for i, t in bus.sync_read_reg(ids, R.TORQUE_ENABLE).items() if t]
    if still_on:
        raise BusError(f"torque still on for {still_on}")
    log(f"torque off: {ids}")
    if restore_path.exists():
        saved = json.loads(restore_path.read_text())
        bus.write_reg(saved["servo"], R.TORQUE_LIMIT, saved["torque_limit"])
        bus.write_eeprom_reg(saved["servo"], R.OFFSET, saved["offset"])
        restore_path.unlink()
        log(f"servo {saved['servo']}: offset restored to {saved['offset']} "
            f"(saved by the run started {saved['started']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--ids", type=int, nargs="+", default=list(JOINTS))
    args = ap.parse_args(argv)
    input("Support the arm, then press Enter to turn torque off on every servo: ")
    try:
        with Bus(args.port) as bus:
            relax(bus, args.ids)
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
