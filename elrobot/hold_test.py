"""Measure how far joints sag while holding, under different controller settings.

    uv run python -m elrobot.hold_test                # joints 3 and 4, all cases
    uv run python -m elrobot.hold_test --joints 2 3 4

The arm holds its current pose (every joint, torque on, goal = where it is). For each case it
holds for HOLD seconds and records, per tested joint, how far the reading ends up from the
target and how hard the servo pushes. Cases:

  factory   the servo's own P/I/D (32/0/32 on these servos)
  stiff-P   P raised to 48 (I/D unchanged)
  servo-I   the servo's integral term on, I = 8 (as NormaCore uses on the SO-101)
  our-I     factory settings, plus our code nudging the target to cancel the error

P/I/D are EEPROM registers: they are written with torque off and the originals are put back
at the end. The same runaway / wrong-way checks as elrobot.move apply. Ctrl-C holds.
"""

import argparse
import math
import sys
import time

import serial

from .calibration import load
from .move import TORQUE, MoveAbort, Mover, steps
from .st3215 import registers as R
from .st3215.bus import Bus, BusError

HOLD = 4.0
PERIOD = 0.02
KI_OURS = 2.0               # 1/s: fraction of the error added to the target per second
OUR_I_LIMIT_DEG = 6


def deg(x: float) -> float:
    return x * 360 / R.STEPS_PER_TURN


class HoldTest:
    def __init__(self, bus: Bus, cals, joints: list[int], clock=time.monotonic, sleep=time.sleep):
        self.bus, self.cals, self.joints = bus, cals, joints
        self.mover = Mover(bus, cals, clock, sleep, log=lambda *a: None)
        self.clock, self.sleep = clock, sleep
        self.original = {j: (bus.read_reg(j, R.P_COEF), bus.read_reg(j, R.I_COEF), bus.read_reg(j, R.D_COEF))
                         for j in joints}

    def set_pid(self, pid: dict[int, tuple[int, int, int]]):
        for j, (p, i, d) in pid.items():
            for reg, v in ((R.P_COEF, p), (R.I_COEF, i), (R.D_COEF, d)):
                if self.bus.read_reg(j, reg) != v:
                    self.bus.write_eeprom_reg(j, reg, v)

    def hold(self, ours: bool) -> dict[int, dict]:
        target = self.mover.enable()            # goal = present, torque on, runaway watch
        integ = {j: 0.0 for j in self.joints}
        stats = {j: {"err": [], "load": []} for j in self.joints}
        t0 = last = self.clock()
        while (now := self.clock()) - t0 < HOLD:
            dt, last = now - last, now
            start = R.PRESENT_POSITION.addr
            raw = self.bus.sync_read(self.mover.ids, start, R.PRESENT_LOAD.addr + 2 - start)
            for j in self.joints:
                st = R.decode_block(start, raw[j])
                err = target[j] - st["present_position"]
                if abs(err) > steps(15):
                    self.mover.freeze()
                    raise MoveAbort(f"joint {j} moved {deg(err):.0f} deg away while holding")
                if now - t0 > HOLD / 2:         # second half: settled
                    stats[j]["err"].append(err)
                    stats[j]["load"].append(st["present_load"])
                if ours:
                    lim = steps(OUR_I_LIMIT_DEG)
                    integ[j] = max(-lim, min(lim, integ[j] + KI_OURS * err * dt))
            if ours:
                self.bus.sync_write_reg(R.GOAL_POSITION,
                                        {j: target[j] + round(integ.get(j, 0)) for j in self.mover.ids})
            self.sleep(PERIOD)
        self.mover.release()
        return {j: {"mean_err_deg": deg(sum(s["err"]) / len(s["err"])),
                    "max_err_deg": deg(max(abs(e) for e in s["err"])),
                    "load_pct": sum(s["load"]) / len(s["load"]) / 10,
                    "correction_deg": deg(integ[j])}
                for j, s in stats.items()}

    def restore(self):
        self.mover.release()
        self.set_pid(self.original)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joints", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--port")
    args = ap.parse_args(argv)
    cals = load()
    try:
        bus = Bus(args.port)
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    test = HoldTest(bus, cals, args.joints)
    print("original P/I/D: " + ", ".join(f"joint {j}: {v}" for j, v in test.original.items()))
    print(f"Torque limits {', '.join(f'{j}: {TORQUE[j] / 10:.0f} %' for j in args.joints)}. The arm holds its "
          f"current pose for {HOLD:.0f} s per case, then goes limp briefly between cases.")
    if input("Arm in a safe pose, supported? Type 'go': ").strip().lower() != "go":
        bus.close()
        return 1

    base = {j: test.original[j] for j in args.joints}
    cases = [("factory", base, False),
             ("stiff-P", {j: (48, i, d) for j, (p, i, d) in base.items()}, False),
             ("servo-I", {j: (p, 8, d) for j, (p, i, d) in base.items()}, False),
             ("our-I", base, True)]
    code = 0
    try:
        test.mover.check_ready()
        for name, pid, ours in cases:
            test.set_pid(pid)
            res = test.hold(ours)
            for j, r in res.items():
                print(f"  {name:<8} joint {j}: sag {r['mean_err_deg']:+5.2f} deg (worst {r['max_err_deg']:.2f}), "
                      f"load {r['load_pct']:+5.1f} %" + (f", our correction {r['correction_deg']:+.2f} deg" if ours else ""))
            time.sleep(0.5)
    except (MoveAbort, BusError, KeyboardInterrupt) as e:
        test.mover.freeze()
        print(f"\nstopped: {e or 'Ctrl-C'}")
        try:
            input("Support the arm, then press Enter to turn torque off: ")
        except (KeyboardInterrupt, EOFError):
            pass
        code = 1
    test.restore()
    print("Torque off, original P/I/D restored.")
    bus.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
