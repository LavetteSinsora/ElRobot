"""Joint calibration: servo reading <-> URDF joint angle.

    angle (rad) = sign * (reading - zero_reading) * 2*pi/4096

Per joint, calibration/calibration.json stores:
  sign          +1 if the reading rises when the URDF angle rises (sign test), else -1
  zero_reading  the reading at URDF angle 0
  offset        the servo offset register this calibration assumes (checked against the servo)
  min_reading, max_reading   safe command limits: the measured ends pulled in by a margin

How the zero readings were found (2026-10-11):
  joints 1-4, 6  measured range width matched the URDF, so the middle of the range is the middle
                 of the URDF limits (0 deg for 1-4, +11.1 deg for 6). Joint 4 re-swept 2026-10-13
                 after a loose screw was tightened: middle moved 37 steps (3.3 deg).
  joint 8        fully open stop = URDF 0 (the URDF's jaw geometry is open at 0)
  joints 5, 7    set by eye to the URDF zero pose (cable limits their range); a few degrees rough

    uv run python -m elrobot.calibration          # build calibration.json from the measurements
    uv run python -m elrobot.calibration --show   # read the arm now and print URDF angles
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .kinematics import Robot
from .probe import JOINTS
from .st3215 import registers as R

TURN = R.STEPS_PER_TURN
RAD_PER_STEP = 2 * math.pi / TURN
CAL_DIR = Path(__file__).resolve().parent.parent / "calibration"
CAL_FILE = CAL_DIR / "calibration.json"

# Measurements from 2026-10-11 (signs from the joint-direction page, roll zeros set by eye).
SIGNS = {1: +1, 2: +1, 3: +1, 4: +1, 5: +1, 6: +1, 7: +1, 8: -1}
MEASURED_ZERO = {5: 1966, 7: 2134}
GRIPPER_OPEN_READING = 2861   # fully open stop after the centring offset
MARGIN_DEG = {1: 3, 2: 3, 3: 3, 4: 3, 5: 5, 6: 3, 7: 5, 8: 3}  # rolls: cable, so more margin


@dataclass
class JointCal:
    joint: int
    name: str
    sign: int
    zero_reading: int
    offset: int
    min_reading: int
    max_reading: int
    zero_source: str

    def to_rad(self, reading: int) -> float:
        return self.sign * (reading - self.zero_reading) * RAD_PER_STEP

    def to_reading(self, rad: float) -> int:
        return round(self.zero_reading + self.sign * rad / RAD_PER_STEP)

    def limits_rad(self) -> tuple[float, float]:
        a, b = self.to_rad(self.min_reading), self.to_rad(self.max_reading)
        return min(a, b), max(a, b)

    def clamp_reading(self, reading: int) -> int:
        return max(self.min_reading, min(self.max_reading, reading))


def build(hand_ranges: dict, robot: Robot) -> dict[int, JointCal]:
    cals = {}
    for j in range(1, 9):
        h = hand_ranges[str(j)]
        span = h["span_steps"]
        lo, hi = 2048 - span // 2, 2048 - span // 2 + span   # measured ends after centring
        if "resweep" in h:  # re-measured later with the offset already in place
            lo, hi = h["resweep"]["ends_with_current_offset"]
            span = hi - lo
        margin = round(math.radians(MARGIN_DEG[j]) / RAD_PER_STEP)
        jt = robot.joints[f"rev_motor_0{j}"]
        if j in MEASURED_ZERO:
            zero, source = MEASURED_ZERO[j], "set by eye to the URDF zero pose"
        elif j == 8:
            zero, source = GRIPPER_OPEN_READING, "fully open stop = URDF 0"
            # closing past the URDF's closed limit only squeezes the jaws
            closed = round(GRIPPER_OPEN_READING + SIGNS[8] * jt.upper / RAD_PER_STEP)
            lo = max(lo, closed)
        else:
            mid = (jt.upper + jt.lower) / 2
            middle = (lo + hi) / 2
            zero = round(middle - SIGNS[j] * mid / RAD_PER_STEP)
            source = f"range matches URDF; middle of range = URDF {math.degrees(mid):+.1f} deg"
            if "resweep" in h:
                source += f" (re-swept {h['resweep']['date']}: {h['resweep']['why']})"
        cals[j] = JointCal(j, JOINTS[j], SIGNS[j], zero, h["centring_offset"],
                           lo + margin, hi - margin, source)
    return cals


def load(path: Path = CAL_FILE) -> dict[int, JointCal]:
    data = json.loads(path.read_text())["joints"]
    return {int(k): JointCal(**v) for k, v in data.items()}


def save(cals: dict[int, JointCal], path: Path = CAL_FILE):
    path.write_text(json.dumps({
        "formula": "angle_rad = sign * (reading - zero_reading) * 2*pi/4096",
        "built": "2026-10-11 from hand_ranges.json, sign test, roll zeros by eye",
        "joints": {str(j): asdict(c) for j, c in cals.items()}}, indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="read the arm and print URDF angles")
    args = ap.parse_args(argv)
    robot = Robot()
    if not args.show:
        cals = build(json.loads((CAL_DIR / "hand_ranges.json").read_text())["joints"], robot)
        save(cals)
        print(f"wrote {CAL_FILE}")
    cals = load()

    readings = {}
    if args.show:
        from .st3215.bus import Bus
        with Bus() as bus:
            readings = bus.sync_read_reg(list(cals), R.PRESENT_POSITION)
            offsets = {j: bus.read_reg(j, R.OFFSET) for j in cals}
        wrong = [j for j in cals if offsets[j] != cals[j].offset]
        if wrong:
            print(f"WARNING: servo offsets differ from the calibration for {wrong}; angles are wrong")

    print(f"\n{'joint':<16}{'sign':>5}{'zero':>6}   {'safe range (deg)':>18}   {'URDF limits':>16}" +
          ("   reading   angle" if readings else ""))
    for j, c in cals.items():
        jt = robot.joints[f"rev_motor_0{j}"]
        a, b = (math.degrees(x) for x in c.limits_rad())
        line = (f"{j} {c.name:<14}{c.sign:>+5}{c.zero_reading:>6}   {a:>8.1f} .. {b:>6.1f}   "
                f"{math.degrees(jt.lower):>7.1f} .. {math.degrees(jt.upper):>5.1f}")
        if readings:
            line += f"   {readings[j]:>7}  {math.degrees(c.to_rad(readings[j])):>+6.1f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
