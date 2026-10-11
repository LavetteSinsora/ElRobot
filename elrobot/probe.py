"""Read-only bus check. Never writes a register, never moves a servo.

    uv run python -m elrobot.probe            # ping 1-8, dump each servo's settings, time group reads
    uv run python -m elrobot.probe --watch    # live positions (with min/max seen) from one sync read per frame
"""

import argparse
import sys
import time

import serial

from .st3215 import registers as R
from .st3215.bus import Bus, BusError

JOINTS = {
    1: "base yaw",
    2: "shoulder",
    3: "elbow",
    4: "wrist pitch",
    5: "forearm roll",
    6: "wrist bend",
    7: "wrist roll",
    8: "gripper",
}

# What a factory-fresh ST3215 looks like; anything else is worth knowing before calibrating.
# Offset is not listed: servos 1-7 read 85 before we ever wrote one, so 0 is not a reliable baseline.
EXPECTED = {
    "model": R.STS3215_MODEL,
    "mode": 0,
    "torque_enable": 0,
    "min_position_limit": 0,
    "max_position_limit": 4095,
    "lock": 1,
}


def deg(steps: int) -> float:
    return steps * 360.0 / R.STEPS_PER_TURN


def check(t: dict[str, int], alarm: int) -> list[str]:
    notes = [f"{k} = {t[k]} (factory {v})" for k, v in EXPECTED.items() if t[k] != v]
    volts = t["present_voltage"] / 10
    if not 9.0 <= volts <= 12.6:
        notes.append(f"supply {volts:.1f} V (expect ~12 V)")
    if t["present_temperature"] > 50:
        notes.append(f"temperature {t['present_temperature']} C")
    if alarm:
        notes.append("alarm: " + ", ".join(R.status_flags(alarm)))
    return notes


def report(bus: Bus, ids: list[int]) -> list[int]:
    found = [i for i in ids if bus.ping(i)]
    missing = sorted(set(ids) - set(found))
    print(f"answered: {found or 'none'}" + (f"   MISSING: {missing}" if missing else ""))
    if not found:
        return found

    print(f"\n{'id':>2}  {'joint':<12} {'pos':>5} {'deg':>6}  {'V':>4} {'C':>3}  {'mode':>4} "
          f"{'torque':>6} {'offset':>6} {'limits':>11}  fw")
    tables = {}
    for i in found:
        t = tables[i] = bus.read_table(i)
        print(f"{i:>2}  {JOINTS.get(i, '?'):<12} {t['present_position']:>5} "
              f"{deg(t['present_position']):>6.1f}  {t['present_voltage'] / 10:>4.1f} "
              f"{t['present_temperature']:>3}  {t['mode']:>4} {t['torque_enable']:>6} "
              f"{t['offset']:>6} {t['min_position_limit']:>5}-{t['max_position_limit']:<5}  "
              f"{t['firmware_major']}.{t['firmware_minor']}")

    print()
    for i in found:
        notes = check(tables[i], bus.alarms.get(i, 0))
        print(f"servo {i}: " + ("factory settings, ok" if not notes else "; ".join(notes)))
    return found


def benchmark(bus: Bus, ids: list[int], n: int = 200):
    t0 = time.perf_counter()
    for _ in range(n):
        bus.sync_read_reg(ids, R.PRESENT_POSITION)
    sync_dt = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n // 4):
        for i in ids:
            bus.read_reg(i, R.PRESENT_POSITION)
    single_dt = (time.perf_counter() - t0) / (n // 4)

    print(f"\nreading all {len(ids)} positions: sync_read {sync_dt * 1e3:.2f} ms "
          f"({1 / sync_dt:.0f} Hz), one read per servo {single_dt * 1e3:.2f} ms "
          f"({1 / single_dt:.0f} Hz)")


class RangeTracker:
    """Follows one joint's motion across the 4095->0 wrap and keeps the extremes.

    Positions are unwrapped by assuming no joint moves more than half a turn between two
    reads (at 50 Hz that would be over 9000 deg/s).
    """

    def __init__(self):
        self.last = self.u = self.lo = self.hi = None

    def update(self, p: int):
        if self.last is None:
            self.u = self.lo = self.hi = p
        else:
            half = R.STEPS_PER_TURN // 2
            self.u += (p - self.last + half) % R.STEPS_PER_TURN - half
            self.lo, self.hi = min(self.lo, self.u), max(self.hi, self.u)
        self.last = p

    @property
    def span(self) -> int:
        return self.hi - self.lo

    @property
    def ends(self) -> tuple[int, int]:
        return self.lo % R.STEPS_PER_TURN, self.hi % R.STEPS_PER_TURN

    @property
    def midpoint(self) -> int:
        return ((self.lo + self.hi) // 2) % R.STEPS_PER_TURN

    @property
    def crosses_wrap(self) -> bool:
        return self.lo < 0 or self.hi >= R.STEPS_PER_TURN


def watch(bus: Bus, ids: list[int]):
    torque = bus.sync_read_reg(ids, R.TORQUE_ENABLE)
    if on := [i for i, t in torque.items() if t]:
        print(f"WARNING: torque is ON for {on}. Those servos are holding position: "
              "do not force them by hand. Power-cycle the 12 V supply to release them.\n")
    print("Read-only: nothing is commanded. Ctrl-C to stop and print the ranges.\n")
    print("  ".join(f"{i}:{JOINTS.get(i, '?')[:10]:>10} now [range seen] span" for i in ids))
    trackers = {i: RangeTracker() for i in ids}
    try:
        while True:
            for i, p in bus.sync_read_reg(ids, R.PRESENT_POSITION).items():
                trackers[i].update(p)
            line = "  ".join(f"{t.last:>4} [{t.ends[0]:>4}-{t.ends[1]:<4}] {deg(t.span):>5.1f}deg"
                             for t in (trackers[i] for i in ids))
            print("\r" + line, end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\nrange seen (raw steps, from one end to the other in the direction travelled):")
        for i in ids:
            t = trackers[i]
            if t.last is None:
                continue
            wrap = "  CROSSES 4095->0" if t.crosses_wrap else ""
            print(f"  {i} {JOINTS.get(i, '?'):<12} {t.ends[0]:>4} .. {t.ends[1]:<4}  span {t.span:>4} "
                  f"steps = {deg(t.span):6.1f} deg  midpoint {t.midpoint}{wrap}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect the Waveshare adapter)")
    ap.add_argument("--ids", type=int, nargs="+", default=list(JOINTS))
    ap.add_argument("--watch", action="store_true", help="stream positions instead of the report")
    args = ap.parse_args(argv)

    try:
        with Bus(args.port) as bus:
            if args.watch:
                watch(bus, args.ids)
                return 0
            found = report(bus, args.ids)
            if found:
                benchmark(bus, found)
            return 0 if len(found) == len(args.ids) else 1
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
