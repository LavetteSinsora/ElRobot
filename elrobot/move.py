"""Move the arm to a named pose (see poses.py), slowly, with every joint watched.

    uv run python -m elrobot.move rest --dry-run   # show the plan, change nothing
    uv run python -m elrobot.move rest             # plan, type 'go', move, hold, Enter to release

What it does:
  1. Checks every servo's offset matches calibration.json and torque is off.
  2. Converts the pose to servo readings and clamps them to each joint's safe range.
  3. Turns torque on holding the current position (goal = present first, so nothing jumps) and
     watches for 0.3 s. Any joint that starts moving on its own means its controller disagrees
     with its reading (the runaway seen on 2026-10-11): torque off everywhere, stop.
  4. Sends each joint its final target once, with speeds/accelerations scaled so all joints start
     and stop together; the servos run the smooth speed-up / cruise / slow-down themselves (the
     furthest joint at SPEED_DEG deg/s). Streaming a target every tick (--stream) made slow
     joints stop and start ~40 times a second (2026-10-11 log). Every tick each joint is still
     checked against that profile: moving the wrong way, or more than LAG_DEG behind, stops the
     move (the wrong-way joint is released, the others hold). Every tick is logged to
     calibration/logs/move_*.csv.
  5. Holds the pose until you press Enter, then turns torque off. Ctrl-C at any point holds.
"""

import argparse
import csv
import math
import sys
import time

import serial

from .calibration import CAL_DIR, RAD_PER_STEP, load
from .poses import POSES
from .st3215 import registers as R
from .st3215.bus import Bus, BusError

SPEED_DEG = 12          # deg/s for the joint that has the furthest to go
TORQUE = {1: 500, 2: 500, 3: 500, 4: 500, 5: 400, 6: 400, 7: 400, 8: 300}  # 0.1 %
GOAL_SPEED = 600        # steps/s cap in the servo itself (the ramp is slower than this)
ACCEL = 30
RAMP_S = 0.6            # speed-up / slow-down time of the furthest joint (servo-planned moves)
WRONG_DEG = 1.5
LAG_DEG = 8
LAG_TIME = 0.5
MAX_MOVE_DEG = 120      # refuse bigger moves unless --far
PERIOD = 0.02
LOG_DIR = CAL_DIR / "logs"


class MoveAbort(Exception):
    pass


def steps(deg: float) -> int:
    return round(math.radians(deg) / RAD_PER_STEP)


class Mover:
    def __init__(self, bus: Bus, cals, clock=time.monotonic, sleep=time.sleep, log=print):
        self.bus, self.cals = bus, cals
        self.ids = sorted(cals)
        self.clock, self.sleep, self.log = clock, sleep, log
        self.torque_on = False
        self.log_file = self.csv = None
        self.log_path = None

    def read(self) -> dict[int, int]:
        return self.bus.sync_read_reg(self.ids, R.PRESENT_POSITION)

    def plan(self, pose: dict[int, float]) -> tuple[dict, dict]:
        now = self.read()
        target = {}
        for j in self.ids:
            c = self.cals[j]
            target[j] = c.clamp_reading(c.to_reading(math.radians(pose[j])))
        return now, target

    def check_ready(self):
        on = [j for j, t in self.bus.sync_read_reg(self.ids, R.TORQUE_ENABLE).items() if t]
        if on:
            raise MoveAbort(f"torque is already on for {on}; run elrobot.relax first")
        off = {j: self.bus.read_reg(j, R.OFFSET) for j in self.ids}
        wrong = [j for j in self.ids if off[j] != self.cals[j].offset]
        if wrong:
            raise MoveAbort(f"offsets of {wrong} differ from calibration.json; recalibrate first")

    def release(self):
        self.bus.sync_write_reg(R.TORQUE_ENABLE, {j: 0 for j in self.ids})
        self.torque_on = False

    def freeze(self):
        try:
            self.bus.sync_write_reg(R.GOAL_POSITION, self.read())
        except BusError:
            pass

    def enable(self):
        start = self.read()
        self.bus.sync_write_reg(R.GOAL_POSITION, start)
        self.bus.sync_write_reg(R.GOAL_SPEED, {j: GOAL_SPEED for j in self.ids})
        self.bus.sync_write_reg(R.ACCELERATION, {j: ACCEL for j in self.ids})
        self.bus.sync_write_reg(R.TORQUE_LIMIT, {j: TORQUE[j] for j in self.ids})
        self.bus.sync_write_reg(R.TORQUE_ENABLE, {j: 1 for j in self.ids})
        self.torque_on = True
        t0 = self.clock()
        while self.clock() - t0 < 0.3:
            pos = self.sample("enable", start)
            drift = {j: pos[j] - start[j] for j in self.ids if abs(pos[j] - start[j]) > steps(WRONG_DEG)}
            if drift:
                self.release()
                raise MoveAbort(f"joints {sorted(drift)} started moving on their own when torque came "
                                "on (controller disagrees with the reading). Torque is off everywhere. "
                                "Power-cycle before trying again.")
            self.sleep(PERIOD)
        return start

    # -- logging ------------------------------------------------------------------------

    def open_log(self, name: str):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"move_{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_file = self.log_path.open("w", newline="", buffering=1)  # line-buffered: survives a crash
        self.csv = csv.writer(self.log_file)
        self.csv.writerow(["t", "phase", "loop_dt"] + [f"{k}{j}" for j in self.ids
                                                       for k in ("goal", "pos", "speed", "load")])
        self.t0_log = self.last_tick = self.clock()

    def sample(self, phase: str, goal: dict[int, int] | None) -> dict[int, int]:
        """One sync read of position, speed and load for every joint; logged if a log is open."""
        start = R.PRESENT_POSITION.addr
        raw = self.bus.sync_read(self.ids, start, R.PRESENT_LOAD.addr + 2 - start)
        st = {j: R.decode_block(start, raw[j]) for j in self.ids}
        if self.csv is not None:
            now = self.clock()
            row = [f"{now - self.t0_log:.4f}", phase, f"{now - self.last_tick:.4f}"]
            self.last_tick = now
            for j in self.ids:
                row += [goal[j] if goal else "", st[j]["present_position"],
                        st[j]["present_speed"], st[j]["present_load"]]
            self.csv.writerow(row)
        return {j: st[j]["present_position"] for j in self.ids}

    def close_log(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = self.csv = None

    def move(self, target: dict[int, int]):
        """Point-to-point move planned by the servos themselves.

        Each joint gets its final target once, with a speed and acceleration scaled so every
        joint starts and stops together (the furthest one runs at SPEED_DEG with a RAMP_S ramp).
        The servos run the speed-up / cruise / slow-down on their own fast internal loop. We keep
        reading every joint and compare it with the same profile, for the safety checks and log.
        """
        start = self.enable()
        dist = {j: target[j] - start[j] for j in self.ids}
        far = max(abs(d) for d in dist.values())
        if far == 0:
            return start
        vmax = steps(SPEED_DEG)                       # steps/s for the furthest joint
        amax = vmax / RAMP_S                          # steps/s^2
        speed, accel = {}, {}
        for j in self.ids:
            f = abs(dist[j]) / far
            # Goal speed 0 means "full speed" on this servo, so never send 0.
            speed[j] = max(1, round(vmax * f))
            accel[j] = max(1, min(254, round(amax * f / 100)))   # register unit: 100 steps/s^2
        self.bus.sync_write_reg(R.ACCELERATION, accel)
        self.bus.sync_write_reg(R.GOAL_SPEED, speed)
        self.bus.sync_write_reg(R.GOAL_POSITION, target)

        def expected(j, t):                           # trapezoid with the speed/accel we sent
            v, a, d = speed[j], accel[j] * 100, abs(dist[j])
            if d == 0:
                return start[j]
            tr = v / a
            if d < v * tr:                            # too short to reach full speed
                tr = math.sqrt(d / a); v = a * tr
            cruise = (d - v * tr) / v
            if t < tr: x = 0.5 * a * t * t
            elif t < tr + cruise: x = 0.5 * a * tr * tr + v * (t - tr)
            elif t < 2 * tr + cruise: x = d - 0.5 * a * (2 * tr + cruise - t) ** 2
            else: x = d
            return round(start[j] + math.copysign(min(x, d), dist[j]))

        duration = max(far / vmax + RAMP_S, 0.5)
        t0 = self.clock()
        lag_since = {}
        while True:
            t = self.clock() - t0
            model = {j: expected(j, t) for j in self.ids}
            pos = self.sample("move", model)
            now = self.clock()
            for j in self.ids:
                d = dist[j]
                if d * (pos[j] - start[j]) < 0 and abs(pos[j] - start[j]) > steps(WRONG_DEG) \
                        or d == 0 and abs(pos[j] - start[j]) > 3 * steps(WRONG_DEG):
                    self.bus.write_reg(j, R.TORQUE_ENABLE, 0)
                    self.freeze()
                    raise MoveAbort(f"joint {j} moved the wrong way ({start[j]} -> {pos[j]}, "
                                    f"target {target[j]}). Its torque is off; the others hold.")
                if abs(model[j] - pos[j]) > steps(LAG_DEG):
                    lag_since.setdefault(j, now)
                    if now - lag_since[j] > LAG_TIME:
                        self.freeze()
                        raise MoveAbort(f"joint {j} is {abs(model[j] - pos[j]) * 360 / 4096:.0f} deg behind "
                                        "where it should be (blocked?). Everything holds where it is.")
                else:
                    lag_since.pop(j, None)
            if t > duration + 0.3:
                break
            self.sleep(PERIOD)
        t_end = self.clock() + 1.5
        while self.clock() < t_end:
            pos = self.sample("settle", target)
            self.sleep(PERIOD)
        return pos

    def move_stream(self, target: dict[int, int]):
        """Old method: stream a new target every tick. Stutters on slow joints (2026-10-11 log)."""
        start = self.enable()
        dist = max(abs(target[j] - start[j]) for j in self.ids)
        duration = max(1.0, dist / steps(SPEED_DEG))
        t0 = self.clock()
        lag_since = {}
        while True:
            t = min(1.0, (self.clock() - t0) / duration)
            s = 0.5 - 0.5 * math.cos(math.pi * t)
            goal = {j: round(start[j] + s * (target[j] - start[j])) for j in self.ids}
            self.bus.sync_write_reg(R.GOAL_POSITION, goal)
            pos = self.sample("move", goal)
            now = self.clock()
            for j in self.ids:
                d = target[j] - start[j]
                if d * (pos[j] - start[j]) < 0 and abs(pos[j] - start[j]) > steps(WRONG_DEG) \
                        or d == 0 and abs(pos[j] - start[j]) > 3 * steps(WRONG_DEG):
                    self.bus.write_reg(j, R.TORQUE_ENABLE, 0)
                    self.freeze()
                    raise MoveAbort(f"joint {j} moved the wrong way ({start[j]} -> {pos[j]}, "
                                    f"target {target[j]}). Its torque is off; the others hold.")
                if abs(goal[j] - pos[j]) > steps(LAG_DEG):
                    lag_since.setdefault(j, now)
                    if now - lag_since[j] > LAG_TIME:
                        self.freeze()
                        raise MoveAbort(f"joint {j} is stuck {abs(goal[j] - pos[j]) * 360 / 4096:.0f} deg "
                                        "behind its target (blocked?). Everything holds where it is.")
                else:
                    lag_since.pop(j, None)
            if t >= 1.0:
                break
            self.sleep(PERIOD)
        # Settle for 1.5 s, still logging, so the final error and any oscillation are recorded.
        t_end = self.clock() + 1.5
        while self.clock() < t_end:
            pos = self.sample("settle", target)
            self.sleep(PERIOD)
        return pos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", choices=sorted(POSES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="old method: stream a target every 20 ms (for comparison; stutters)")
    ap.add_argument("--far", action="store_true", help=f"allow joints to move more than {MAX_MOVE_DEG} deg")
    ap.add_argument("--port")
    args = ap.parse_args(argv)
    cals = load()
    pose = POSES[args.pose]

    try:
        bus = Bus(args.port)
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    mover = Mover(bus, cals)
    try:
        mover.check_ready()
        now, target = mover.plan(pose)
    except (MoveAbort, BusError) as e:
        print(f"error: {e}", file=sys.stderr)
        bus.close()
        return 2

    deg = lambda j, r: math.degrees(cals[j].to_rad(r))
    print(f"{'joint':<16}{'now':>8}{'target':>8}{'move':>8}")
    biggest = 0
    for j in mover.ids:
        a, b = deg(j, now[j]), deg(j, target[j])
        biggest = max(biggest, abs(b - a))
        clamp = "  (clamped to safe range)" if abs(b - pose[j]) > 0.5 else ""
        print(f"{j} {cals[j].name:<14}{a:>+8.1f}{b:>+8.1f}{b - a:>+8.1f}{clamp}")
    secs = max(1.0, biggest / SPEED_DEG)
    print(f"\nLargest move {biggest:.0f} deg, about {secs:.0f} s at {SPEED_DEG} deg/s.")
    if args.dry_run:
        bus.close()
        return 0
    if biggest > MAX_MOVE_DEG and not args.far:
        print(f"Refusing: a joint would move more than {MAX_MOVE_DEG} deg. Move the arm closer by hand, "
              "or pass --far.")
        bus.close()
        return 1
    print("Clear space around the arm, hand near the 12 V switch. Ctrl-C holds everything.")
    if input("Type 'go' to move: ").strip().lower() != "go":
        print("cancelled, nothing moved")
        bus.close()
        return 1

    code = 0
    mover.open_log(args.pose + ("_stream" if args.stream else ""))
    try:
        final = mover.move_stream(target) if args.stream else mover.move(target)
        print("\nReached:")
        for j in mover.ids:
            print(f"  {j} {cals[j].name:<14}{deg(j, final[j]):>+7.1f} deg  "
                  f"(target {deg(j, target[j]):+.1f}, off by {deg(j, final[j]) - deg(j, target[j]):+.1f})")
    except (MoveAbort, BusError, KeyboardInterrupt) as e:
        if mover.torque_on:
            mover.freeze()
        print(f"\nstopped: {e or 'Ctrl-C'}")
        code = 1
    if mover.torque_on:
        try:
            input("\nHolding. Support the arm, then press Enter to turn torque off: ")
        except (KeyboardInterrupt, EOFError):
            print()
        mover.release()
        print("Torque off.")
    mover.close_log()
    print(f"Log: {mover.log_path}")
    bus.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
