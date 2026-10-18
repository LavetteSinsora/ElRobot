"""Find one joint's range of motion by driving it gently into both of its stops.

    uv run python -m elrobot.find_range --joint 8                 # measure, then restore settings
    uv run python -m elrobot.find_range --joint 8 --write-offset  # ...and centre the range on 2048
    uv run python -m elrobot.relax                                # clean up after a run that died

One joint per run. The other joints hold their current position, so arrange the arm by hand
(torque off) into the pose suggested for the joint before starting.

How it works:
  1. The joint's original offset is saved to calibration/restore.json before anything changes.
  2. The joint gets a low torque limit and a slow speed. The protection current is left at its
     factory value: lowering it starves the motor (first gripper run, 2026-10-11).
  3. A temporary offset makes the current position read near 4035, so the joint can travel
     almost a full turn downward without crossing the 4095->0 wrap (the servo cannot move
     across it; it would turn the long way round into the stop). It moves in ~2 degree steps.
  4. When it lags its target and stops moving, the torque limit is raised one level and it
     tries again, and stays raised for a few steps once it moves. Friction gives way; a real
     stop may squeeze a little but stalls again. Only a stall at the highest level counts as a
     stop. Then it holds, backs off ~5 degrees and records the end.
     The gripper rests fully open against its stop: that end is read without moving, and the
     search only closes it, up to the URDF width or until the jaws meet.
  A joint that moves against its command (seen on the gripper, 2026-10-11) has its torque
  turned off at once and the run stops.
  Held joints get a gentle speed, and a held joint within ~13 degrees of the 4095->0 point is
  refused (see elrobot.precentre): it could turn the long way round into its stop.
  5. The offset is changed so that end reads 60, and the upward search repeats. The span is the
     distance travelled in that search, so it cannot be misread the long way round.
  6. Travel is capped at the URDF range + 15 degrees, so a joint with no hard stop (roll joints,
     limited by their cable) cannot wind its cable up.
  7. Ctrl-C, an error or the terminal closing: every servo holds where it is; you support the
     arm and press Enter; torque goes off and the original settings come back.

Every sample (position, load, current, torque limit) is logged to calibration/logs/*.csv.
Results are merged into calibration/ranges.json. --write-offset only writes when the result
passes its checks (both ends are stops, span close to the URDF), unless --accept is given.
"""

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import serial

from .kinematics import UPSTREAM_URDF, Robot
from .probe import JOINTS
from .st3215 import registers as R
from .st3215.bus import Bus, BusError

TURN = R.STEPS_PER_TURN
CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"

STEP = 23            # steps per position command (~2 deg)
SPEED = 120          # steps/s (~10.5 deg/s); NormaCore uses 365
ACCEL = 20           # x100 steps/s^2
REACHED_TOL = 10     # |goal - position| that counts as arrived
STALL_ERR = 30       # lagging by more than this ...
STALL_TIME = 0.3     # ... without moving for this long (s) = stalled
STILL = 3            # movement below this many steps counts as not moving
STEP_TIMEOUT = 2.0   # s
BACKOFF = 57         # ~5 deg
EDGE = 60            # search starts this far from 0 / 4095
EXTRA_TRAVEL = math.radians(15)
HOLD_TORQUE_LIMIT = 500  # other joints, while holding (0.1 %)
HOLD_SPEED = 200         # steps/s; after power-up speed reads 0, which means "as fast as possible"
HOLD_ACCEL = 10          # x100 steps/s^2
SEAM_MARGIN = 150        # a held joint this close (~13 deg) to the 4095->0 point is refused
WRONG_WAY = 17           # moving this far (~1.5 deg) against the command aborts the run
RECENT_ESCALATION = 3    # steps to stay at a raised torque level after it got the joint moving
POLL = 0.02
SPAN_TOLERANCE_DEG = 20  # measured span vs URDF width before the result is flagged


@dataclass
class Profile:
    torque_levels: tuple[int, ...]  # 0.1 % of max torque; escalated on each stall
    pose: str
    # Joints that rest against one of their stops (the gripper sits fully open): that end is read
    # where it rests, with no motion, and the search only moves AWAY from it (+1 = the resting
    # stop is at the encoder-increasing end). It stops when it meets the other stop or has moved
    # the URDF width, whichever comes first, so it never pushes past the resting stop.
    rests_at_stop: int | None = None


PROFILES = {
    8: Profile((150, 220), "Rest pose, gripper fully open against its stop. Nothing between the jaws.",
               rests_at_stop=+1),  # fully open = encoder ~3300, readings rise when opening
    7: Profile((150, 220, 300), "Upper arm straight up, forearm level and pointing forward (the URDF "
                                "zero pose). Camera cable slack; watch it, Ctrl-C if it pulls tight."),
    6: Profile((200, 280, 350), "Upper arm straight up, forearm level, so the gripper can swing up "
                                "and down without touching the table."),
    5: Profile((150, 220, 300), "Upper arm straight up, forearm level. Cables slack; Ctrl-C if they "
                                "pull tight."),
    4: Profile((250, 330, 400), "Upper arm straight up, forearm level, so the wrist can swing up and "
                                "down freely."),
    1: Profile((250, 330, 400), "Rest pose, with clear space all around the base."),
    3: Profile((300, 400, 500), "Upper arm tilted back (about 45 deg) so the forearm cannot reach the "
                                "table when it folds down. Watch closely."),
    2: Profile((300, 400, 500), "Elbow folded (forearm over the upper arm) so the arm is short. Clear "
                                "space in front and behind. Watch closely."),
}


class Abort(Exception):
    pass


@dataclass
class End:
    encoder: int       # encoder position at the end (independent of any offset)
    kind: str          # "stop": stalled at the highest torque level; "cap": travel cap, no stop;
                       # resting-stop joints: "rest" (read, not pushed), "defined" (moved the
                       # URDF width without meeting a stop)
    travel: int        # steps from the search's reference point (the anchor, if one was given)
    torque_limit: int  # level at which it finally stalled
    peak_load: int
    peak_current: int


def wrapped(diff: int) -> int:
    """diff mod 4096 as a signed value in [-2047, 2048]."""
    d = diff % TURN
    return d - TURN if d > TURN // 2 else d


def representable(offset: int) -> int:
    """The offset register holds -2047..2047; 2048 and -2048 are the same position."""
    return 2047 if offset == 2048 else offset


class RangeFinder:
    def __init__(self, bus: Bus, joint: int, width_rad: float, hold: list[int],
                 profile: Profile | None = None, workdir: Path = CALIBRATION_DIR,
                 clock=time.monotonic, sleep=time.sleep, log=print):
        self.bus, self.j, self.hold = bus, joint, hold
        self.profile = profile or PROFILES[joint]
        self.width_steps = round(width_rad * TURN / (2 * math.pi))
        self.cap_steps = round((width_rad + EXTRA_TRAVEL) * TURN / (2 * math.pi))
        self.workdir = workdir
        self.clock, self.sleep, self.log = clock, sleep, log
        self.torque_on: list[int] = []
        self.saved: dict | None = None
        self.phase = "prepare"
        self.torque_limit = self.profile.torque_levels[0]
        self.t0 = clock()
        self.csv_file = self.csv = None
        self.log_path: Path | None = None

    # -- bookkeeping ----------------------------------------------------------------------

    @property
    def restore_path(self) -> Path:
        return self.workdir / "restore.json"

    def open_log(self):
        logs = self.workdir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / f"find_range_j{self.j}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = path.open("w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["t", "phase", "goal", "position", "load", "current", "torque_limit", "event"])
        self.log_path = path

    def record(self, goal=None, s=None, event=""):
        if self.csv is None:
            return
        s = s or {}
        self.csv.writerow([f"{self.clock() - self.t0:.3f}", self.phase, goal,
                           s.get("present_position"), s.get("present_load"),
                           s.get("present_current"), self.torque_limit, event])
        if event:
            self.csv_file.flush()

    def close_log(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = self.csv = None

    # -- low level ------------------------------------------------------------------------

    def state(self) -> dict[str, int]:
        start = R.PRESENT_POSITION.addr
        data = self.bus.read(self.j, start, R.PRESENT_CURRENT.addr + 2 - start)
        return R.decode_block(start, data)

    def position(self, servo: int | None = None) -> int:
        return self.bus.read_reg(self.j if servo is None else servo, R.PRESENT_POSITION)

    def encoder(self, reported: int) -> int:
        return (reported + self.offset) % TURN

    def enable(self, servo: int, torque_limit: int, speed: int = SPEED, accel: int = ACCEL):
        """Hold the current position: goal = present BEFORE torque on, so nothing jumps, and a
        gentle speed so any correction (slack, sag) is slow."""
        self.bus.write_reg(servo, R.GOAL_SPEED, speed)
        self.bus.write_reg(servo, R.ACCELERATION, accel)
        self.bus.write_reg(servo, R.GOAL_POSITION, self.position(servo))
        self.bus.write_reg(servo, R.TORQUE_LIMIT, torque_limit)
        self.bus.write_reg(servo, R.TORQUE_ENABLE, 1)
        if servo not in self.torque_on:
            self.torque_on.append(servo)

    def disable(self, servo: int):
        self.bus.write_reg(servo, R.TORQUE_ENABLE, 0)
        if servo in self.torque_on:
            self.torque_on.remove(servo)

    def set_torque_limit(self, value: int):
        self.torque_limit = value
        self.bus.write_reg(self.j, R.TORQUE_LIMIT, value)

    def check_alarm(self):
        if alarm := self.bus.alarms.get(self.j):
            raise Abort(f"servo {self.j} reports {', '.join(R.status_flags(alarm))}; "
                        "power-cycle and check it before searching")

    def set_offset_so_it_reads(self, want: int) -> int:
        """Re-offset the (torque-off) joint so its current position reads `want`."""
        enc = self.encoder(self.position())
        off = representable(wrapped(enc - want))
        self.bus.write_eeprom_reg(self.j, R.OFFSET, off)
        self.offset = off
        got, expect = self.position(), (enc - off) % TURN
        self.record(event=f"offset {off}: expect {expect}, read {got}")
        if abs(got - expect) > 15:
            raise Abort(f"offset check failed: expected position {expect} after writing offset "
                        f"{off}, read {got}. The offset does not work the way this tool assumes; "
                        "stopping before anything moves.")
        return got

    # -- motion ---------------------------------------------------------------------------

    def wait(self, goal: int, peaks: dict) -> str:
        """Returns "reached" or "stall"."""
        t0 = last_move = self.clock()
        ref = None
        origin = None
        while True:
            s = self.state()
            self.record(goal, s)
            pos = s["present_position"]
            peaks["load"] = max(peaks["load"], abs(s["present_load"]))
            peaks["current"] = max(peaks["current"], abs(s["present_current"]))
            if alarm := self.bus.alarms.get(self.j):
                self.log(f"  servo alarm: {', '.join(R.status_flags(alarm))}")
                self.record(goal, s, "alarm " + ",".join(R.status_flags(alarm)))
                return "stall"
            if origin is None:
                origin = pos
            if (goal - origin) * (pos - origin) < 0 and abs(pos - origin) > WRONG_WAY:
                self.disable(self.j)
                self.record(goal, s, "WRONG WAY: torque off")
                raise Abort(f"servo {self.j} moved the wrong way: told to go from {origin} to "
                            f"{goal}, went to {pos}. Torque turned off on it.")
            err = abs(goal - pos)
            if err <= REACHED_TOL:
                return "reached"
            now = self.clock()
            if ref is None or abs(pos - ref) > STILL:
                ref, last_move = pos, now
            if err > STALL_ERR and now - last_move >= STALL_TIME:
                return "stall"
            if now - t0 >= STEP_TIMEOUT:
                return "stall" if err > STALL_ERR else "reached"
            self.sleep(POLL)

    def search(self, direction: int, anchor: int | None = None) -> End:
        """Move in `direction` (-1 down, +1 up) until a stop or the travel cap.

        With `anchor` (the encoder position of the end already found), the offset is chosen so
        the anchor reads EDGE (or 4095 - EDGE) and the whole range lies ahead of the joint.
        """
        name = "lower" if direction < 0 else "upper"
        self.phase = name
        levels = self.profile.torque_levels
        self.disable(self.j)
        base = TURN - 1 - EDGE if direction < 0 else EDGE
        if anchor is None:
            start = self.set_offset_so_it_reads(base)
        else:
            enc = self.encoder(self.position())
            start = self.set_offset_so_it_reads(base + direction * ((direction * (enc - anchor)) % TURN))
        self.torque_limit = levels[0]
        self.enable(self.j, levels[0])
        self.check_alarm()

        self.log(f"  searching {name} end ...")
        peaks = {"load": 0, "current": 0}
        goal, kind, level, recent = start, "cap", 0, 0
        while True:
            nxt = goal + direction * STEP
            if abs(nxt - base) > self.cap_steps or not EDGE // 2 <= nxt <= TURN - 1 - EDGE // 2:
                self.log(f"  reached the travel limit "
                         f"({math.degrees(self.cap_steps * 2 * math.pi / TURN):.0f} deg) without a stop")
                self.record(goal, event="travel cap")
                break
            self.bus.write_reg(self.j, R.GOAL_POSITION, nxt)
            goal = nxt
            outcome = self.wait(goal, peaks)
            while outcome == "stall" and level + 1 < len(levels):
                level += 1
                self.set_torque_limit(levels[level])
                self.record(goal, event=f"stall, torque limit -> {levels[level]}")
                self.check_alarm()
                outcome = self.wait(goal, peaks)
            if outcome == "stall":
                kind = "stop"
                self.record(goal, event="stop")
                break
            # It moved. After a stall, stay at the raised level for a few steps: friction keeps
            # moving, while a stop that only squeezed stalls again at the top level (= stop).
            if level and recent == 0:
                recent = RECENT_ESCALATION
                self.record(goal, event=f"moved at {levels[level]}")
            elif level:
                recent -= 1
                if recent == 0:
                    self.record(goal, event=f"clear, back to {levels[0]}")
                    level = 0
                    self.set_torque_limit(levels[0])

        pos = self.position()
        end = End(self.encoder(pos), kind, abs(pos - base), self.torque_limit,
                  peaks["load"], peaks["current"])
        # Hold at the gentlest level (re-enabling also clears an overload unload), then back off.
        self.enable(self.j, levels[0])
        self.torque_limit = levels[0]
        back = min(TURN - 1 - EDGE // 2, max(EDGE // 2, pos - direction * BACKOFF))
        self.bus.write_reg(self.j, R.GOAL_POSITION, back)
        self.wait(back, {"load": 0, "current": 0})
        self.log(f"  {name} end at encoder {end.encoder} ({kind} at torque limit "
                 f"{end.torque_limit / 10:.0f} %), travelled {end.travel * 360 / TURN:.1f} deg")
        return end

    # -- procedure ------------------------------------------------------------------------

    def prepare(self):
        if self.restore_path.exists():
            raise Abort(f"{self.restore_path.name} exists: an earlier run did not clean up. "
                        "Run `uv run python -m elrobot.relax` first.")
        t = self.bus.read_table(self.j)
        if t["mode"] != 0:
            raise Abort(f"servo {self.j} is in mode {t['mode']}, expected 0 (position)")
        if (t["min_position_limit"], t["max_position_limit"]) != (0, 4095):
            raise Abort(f"servo {self.j} has position limits {t['min_position_limit']}-"
                        f"{t['max_position_limit']}; the search needs the factory 0-4095")
        if t["torque_enable"]:
            raise Abort(f"servo {self.j} already has torque on; power-cycle first")
        self.check_alarm()
        near = {h: p for h in self.hold
                if min(p := self.position(h), R.STEPS_PER_TURN - 1 - p) < SEAM_MARGIN}
        if near:
            detail = ", ".join(f"servo {h} reads {p}" for h, p in near.items())
            raise Abort(f"{detail}: too close to the 4095->0 point to hold safely. Run "
                        f"`uv run python -m elrobot.precentre {' '.join(map(str, near))}` first.")
        self.offset = t["offset"]
        self.saved = {"servo": self.j, "offset": t["offset"], "torque_limit": t["torque_limit"],
                      "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.restore_path.write_text(json.dumps(self.saved, indent=2) + "\n")
        self.open_log()
        self.record(event=f"start, offset {t['offset']}, position {t['present_position']}")
        for h in self.hold:
            self.enable(h, HOLD_TORQUE_LIMIT, HOLD_SPEED, HOLD_ACCEL)

    def freeze(self):
        for s in list(self.torque_on):
            try:
                self.bus.write_reg(s, R.GOAL_POSITION, self.position(s))
            except BusError:
                pass

    def release(self, new_offset: int | None):
        self.phase = "release"
        for s in list(self.torque_on):
            try:
                self.disable(s)
            except BusError as e:
                self.log(f"  could not turn torque off on servo {s}: {e}")
        if self.saved is not None:
            self.bus.write_reg(self.j, R.TORQUE_LIMIT, self.saved["torque_limit"])
            final = self.saved["offset"] if new_offset is None else new_offset
            self.bus.write_eeprom_reg(self.j, R.OFFSET, final)
            self.offset = final
            self.record(event=f"released, offset {final}")
            self.restore_path.unlink(missing_ok=True)
        self.close_log()

    def run(self) -> tuple[End, End]:
        self.prepare()
        if (d := self.profile.rests_at_stop) is not None:
            rest = End(self.encoder(self.position()), "rest", 0, 0, 0, 0)
            self.record(event=f"resting end at encoder {rest.encoder}, not pushed")
            full_cap, self.cap_steps = self.cap_steps, self.width_steps
            try:
                other = self.search(-d)
            finally:
                self.cap_steps = full_cap
            if other.kind == "cap":
                other.kind = "defined"
            return (other, rest) if d > 0 else (rest, other)
        lo = self.search(-1)
        hi = self.search(+1, anchor=lo.encoder)
        return lo, hi


def summarize(joint: int, lo: End, hi: End, width_rad: float) -> dict:
    # The second search starts with the first end at its reference point, so the distance it
    # travelled is the span: the upper end in a two-stop search, the defined end otherwise.
    if "rest" in (lo.kind, hi.kind):
        second = hi if lo.kind == "rest" else lo
    else:
        second = hi
    span = second.travel
    mid = (lo.encoder + span // 2) % TURN
    problems = []
    kinds = sorted((lo.kind, hi.kind))
    if "cap" in kinds:
        problems.append("at least one end is the travel cap, not a stop")
    if abs((hi.encoder - lo.encoder) % TURN - span) > 2 * STEP:
        problems.append(f"end positions ({lo.encoder} -> {hi.encoder}) disagree with the "
                        f"distance travelled ({span} steps)")
    span_deg = span * 360 / TURN
    if abs(span_deg - math.degrees(width_rad)) > SPAN_TOLERANCE_DEG:
        problems.append(f"span {span_deg:.1f} deg is more than {SPAN_TOLERANCE_DEG} deg from "
                        f"the URDF's {math.degrees(width_rad):.1f}")
    return {
        "joint": joint,
        "name": JOINTS[joint],
        "lower": asdict(lo),
        "upper": asdict(hi),
        "span_steps": span,
        "span_deg": round(span_deg, 1),
        "urdf_width_deg": round(math.degrees(width_rad), 1),
        "midpoint_encoder": mid,
        "centring_offset": representable(wrapped(mid - 2048)),
        "problems": problems,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save(result: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[str(result["joint"])] = result
    path.write_text(json.dumps(data, indent=2) + "\n")


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", type=int, required=True, choices=sorted(PROFILES))
    ap.add_argument("--hold", type=int, nargs="*",
                    help="joints that hold position during the search (default: all other joints)")
    ap.add_argument("--write-offset", action="store_true",
                    help="if the result passes its checks, centre the range on 2048 (EEPROM offset)")
    ap.add_argument("--accept", action="store_true",
                    help="with --write-offset: write even if the result has problems")
    ap.add_argument("--port")
    args = ap.parse_args(argv)

    j = args.joint
    joint = Robot(UPSTREAM_URDF).joints[f"rev_motor_0{j}"]     # upstream's full width, not ours
    width = joint.upper - joint.lower
    hold = [h for h in (args.hold if args.hold is not None else JOINTS) if h != j]
    p = PROFILES[j]

    print(f"Joint {j} ({JOINTS[j]}): URDF range {math.degrees(width):.1f} deg.")
    print(f"Torque limit {p.torque_levels[0] / 10:.0f} %, raised to "
          f"{', '.join(f'{t / 10:.0f} %' for t in p.torque_levels[1:])} on a stall; "
          f"speed {SPEED * 360 / TURN:.1f} deg/s.")
    print(f"\nPose: {p.pose}\nHolding in place: {hold or 'nothing'}")
    print("Base clamped? Hand near the 12 V switch? Ctrl-C at any time holds everything.\n")
    if input("Type 'go' to start: ").strip().lower() != "go":
        print("cancelled, nothing was changed")
        return 1

    try:
        bus = Bus(args.port)
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Closing the terminal (SIGHUP) or `kill` (SIGTERM) takes the same path as Ctrl-C.
    signal.signal(signal.SIGHUP, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)

    finder = RangeFinder(bus, j, width, hold)
    result, code, interactive = None, 0, True
    try:
        lo, hi = finder.run()
        result = summarize(j, lo, hi, width)
    except (KeyboardInterrupt, Abort, BusError) as e:
        finder.freeze()
        print(f"\nstopped: {e or 'Ctrl-C'}. Everything is holding its position.")
        interactive = not (isinstance(e, KeyboardInterrupt) and str(e).startswith("signal"))
        code = 1

    new_offset = None
    if result:
        save(result, CALIBRATION_DIR / "ranges.json")
        diff = result["span_deg"] - result["urdf_width_deg"]
        print(f"\nJoint {j}: span {result['span_deg']} deg (URDF {result['urdf_width_deg']}, "
              f"difference {diff:+.1f}), midpoint encoder {result['midpoint_encoder']}, "
              f"centring offset {result['centring_offset']}. Saved to ranges.json.")
        for problem in result["problems"]:
            print(f"  PROBLEM: {problem}")
        if args.write_offset:
            if result["problems"] and not args.accept:
                print("Not writing the offset because of the problems above (--accept overrides).")
            else:
                new_offset = result["centring_offset"]
    if finder.log_path:
        print(f"Log: {finder.log_path}")

    if interactive:
        try:
            input("\nSupport the arm, then press Enter to turn torque off: ")
        except (KeyboardInterrupt, EOFError, OSError):
            print()
    finder.release(new_offset)
    print("Torque off. Offset "
          + (f"set to {new_offset} (range centred on 2048)." if new_offset is not None
             else "restored to its previous value."))
    bus.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
