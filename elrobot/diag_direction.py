"""Find out when a servo moves against its command. Gripper (8) only, small moves.

    uv run python -m elrobot.diag_direction        # all cases
    uv run python -m elrobot.diag_direction A      # only case A: the offset the servo powered up with
    uv run python -m elrobot.diag_direction offsets -1871 85 0 479 -1871
                                                    # write each offset in turn and test both ways

The gripper must be about half open, so ~10 degrees either way is free. Each case writes the
settings below, turns torque on at 15 % holding where it is, commands 2 degrees down, then
2 degrees up, and records which way it actually went. Any move of more than 1.5 degrees the
wrong way turns torque off immediately and skips to the next case. The original offset is
restored at the end.

Cases (what happens between "torque off" and the 2-degree moves):
  A  nothing: offset unchanged
  B  offset changed, then speed/accel/goal/torque immediately   (current find_range order)
  C  speed/accel written first, then offset change, then goal/torque   (old order that worked)
  D  offset changed, 300 ms pause, then speed/accel/goal/torque
  E  offset changed so the position reads ~4035 (where the failures happened), immediate
  F  same as E with the 300 ms pause
"""

import sys
import time

from .st3215 import registers as R
from .st3215.bus import Bus, BusError

J = 8
STEP = 23
WRONG = 17
TORQUE = 150


def wrapped(d):
    d %= 4096
    d = d - 4096 if d > 2048 else d
    return 2047 if d == 2048 else d


def move(bus, goal):
    """Command `goal`, watch for 0.6 s. Returns (start, end, verdict)."""
    start = bus.read_reg(J, R.PRESENT_POSITION)
    bus.write_reg(J, R.GOAL_POSITION, goal)
    t0, pos = time.monotonic(), start
    while time.monotonic() - t0 < 0.6:
        pos = bus.read_reg(J, R.PRESENT_POSITION)
        if (goal - start) * (pos - start) < 0 and abs(pos - start) > WRONG:
            bus.write_reg(J, R.TORQUE_ENABLE, 0)
            return start, pos, "WRONG WAY (torque off)"
        time.sleep(0.02)
    ok = (goal - start) * (pos - start) > 0 and abs(pos - start) > 5
    return start, pos, "ok" if ok else "did not move"


def hold(bus):
    bus.write_reg(J, R.GOAL_POSITION, bus.read_reg(J, R.PRESENT_POSITION))
    bus.write_reg(J, R.TORQUE_LIMIT, TORQUE)
    bus.write_reg(J, R.TORQUE_ENABLE, 1)


def speed(bus):
    bus.write_reg(J, R.GOAL_SPEED, 120)
    bus.write_reg(J, R.ACCELERATION, 20)


def set_reading(bus, want):
    off = bus.read_reg(J, R.OFFSET)
    enc = (bus.read_reg(J, R.PRESENT_POSITION) + off) % 4096
    bus.write_eeprom_reg(J, R.OFFSET, wrapped(enc - want))


def case(bus, name, prepare):
    bus.write_reg(J, R.TORQUE_ENABLE, 0)
    prepare()
    hold(bus)
    p = bus.read_reg(J, R.PRESENT_POSITION)
    results = []
    for goal in (p - STEP, None):
        if goal is None:  # back up to where it started
            goal = bus.read_reg(J, R.PRESENT_POSITION) + STEP
        if not bus.read_reg(J, R.TORQUE_ENABLE):
            break
        results.append((goal,) + move(bus, goal))
    bus.write_reg(J, R.TORQUE_ENABLE, 0)
    off = bus.read_reg(J, R.OFFSET)
    for goal, start, end, verdict in results:
        print(f"  {name}  offset {off:>6}  {start:>4} -> goal {goal:>4}: went to {end:>4}  {verdict}")


def main(argv=None):
    only = (argv if argv is not None else sys.argv[1:])
    input("Gripper about half open, hand near the 12 V switch. Enter to start (Ctrl-C to cancel): ")
    with Bus(timeout=0.1) as bus:
        if any(bus.read_reg(i, R.TORQUE_ENABLE) for i in range(1, 9)):
            print("torque is on somewhere; run elrobot.relax first")
            return 2
        original = bus.read_reg(J, R.OFFSET)
        print(f"servo {J}: offset {original}, position {bus.read_reg(J, R.PRESENT_POSITION)}")
        try:
            case(bus, "A", lambda: None)
            if only == ["A"]:
                return 0
            if only[:1] == ["offsets"]:
                for v in map(int, only[1:]):
                    case(bus, f"set {v:>6}", lambda v=v: bus.write_eeprom_reg(J, R.OFFSET, v))
                return 0
            case(bus, "B", lambda: (set_reading(bus, 2048), speed(bus)))
            case(bus, "C", lambda: (speed(bus), set_reading(bus, 1500)))
            case(bus, "D", lambda: (set_reading(bus, 2600), time.sleep(0.3), speed(bus)))
            case(bus, "E", lambda: (set_reading(bus, 4035), speed(bus)))
            case(bus, "F", lambda: (set_reading(bus, 4035), time.sleep(0.3), speed(bus)))
        except (BusError, KeyboardInterrupt) as e:
            print(f"stopped: {e or 'Ctrl-C'}")
        finally:
            bus.write_reg(J, R.TORQUE_ENABLE, 0)
            bus.write_eeprom_reg(J, R.OFFSET, original)
            print(f"torque off, offset restored to {original}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
