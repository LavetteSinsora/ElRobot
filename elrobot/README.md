# elrobot: control stack for the ElRobot follower

New code, written from scratch. It doesn't depend on `servo_setup/`.

| Module | What it does |
|---|---|
| `st3215/registers.py` | ST3215 control table, including sign-magnitude encoding |
| `st3215/protocol.py` | Packet building and status parsing |
| `st3215/bus.py` | `Bus`: `ping`, `read`/`write` for one servo; `sync_read`/`sync_write` for a group |
| `probe.py` | Read-only check of the bus (see below) |
| `find_range.py` | Powered range search for one joint: low torque, slow speed, stops when the joint stalls |
| `relax.py` | Torque off on every servo; undoes a find_range run that died (`calibration/restore.json`) |
| `precentre.py` | Provisional offset so a joint reads 2048 where it rests (moves the 4095→0 point away) |
| `kinematics.py` | URDF parser and forward kinematics |

The upstream URDF and meshes are in `../description/elrobot_upstream/`; see `SOURCE.md` there.
Ours (`../description/elrobot_ours/`, generated from upstream's plus calibration) is the one
every pose is written in.

## Conventions

- Joint angles follow the **upstream URDF** (ours keeps them). At q = 0 the upper arm is
  vertical and the wrist and gripper are horizontal, pointing forward; z is up.
- Base frame: ours is REP-103, **x forward** (where the arm reaches at q = 0), **y left, z up**.
  Upstream's `base_link` has the arm along **+y** (x to the robot's right); ours renames it
  `base_body` and turns it under a new `base_link`. `kinematics` prints ours by default.
- Joint **limits** for `calibration` and `find_range` come from the upstream URDF (`Robot()`'s
  default): the middle of its ranges sets zero readings, its width caps a search. Our URDF's
  limits are the calibrated safe ranges built from those, so they can't be used here.
- The folded rest pose is the calibration reference and the home pose. It is *not* zero;
  it is shoulder −90°, elbow 0, wrist pitch +90° (`poses.py`).
- Servo offset (EEPROM): chosen so the middle of each joint's travel reads 2048, which keeps
  the 4095→0 wrap outside the joint's travel.
- Conversion (calibration file, coming next): `q = sign × (reported − 2048) × 2π/4096 + c`.

## Commands (from `ElRobotRepo/`)

```bash
uv run python -m elrobot.probe             # ping 1-8, dump settings, time sync vs single reads
uv run python -m elrobot.probe --watch     # live positions + min/max seen (read-only)
uv run python -m elrobot.find_range --joint 8                # MOVES joint 8; measures, then restores settings
uv run python -m elrobot.find_range --joint 8 --write-offset # ...and centres its range on 2048 if the checks pass
uv run python -m elrobot.relax                               # torque off everywhere + undo an unfinished run
uv run python -m elrobot.precentre 4                         # joint 4 rests near 4095->0: move that point away
uv run python -m elrobot.kinematics --deg 0 -90 0 90 0 0 0 0   # the rest pose, our base frame (mm)
uv run pytest                              # simulated-bus and kinematics tests
```

`probe` only sends PING and READ packets. It never writes a register, so nothing moves.

`find_range` moves the arm. Run it one joint at a time, with the base clamped and a hand near
the 12 V switch. Order: 8, 4, 6, 7, 5, 1, then 3 and 2 with close supervision.
Results go to `calibration/ranges.json`, and a per-sample log to `calibration/logs/`.

Lessons from the first real gripper run (2026-10-11): never lower the protection current (10
starved the motor, the factory value is 310); a stall at low torque can be friction, so the
torque is raised in steps before a stall counts as a stop; and a run that dies must be
recoverable, hence `restore.json` and `relax`. Holding joint 4 while it rested at reading 29
made it swing the long way round (second run), hence the guard and `precentre`; servo 4 got
offset −2002 (provisional) that day. Third run: the gripper (which rests fully open, against
its open stop at encoder ~3300) was pushed wider in both searches: the first by a wrong
direction label, the second because the servo moved against its command after a second
re-offset. Now the gripper's open end is read without motion and it only closes, and any
wrong-way motion turns the joint's torque off and stops the run. Cause of the wrong-way
motion still unknown.
