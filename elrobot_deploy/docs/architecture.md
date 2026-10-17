# elrobot_deploy: how it fits together

The runtime that drives the ElRobot follower (one 7-DoF arm plus a gripper, 8 Feetech ST3215
servos on one serial bus) from an iPhone. A port of g1d_deploy (`../UPSTREAM.md`), rebuilt for
one arm; the decisions behind it are `../PORTING_PLAN.md`, the g1d code not carried over is
`../legacy_g1d/`.

## The chain

```
 iPhone (ARKit, ~60 Hz)                                         this Mac
 ──────────────────────    UDP JSON     ┌────────────────────────────────────────────────────┐
 pose, track, world_epoch ─────────────▶│ P11 phone source   clutch, room-frame motion,      │
 engage / grip buttons    ◀── echo ─────│                    gripper, deadlines, guards      │
                                         │        │ intervene ring: motion since the press   │
                                         │        ▼                                           │
                                         │ P6 target          anchor TCP, IK (one interface), │
                                         │                    grid, velocity clamp, box       │
                                         │        │ target ring: 8 joints + deadline per tick │
                                         │        ▼                                           │
                                         │ P7 bus process     segment walk, 15 Hz filter,     │
                                         │  (owns the port)   clamps, squeeze bound, guards   │──▶ 8 × ST3215
                                         │        │ state ring: measured q, command, load ... │◀── (250 Hz)
                                         │ P0 launcher        sessions, homing/exit ramps     │
                                         └────────────────────────────────────────────────────┘
```

Every arrow between processes is a region of one shared-memory **arena**, and every region has
exactly one writing process. Nothing calls anything else.

## The processes

| | process | rate | writes | module |
|---|---|---|---|---|
| P0 | launcher | | `session`, ramps into the target ring, `release_torque` | `runtime/launcher/` |
| P6 | target | per phone frame (60 Hz) | the target ring | `runtime/processes/p6_target.py` |
| P7 | bus process | 250 Hz | `k_exec`, the state ring, `bus`, `fault` | `runtime/processes/p7_bus.py` |
| P10 | dashboard | the page polls at 20 Hz | `request` (its buttons) | `dashboard/` |
| P11 | phone source | per packet (60 Hz) | `intervene`, the intervention ring | `runtime/processes/p11_phone.py` |

P7 replaces g1d's P1 (state), P7 (body) and P8 (hand): the bus is half-duplex, so one process
owns the port, reads and writes. It runs for the whole session and is never restarted inside
one (that would release torque), so ticks (`k`) count up across rollouts and never rewind.

## A session

```
INIT -> CHECK -(Home)-> HOME -> READY -(Start)-> ACTIVE -> EXITING -> STOPPED
                                  ^                |
                                  └──── HOME ◀─────┘ (Reset)
```

- **INIT**: create the arena; start the dashboard (P10).
- **CHECK**: start the bus process. Before it writes anything it runs the **hardware check**
  (`St3215Driver.check`): every servo answers, is an STS3215 in position mode, has its
  calibrated offset, has torque off, reads a sane supply voltage and temperature, has no alarm
  bit, and reads inside its calibrated range (8 deg of slack: the arm rests on its stops). The
  report is printed as a table and shown on the dashboard; any problem refuses the start. Then
  it turns torque on holding the measured posture (in a dry run: not) and sets its `ready`
  latch. Then the phone source, and CHECK **waits for the phone** (as g1d's
  did for the PICO): it must stream at 30 Hz or more for a second with ARKit tracking normal.
  The terminal and the dashboard say what is missing (nothing received, a low rate, tracking).
  `--no-phone` skips the phone entirely.
- **Wait for Home**: the arm is held where it was found; the dashboard draws the home pose as a
  green ghost beside it. Nothing moves until **Home** (the dashboard's button, or Enter in the
  terminal; `--auto-home` skips the wait).
- **HOME**: a cosine ramp from the held command to `[poses] home`, written into the target
  ring and executed like any other ticks; then the measured posture is checked against home.
- **READY**: the arm holds at home until **Start** (button or Enter; `--auto-start` skips).
- **ACTIVE**: P6 runs. The arm moves only while the operator holds the clutch.
- **RESET** (a request): P6 stops, the arm ramps home, READY again.
- **EXITING** (Ctrl-C or an EXIT request): P6 stops, the phone source hands back, the arm
  ramps to `[poses] exit` (the rest pose). Then **STOPPED**: every child sees it and closes
  itself; the bus process turns torque off **only if** the exit ramp completed (and
  `release_at_exit`).

**The dashboard and Enter only REQUEST** (`control.request`): HOME, START, RESET, EXIT. P0 takes
a request only when it is the one it waits for (`control.wait`), so a stray click does nothing.

**The guards** (`runtime/execution/guard.py`, thresholds in the profile's `[safety]`):
a **runaway** is a joint more than 25 deg off its command AND moving away from it (its error grew
10 deg within 0.3 s, the joint moving that way), or more than 60 deg off whatever it does (not
the gripper); a lagging joint moves toward its command and never counts. Its torque goes off
at once. **Blocked** is a joint more than 15 deg behind and not moving (less than 3 deg) for
1.5 s; not the gripper, which an object holds off its command. `run --runaway-only` (or
`safety.blocked = false`) turns the blocked rule off. Either one **recovers**: the bus
process keeps following the ring (`BUS_RECOVERING`), P0 ramps the arm straight to the exit
pose (through home first with `recover_via_home`), then torque goes off. **Unreadable** (5 failed reads in a row): the bus process cannot
see the arm, so it HOLDS and the session ends with torque on; as does a second fault during a
recovery.

**The arm is never dropped by software.** A failure, a hold or a second Ctrl-C (which skips
the exit ramp) all end the session with torque left ON: the servos hold the last goal, P0
says so, and the operator supports the arm and runs `uv run python -m elrobot.relax`.
(One Ctrl-C under `uv run` reaches the process twice; two within 0.5 s count as one.)

## Dry runs

- `run --sim`: no hardware. `SimDriver` stands in for the bus (first-order lags at the real
  bus's timing), everything else is the real code; with `fake-phone` there is no phone either.
- `run --dry-run`: **the real arm, torque OFF**. The bus process connects passively (the
  hardware check runs, torque is never enabled, no goal is ever written), reads the arm every
  tick, and still walks the ring, filters and clamps every command and publishes it as `q_cmd`.
  So homing, Start, the phone, P6 and the IK all run for real, and the dashboard shows the
  commanded arm as the blue ghost moving while the measured arm lies still. The lag and
  runaway guards are off (the arm is not meant to follow), the homing check is skipped.
- `view`: no session at all. The dashboard's 3D view of the URDF with the profile's home and
  exit poses as ghosts and a slider per joint, to look at a pose before using it.

## The dashboard (`dashboard/`)

`http://127.0.0.1:8080` (`[dashboard] host / port`; `0.0.0.0` to open it from another device,
without a password). A stdlib HTTP server (P10) and one page; three.js is vendored, so it works
offline. The page draws our URDF (`/api/robot`, the meshes from `description/`) up to four
times: the arm as **measured** (solid), as **commanded** by the bus process (blue ghost), the
**home** pose (green ghost) and the exit pose (orange), plus the IK's target point. Beside it:
the session and what P0 waits for, the Home / Start / Reset / Exit buttons (enabled only when
P0 would take them), the phone (streaming, clutch, tracking, latency, forced releases), every
joint (measured, commanded, error, temperature, load) and the hardware check. It polls
`/api/state` 20 times a second and redraws only when something changed.

## Teleoperation

**The clutch.** The arm follows the phone only while ENGAGE is on (a toggle in the app: tap to
start, tap to stop; the app switches it off whenever the Mac releases the arm), and only
relative to where the phone was when it was switched on. A release holds the arm where it is. Pressing
again re-anchors both sides, which is how the operator moves further than their arm reaches or
re-orients themselves. A button already down when a rollout starts (or after a guard fired)
counts only after it has been seen up.

**Room frame** (`runtime/teleop/intervention.py`, `RoomFrame`). The grip: screen facing left,
back camera right, the top pointing forward. At each press the operator's heading is taken
from where the phone's top points, horizontally (ARKit's world is gravity-aligned), and
mapped onto the robot base's forward/left/up (`[teleop] base_forward`, `base_up`; base_link has
+y forward, +z up). Then the phone's displacement since the press moves the TCP the same way
in the room, and its rotation since the press rotates the TCP about the same room axes, about
the TCP itself. `[teleop] scale` scales translation only.

**The anchor** is P6's: at each press it freezes the TCP pose of the command the bus process
is holding (`fk(q_cmd)`), seeds the IK there, and hands that configuration to the IK as its
nominal pose. The phone source never needs a model of the arm: its rows carry motion, not
poses.

**The gripper**: hold-to-close (the pad's forward zone). The jaw command closes at
`gripper_rate` while held and opens at the same rate when released, integrated on the phone's
clock. The command may lead the measured jaw toward closed by at most `squeeze_margin`, which
therefore sets the grip force (P × margin on a position servo); the bounded value is stored
(anti-windup) and the bus process applies the same bound every tick against the freshest read.
While the clutch is up the jaw holds.

**Guards** (each forces a release, which needs a fresh press): tracking not "normal"; the
phone's `world_epoch` changed; one frame moved more than `max_step_m` / `max_step_deg`; no
packet for `stale_s`; the phone pointing straight up or down at the press; the rollout ended.
P6 clamps the TCP into the workspace box (widened to contain the anchor, so a press outside it
never jumps).

## Timing

- **Deadlines** are the phone's capture time mapped onto this Mac's clock by `Pace`: pinned at
  each press to `arrival + lead_s` (40 ms), then advancing with the phone's own clock, so the
  motion replays at the pace it was made, whatever bursts WiFi delivers it in. A packet later
  than the lead covers slides the map later (once, by just enough); the phone source reports,
  per engagement, the latency spread `L_n - L_1` and how far the map slid. That is the number
  to set `lead_s` from.
- **P6** decimates rows onto a 60 Hz grid (`control_hz`) and clamps each target's joint speed
  over the deadline difference to the previous one.
- **P7** walks the target line at 250 Hz: the segment between the two targets around now, read
  `2/w` ahead by a 15 Hz critically damped filter (so it smooths without lagging), then the
  slew clamp, the joint limits and the gripper's squeeze bound, then one sync_write of 8 goals
  and one sync_read. A change of the intervention counter abandons the segment in flight at
  once; an empty ring holds the last command.

## The arena (`runtime/shared_memory/`)

| region | writer | what |
|---|---|---|
| `control` | one per word | `session` (P0), `k_exec` (P7), `request` (requester; P0 clears), `intervene` (P11), `bus` + `fault` (P7), `release_torque` (P0) |
| `state` | P7 | one row per bus read: measured `q`, the command `q_cmd`, speed / load / temperature (every 5th), alarms |
| `target` | P6, P0 | one row per tick: `k`, `deadline`, `intervene`, the 8 joints, the IK's input pose and residual |
| `intervene` | P11 | one row per phone packet while engaged: the motion since the press, the jaw command, the deadline |

`schema.py` is the vocabulary (bump `VERSION` when a record's meaning changes), `layout.py` the
regions and their hash (every process refuses an arena built differently), `rings.py` every
read and write. Records are written under per-slot seqlocks; reads retry until consistent. On
Apple Silicon (ARM) the seqlock needs real memory barriers, which `primitives.fence()` supplies
(`OSMemoryBarrier`); on x86 it is a no-op.

The **intervention counter** (`control.intervene`) is how authority changes hands: odd while
the operator drives, even otherwise. Every row is stamped with it, the bus process only
executes rows of the current value, and any change abandons what was in flight. So a release
needs no wipe and cannot be outrun by an older row.

## The pluggable parts

**The IK** (`runtime/ik/__init__.py`). P6 holds one object: `solve(pose7, seed_q7, budget) ->
(q7, err, ok)`, `fk(q7) -> pose7`, and optionally `set_nominal(q7)`. `[ik] solver` names a
shipped one (`tcp_qp`, the mink QP on our URDF) or any `module:factory` / `file.py:factory`;
the other `[ik]` keys go to the factory. A new strategy is tried with
`--set ik.solver=my_ik.py:build`, without touching P6.

**The executor** (`runtime/execution/bus_servo.py`) and **the servo driver**
(`runtime/drivers/st3215_driver.py`, `ServoDriver`: connect, write, read, read_full, release).
Everything that tunes how the servos follow is in the profile's `[executor]`: filter, friction
compensation per joint, P / D per servo (written at connect, restored at release), torque
limits. `elrobot_deploy executor-test` is the bench for it (sine targets through the same ring
and process, with a lag / dead zone / smoothness summary). `bus.driver = "sim"` swaps in
`SimDriver` (first-order lags at the real bus's timing).

## Configuration

One TOML robot profile, `src/elrobot_deploy/robot/profiles/elrobot.toml` (every key commented
there), resolved by `config_loading/config.py` into a frozen `Config` every process is handed.
Joint limits are the URDF's (`description/elrobot_ours/elrobot.urdf`, the calibrated ranges);
the velocity limit is the URDF's times `[limits] velocity_scale`. Override any key for one run:

```bash
uv run elrobot_deploy run --set teleop.lead_s=0.05 --set 'executor.friction=[0,0,0,0,1,1,1,0]'
uv run elrobot_deploy config --set ...        # print what a run would use
```

## Commands (from `ElRobot/elrobot_deploy/`)

```bash
uv sync --extra runtime                          # once
uv run pytest -q                                 # the tests (no hardware)
uv run elrobot_deploy view                       # the URDF and the poses in the browser, no robot
uv run elrobot_deploy phone-bench                # the grip, with the phone and no robot
uv run elrobot_deploy run --sim                  # the whole chain on simulated servos
uv run elrobot_deploy fake-phone                 # ... driven by a synthetic phone instead
uv run elrobot_deploy run --dry-run              # the real arm read, torque OFF: ghost only
uv run elrobot_deploy run --no-phone             # the real arm: check, home, hold, exit
uv run elrobot_deploy run                        # the real arm, driven by the phone
#   run flags: --auto-home, --auto-start (skip the waits), --no-dashboard (Enter only)
uv run elrobot_deploy executor-test --joints 6   # the executor's sine test
```

The bus process's tick log (every tick: the line, the command, the measurement, timing; the
last ten minutes of a session) is written to `ElRobot/logs/sessions/<stamp>_bus.csv` at exit;
`p7_bus.load_log` reads it back.

## Not here yet

Policies (P5, the policy contract, chunk blending), recording (P9; today only the bus
process's tick log), rollout inspection and cameras (P2) are g1d's, in `../legacy_g1d/`, to be
ported when their milestone comes (`../PORTING_PLAN.md` section 4).
