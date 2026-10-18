# The Astra harness: an agent drives the ElRobot

GPT-6 Astra, running Codex-style (shell, files, view an image by path, MCP tools), drives the
arm through three MCP tools, on top of `elrobot_deploy`. A port of g1d's exp_011 design
(`ego2g1_v2/experiments/exp_011_astra_manipulation_bench/SPEC.md`, 2026-09-28); section 9 lists
what changed for this arm and why. Built 2026-10-17; run in simulation end to end, **not yet
with Astra or on the real arm**.

```bash
# terminal 1 (from ElRobot/elrobot_deploy/): the session. Home and Start as usual
uv run elrobot_deploy astra --task "put the red cube in the cup"
#   --sim --fake-cameras   the whole thing without hardware
#   --dry-run              the real arm read, torque off (only the dashboard's ghost moves)
# terminal 2, once terminal 1 says READY: Astra
logs/astra/current/start_codex.sh          # ASTRA_MODEL=<model> to pick the model
```

## 1. What it is

1. **Tools**: `send_action`, `inspect_observation`, `finish` (sections 5-7).
2. **A session folder** with everything Astra may want to look at, readable with `ls`, `cat`,
   `jq`, `sqlite3` and the image viewer (sections 3-4).
3. **A bootstrap doc**, the session's `README.md` (`src/elrobot_deploy/astra/session_readme.md`
   filled in): frames, units, the tools, the folder.
4. **Wiring into elrobot_deploy**: the Astra source (P12), the recorder (P9), the harness
   process, the MCP server (section 8).

## 2. Conventions (what Astra sees)

| quantity | convention |
|---|---|
| time index | **`tick`**: the anchor camera's (`[astra] anchor_camera`, the wrist) frame index, 30 Hz, from the session start, **through holds**. Deploy's `k` is never shown. |
| time | `t`: seconds since tick 0. One `t` per tick: every camera and the state as if taken then. |
| base frame | **x forward, y left, z up** (REP-103): the URDF's `base_link` itself (rebased 2026-10-18, `description/elrobot_ours/make_description.py` step 0). `astra/frames.py`'s `TaskFrame` is then the identity; it stays so another description needs only `[teleop] base_forward / base_up`. |
| pose | the **virtual TCP** (the URDF's `tcp` link), base frame; metres (3 decimals); roll / pitch / yaw degrees, `R = Rz(yaw)·Ry(pitch)·Rx(roll)`. No quaternions. |
| TCP axes | **x out of the jaws**, **y the jaw-opening direction**, z = x × y: the URDF's `tcp` frame as it is (checked by FK: at the zero pose its axes are the base axes, rpy ≈ 0; at home pitch = +45.5°, TCP at [0.246, 0.000, 0.096] m). |
| gripper | **opening in mm**, in and out: 55 mm at the open end (3°, ruler, 2026-10-17) to 0 at the closed end of the calibrated range (139.8°), linear (`GripperMap`). Astra never sees servo angles. |
| cameras | `exo` (an iPhone on a stand, Continuity Camera, 1280×720) and `wrist` (the HX200S fisheye), in that order left to right. Either may be absent; the README then says so. |
| fisheye views | the circle of the recorded lens square, outside it black, no undistortion. |

Lag and sag are not mentioned to Astra (decided 2026-10-17): the result table shows commanded
next to measured, and that is enough.

## 3. Session folder

`[astra] sessions_dir` (default `ElRobot/logs/astra/`), one folder per session, `current`
pointing at the newest:

```
logs/astra/2026-10-17_143200_put_the_red_cube_in_the_cup/
  README.md            the bootstrap doc
  session.json         task, robot, limits, home TCP pose, clock (tick 0), cameras, sockets,
                       Astra's finish verdict + an empty human_outcome
  opening_prompt.md    the opening message (task + "Current observation")
  start_codex.sh       Codex on this session (section 8.4)
  record -> ../../records/<session id>     P9's raw record of the rollout
  frames/<cam>/tNNN/tickNNNNNN.jpg         every tick's frame of every camera (P9's JPEG bytes)
  timeline.sqlite      one row per tick (section 4)
  actions.jsonl        one line per action: folder, tick range, waypoints, interrupted
  actions/NNNN/        decision step N:
    observation_now.jpg       strip A Astra decided action N from
    observation_history.jpg   sheet H2 of action N-1's motion (absent in 0001)
    observation.txt           the exact text Astra had (action N-1's reply / the opening)
    action.json               what Astra sent for action N
    execution_outcome.json    how action N went (section 5.2)
    inspections/NNN.jpg       inspect_observation images made while deciding N
    rejected.jsonl            refused send_action calls made while deciding N
```

Folder N = decision step N (the rule from g1d): what was seen when deciding, what was sent,
how it executed. The images that show action N's motion are what Astra sees before deciding
N+1, so they are in N+1.

## 4. `timeline.sqlite`

One row per tick (`astra/timeline.py`); `SELECT * FROM column_docs` explains every column. One
arm, so no `left_` / `right_` prefixes: `tick` (primary key), `t`, `action_id` (NULL while
holding), `phase` (`executing` / `holding`; no `homing`: the record is the rollout only),
`<cam>_path`, `meas_{x,y,z}_m`, `meas_{roll,pitch,yaw}_deg`, `meas_gripper_mm`, the same
`cmd_*` (the pose the IK was asked for: P6's input), `ik_residual_mm` / `_deg`, and JSON
arrays `meas_q_rad`, `meas_dq_rad_s`, `meas_load_frac` (fraction of stall torque: the
servos report load, not torque), `cmd_q_rad`; tables `joints`, `column_docs`. WAL; the
harness commits every 0.25 s.

## 5. `send_action`

### 5.1 Input

```json
{"waypoints": [
   {"xyz": [0.04, 0, 0],     "rpy": [0, 0, 0], "gripper_mm": 50},
   {"xyz": [0.04, 0, -0.03], "rpy": [0, 0, 0], "gripper_mm": 50},
   {"xyz": [0.04, 0, -0.03], "rpy": [0, 0, 0], "gripper_mm": 0}],
 "speed": "normal"}
```

- Per waypoint `{xyz [m], rpy [deg], gripper_mm}`, all required. One arm: no `left` / `right`,
  no `"hold"`.
- **Relative, one frame**: offsets from the **reference pose** in its TCP frame,
  `p = p_ref + R_ref·dp`, `R = R_ref·R_d`; all from the same reference.
- **Reference** = the last waypoint the previous action composed (so a zero offset holds
  exactly and nothing accumulates); for a rollout's first action (and after an interrupted
  one) `fk` of the command the bus process holds. (g1d used the measured pose for action 1;
  here the held command is as easy to get and carries no sag.)
- **Refused** with an error reply, nothing moves: malformed input, `gripper_mm` outside
  0 … 55, a waypoint rotated ≥ 170° from the reference, another action still running (not
  queued), any robot tool after `finish`, the rollout over (Reset / Exit / a guard). Every
  problem in one reply, naming the waypoint and the field. No reach check: an unreachable
  target shows as IK residual.
- **The timing is ours** (`astra/trajectory.py`): PCHIP through the waypoints, at rest at both
  ends, no stop at a waypoint, no overshoot between two (each coordinate monotone); rotation
  as the rotation vector from the reference; the gripper width interpolated. Each segment gets
  the shortest duration on the 60 Hz grid whose peaks stay inside `[astra]`'s caps (0.10 m/s,
  45°/s, 0.5 m/s², 180°/s², gripper 60 mm/s, ≥ 0.1 s), then P12 **solves the sampled path with
  the IK and stretches every segment whose joint speeds would pass `joint_speed_margin` (0.8) ×
  the run's velocity limit**: the velocity limits we have, applied where they bite. `slow` =
  twice as long. Knots land on grid ticks, so the command at a deadline is exactly the
  composed waypoint.

### 5.2 Reply

```
# Current observation (tick 216, t=7.20 s)
[image observation_now.jpg: exo | wrist]

Virtual TCP, base frame:  position [ 0.253,  0.020,  0.061] m   rotation (roll, pitch, yaw) [  20.5,   41.6,   27.6]°   gripper opening 10 mm

Your next action's waypoints are offsets from the arm's last commanded pose, in its TCP frame.

# Action 1 result (tick 136 → 213, 2.6 s)
[image observation_history.jpg: rows exo, wrist; columns tick 136, 162, 187]

 wp  deadline tick   commanded                                     measured at deadline
  1  154             [0.267, 0.000, 0.075] m  [2, 46, 0]°   40 mm  [0.267, 0.000, 0.075] m  [2, 46, 1]°   40 mm
  2  182             [0.253, 0.021, 0.061] m  [21, 42, 28]°   40 mm  [0.254, 0.020, 0.061] m  [20, 42, 27]°   40 mm
  3  204             [0.253, 0.021, 0.061] m  [21, 42, 28]°   10 mm  [0.253, 0.021, 0.061] m  [20, 42, 28]°   10 mm
Largest IK residual: 0.1 mm / 0.0° at tick 144.

For more detail:
  this action's full result:  actions/0001/execution_outcome.json
  per-tick data:              sqlite3 timeline.sqlite "SELECT tick, t, meas_z_m, cmd_z_m FROM timeline WHERE action_id = 1"
```

(A real reply from a simulated session.) The tick range runs from the first sample to the
release (the last pose is held `settle_s` = 0.3 s first). An interrupted action says so and
why. `execution_outcome.json` holds the same, structured, plus the reference pose and the
planned duration.

### 5.3 Images (`astra/render.py`)

- **Strip A**: `exo | wrist`, views 480 px tall + a 32 px label bar = **exactly 512 px** (one
  tile row). Fisheye 480×480, 16:9 853×480: 1337×512 with both.
- **Sheet H2**: rows = cameras, columns = the action's start, ⅓, ⅔; each column 341 px wide,
  every view its own aspect (fisheye 341×341, 16:9 341×192); a label bar per column.
- Only tick / `t` labels in pixels; every number is in the text.

## 6. `inspect_observation(ticks, cams?)`

1 tick → strip A at that tick; 2-4 → the H2 layout, one column per tick; more → several images
of ≤ 4 columns. `cams` narrows the rows. Saved in the current step's `inspections/`. A raw
frame needs no tool: `frames/<cam>/tNNN/tickNNNNNN.jpg`.

## 7. Start and `finish`

- **Start (no tool)**: `elrobot_deploy astra` runs the usual CHECK / Home / Start gates
  (dashboard buttons or Enter; `--auto-home --auto-start` skip them). At ACTIVE the harness
  waits for 6 ticks, writes `actions/0001` (strip A + "Current observation"),
  `opening_prompt.md` and `start_codex.sh`, and prints `READY`.
- **`finish(success, summary, lessons?)`**: the verdict goes into `session.json` (`finish`:
  the three fields, tick, `t`, number of actions, `human_outcome: null` for you); the harness
  requests RESET, the arm goes home. **One session = one rollout**: after it P0 waits only for
  Exit (dashboard or Ctrl-C), which ramps to rest and releases torque.

## 8. Wiring

### 8.1 The seam: the intervention ring

g1d's design ran Astra as a sync-mode policy adapter inside P5. This runtime has no P5 yet,
and it does not need one: the phone's path, the **intervention ring**, already takes "TCP
motion relative to an anchor, with a deadline, plus the gripper", and a released clutch
already means "the arm holds its last command". So Astra drives exactly as the phone does,
from **P12, the Astra source** (`runtime/processes/p12_astra.py`), which P0 starts at CHECK in
place of P11 (each is the only writer of `control.intervene`, so never both). One action = one
engagement:

1. compose the waypoints on the reference, time them (section 5.1);
2. counter → odd; P6 freezes its anchor = `fk` of the held command, the same configuration P12
   composed against, so each row's delta lands exactly on the composed pose;
3. **probe**: rows at rest until P6 has published a target of this world (proof it is taking
   rows before any row moves the arm; gives up after 5 s);
4. stream one row per 60 Hz tick, each 40 ms (`lead_s`) before its deadline (P6 takes the
   newest row, so a path written at once would jump to its end);
5. hold the last pose `settle_s`, then counter → even: the arm holds.

What this removes from g1d's design: the `P` = 150 slices and their stops, the throwaway row 0,
the `(chunk_id, row)` mapping, the numpy-only constraint. Deadlines are this Mac's monotonic
clock, which the camera frames are stamped on too, so a deadline's tick is the nearest frame.
Safety is the runtime's: P6's and the bus process's speed and limit clamps, the bus guards
(runaway: torque off; recovery to rest), and P12 hands back at once if the session leaves
ACTIVE or the bus stops following. No workspace box (decided 2026-10-17).

### 8.2 P9, the recorder

`runtime/processes/p9_record.py`, ported from g1d: one record per ACTIVE period under
`[record] dir` (`logs/records/`; an Astra session's is named after the session): `state.bin`
(every bus read), `targets.bin`, `intervene.bin`, `session.bin`, per camera `<cam>.jpg.bin` +
`<cam>.idx.bin`, `manifest.json` (dtypes, config, code revision). New against g1d: it encodes
the JPEGs (a fisheye as its lens square), and flushes every 0.1 s so the harness can tail it.
Frames it could not copy before the camera ring lapped them are counted (`dropped`). Also for
teleop: `run --record`. `elrobot_deploy.record` reads a record back (`load`, and `Tail` while
it is written).

### 8.3 Processes

```
Codex / Astra
  | stdio (MCP)
MCP server       astra/mcp_server.py: a child of Codex, stateless, stdlib only
  | Unix socket  /tmp/elrobot-astra-<stamp>/harness.sock   (paths: session.json)
harness          astra/harness.py, a child of P0: the tools, the folder, the images, the timeline;
  |              tails P9's record; writes only control.request (RESET at finish)
  | Unix socket  /tmp/elrobot-astra-<stamp>/player.sock
P12              the Astra source -> intervention ring -> P6 (IK) -> P7 (bus) -> the arm
```

Sockets live under `/tmp` because macOS caps a socket path at 104 bytes and the session folder
(under this repo) is far longer.

### 8.4 Codex

`start_codex.sh` wires the MCP server in with `-c mcp_servers.elrobot.*` overrides (no edit of
`~/.codex/config.toml`), sets `tool_timeout_sec = 900` (one `send_action` blocks for the whole
motion; Codex's default is 60 s), runs in the session folder, and passes the opening prompt
and `--image actions/0001/observation_now.jpg`. **Not yet run: Codex is not installed on this
Mac.** If its flags differ, the server command is in the script.

## 9. What changed from the G1-D design

| G1-D (exp_011) | ElRobot |
|---|---|
| `AstraPolicy` in P5, sync mode, chunk ring | P12 on the intervention ring (section 8.1) |
| reads the P9 record | the same; P9 ported (and encodes JPEG) |
| two arms, `left` / `right`, `"hold"` | one arm, one waypoint format, no hold |
| base x forward (G1 frame) | the same: our URDF's `base_link` was rebased to REP-103 (it had +y forward) |
| TCP 184.8 mm on the flange | the URDF's `tcp` link (axes already x approach, y opening) |
| gripper 0 … 90 mm, Dex-1 rad | 0 … 55 mm, ST3215 rad, linear |
| 0.20 m/s, 1.0 m/s² | 0.10 m/s, 0.5 m/s², + an IK pre-solve against the joint speed limits |
| reach left to the waist IK | no waist: an unreachable target shows as residual (no reject) |
| action 1 from the measured pose | from the held command (no sag offset to accept) |
| `head_left`, `wrist_left`, `wrist_right` | `exo` (iPhone, Continuity Camera), `wrist` |
| lag / sag explained in the README | not mentioned |
| `phase` incl. `homing` | `executing` / `holding` (the record is the rollout only) |

## 10. Open

- **First runs**: `astra --sim --fake-cameras` with Codex; then `--dry-run`; then the arm, slow
  (`--set limits.velocity_scale=0.3`; the joint check follows it).
- **Codex flags** (section 8.4) and its tool-call timeout, on a Mac with Codex.
- **The exo iPhone** streams through AVFoundation as a Continuity Camera (listed on this Mac,
  2026-10-17, model iPhone15,3, 1280×720 @ 30); not yet streamed in a session. USB bandwidth is
  not an issue for it (WiFi), but the wrist camera and the servo adapter share a USB 2 hub.
- **Gripper map**: linear from one ruler measurement; a caliper sweep would confirm it.
- How many waypoints Astra uses (whether to cap); whether `settle_s` 0.3 s is right on the arm.
