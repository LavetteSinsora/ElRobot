# The Astra harness: an agent drives the ElRobot

GPT-6 Astra, running Codex-style (shell, files, view an image by path, MCP tools), drives the
arm through four MCP tools, on top of `elrobot_deploy`. A port of g1d's exp_011 design
(`ego2g1_v2/experiments/exp_011_astra_manipulation_bench/SPEC.md`, 2026-09-28); section 9 lists
what changed for this arm and why. Built 2026-10-17; run in simulation end to end, **not yet
with Astra or on the real arm**.

```bash
# once: paste ElRobot/astra/codex_config.toml (written by the first launch) into
# ~/.codex/config.toml: the `elrobot` MCP server (the Codex app and CLI share that file)

# the robot (from ElRobot/elrobot_deploy/): check, then press Home; it waits at READY
uv run elrobot_deploy astra
#   --sim --fake-cameras   the whole thing without hardware
#   --dry-run              the real arm read, torque off (only the dashboard's ghost moves)

# Codex (the app, or `codex -C ElRobot/astra`) opened on the workspace ElRobot/astra/:
# it reads AGENTS.md by itself. Give it the task; Astra calls `start(task)`, works, `finish`es.
# The arm goes home; the next task is the next episode, as many as you like.
```

## 1. What it is

1. **Tools**: `start`, `send_action`, `inspect_observation`, `finish` (sections 5-7).
2. **A workspace** (`[astra] workspace`, `ElRobot/astra/`, not versioned) that Codex is
   opened on, with one folder per episode: everything Astra may want to look at, readable
   with `ls`, `cat`, `jq`, `sqlite3` and the image viewer (sections 3-4).
3. **`AGENTS.md`** at the workspace root (`src/elrobot_deploy/astra/agents_template.md`
   filled in from the profile at every launch), which Codex loads by itself before the first
   message: the robot, how an episode goes, frames, units, the tools, the folder.
4. **Wiring into elrobot_deploy**: the Astra source (P12), the recorder (P9), the harness
   process, the MCP server (section 8).

## 2. Conventions (what Astra sees)

| quantity | convention |
|---|---|
| time index | **`tick`**: the anchor camera's (`[astra] anchor_camera`, the wrist) frame index, 30 Hz, from the episode's start, **through holds**. Deploy's `k` is never shown. |
| time | `t`: seconds since tick 0. One `t` per tick: every camera and the state as if taken then. |
| base frame | **x forward, y left, z up** (REP-103): the URDF's `base_link` itself (rebased 2026-10-18, `description/elrobot_ours/make_description.py` step 0). `astra/frames.py`'s `TaskFrame` is then the identity; it stays so another description needs only `[teleop] base_forward / base_up`. |
| pose | the **virtual TCP** (the URDF's `tcp` link), base frame; metres (3 decimals); roll / pitch / yaw degrees, `R = Rz(yaw)·Ry(pitch)·Rx(roll)`. No quaternions. |
| TCP axes | **x out of the jaws**, **y the jaw-opening direction**, z = x × y: the URDF's `tcp` frame as it is (checked by FK: at the zero pose its axes are the base axes, rpy ≈ 0; at home pitch = +45.5°, TCP at [0.246, 0.000, 0.096] m). |
| gripper | **opening in mm**, in and out: 55 mm at the open end (3°, ruler, 2026-10-17) to 0 at the closed end of the calibrated range (139.8°), linear (`GripperMap`). Astra never sees servo angles. |
| cameras | `exo` (an iPhone on a stand, Continuity Camera, 1280×720) and `wrist` (the HX200S fisheye), in that order left to right. Either may be absent; `start`'s reply then says so. |
| fisheye views | the circle of the recorded lens square, outside it black, no undistortion. |

Lag and sag are not mentioned to Astra (decided 2026-10-17): the result table shows commanded
next to measured, and that is enough.

## 3. The workspace and its episodes

```
ElRobot/astra/                       [astra] workspace; Codex's working directory
  AGENTS.md                          rewritten at every launch from the profile
  codex_config.toml                  the [mcp_servers.elrobot] table to paste, once
  episodes/current -> <newest>
  episodes/2026-10-18_143200_put_the_red_cube_in_the_cup/
    episode.json        task, robot, limits, home TCP pose, clock (tick 0), cameras, the
                        record, Astra's finish verdict + an empty human_outcome
    record -> ../../../logs/records/<stamp>     P9's raw record of the rollout
    frames/<cam>/tNNN/tickNNNNNN.jpg            every tick's frame of every camera (P9's bytes)
    timeline.sqlite     one row per tick (section 4)
    actions.jsonl       one line per action: folder, tick range, waypoints, interrupted
    actions/NNNN/       decision step N:
      observation_now.jpg       strip A Astra decided action N from
      observation_history.jpg   sheet H2 of action N-1's motion (absent in 0001)
      observation.txt           the exact text Astra had (action N-1's reply / `start`'s)
      action.json               what Astra sent for action N
      execution_outcome.json    how action N went (section 5.2)
      inspections/NNN.jpg       inspect_observation images made while deciding N
      rejected.jsonl            refused send_action calls made while deciding N
```

Paths in replies are relative to the workspace (`episodes/<id>/actions/0003/...`), so Astra
opens them as given; every image a tool returns is also named in the text, for a client that
does not pass MCP images on.

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
[image episodes/2026-10-18_143200_touch_the_table/actions/0002/observation_now.jpg: exo | wrist]

Virtual TCP, base frame:  position [ 0.253,  0.020,  0.061] m   rotation (roll, pitch, yaw) [  20.5,   41.6,   27.6]°   gripper opening 10 mm

Your next action's waypoints are offsets from the arm's last commanded pose, in its TCP frame.

# Action 1 result (tick 136 → 213, 2.6 s)
[image episodes/2026-10-18_143200_touch_the_table/actions/0002/observation_history.jpg: rows exo, wrist; columns tick 136, 162, 187]

 wp  deadline tick   commanded                                     measured at deadline
  1  154             [0.267, 0.000, 0.075] m  [2, 46, 0]°   40 mm  [0.267, 0.000, 0.075] m  [2, 46, 1]°   40 mm
  2  182             [0.253, 0.021, 0.061] m  [21, 42, 28]°   40 mm  [0.254, 0.020, 0.061] m  [20, 42, 27]°   40 mm
  3  204             [0.253, 0.021, 0.061] m  [21, 42, 28]°   10 mm  [0.253, 0.021, 0.061] m  [20, 42, 28]°   10 mm
Largest IK residual: 0.1 mm / 0.0° at tick 144.

For more detail:
  this action's full result:  episodes/2026-10-18_143200_touch_the_table/actions/0001/execution_outcome.json
  per-tick data:              sqlite3 episodes/2026-10-18_143200_touch_the_table/timeline.sqlite "SELECT tick, t, meas_z_m, cmd_z_m FROM timeline WHERE action_id = 1"
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

## 7. `start` and `finish`: episodes

- **The launch**: `elrobot_deploy astra` writes the workspace's `AGENTS.md` and
  `codex_config.toml`, runs CHECK, and waits for **Home** (the dashboard's button or Enter; the
  one step where the arm moves on the person's say-so). Then the arm holds at home, P0 at READY.
- **`start(task)`** (decided 2026-10-18: Astra begins episodes itself, the person starts the
  runtime): the harness requests START (no motion: it only hands the arm over), finds P9's
  record of the new rollout, opens `episodes/<stamp>_<task>/`, waits for 6 ticks, writes
  `actions/0001` (strip A + "Current observation") and replies with the episode's folder, the
  cameras running and the opening observation. Refused while an episode runs. Called while the
  arm is still homing after the last episode, it waits for READY (up to 30 s). If the person
  pressed Start themselves, `start` adopts that rollout.
- **`finish(success, summary, lessons?)`**: the verdict goes into `episode.json` (`finish`: the
  three fields, tick, `t`, number of actions, `human_outcome: null` for you); the harness
  requests RESET, the arm goes home, P0 waits at READY for the next `start`. Any number of
  episodes per launch; Ctrl-C (or Exit) ends the launch: exit ramp to rest, torque off.
- A rollout that ends any other way (Reset or Exit pressed, a guard) ends its episode: the
  robot tools say so, and `start` begins a new one.

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
`[record] dir` (`logs/records/<stamp>/`; an episode links to its own): `state.bin`
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
  | Unix socket  /tmp/elrobot-astra/harness.sock   (fixed: the MCP server needs no argument)
harness          astra/harness.py, a child of P0, for the whole launch: the tools, the episodes,
  |              images, timelines; tails P9's records; writes only control.request
  |              (START at `start`, RESET at `finish`)
  | Unix socket  /tmp/elrobot-astra/player.sock
P12              the Astra source -> intervention ring -> P6 (IK) -> P7 (bus) -> the arm
```

Sockets live under `/tmp` because macOS caps a socket path at 104 bytes and the workspace
(under this repo) is far longer. One launch at a time on a machine (they share the sockets).

### 8.4 Codex

Registered once: the `[mcp_servers.elrobot]` table in `~/.codex/config.toml` (the launch
writes it to `ElRobot/astra/codex_config.toml`: the venv's python, `-m
elrobot_deploy.astra.mcp_server`, `tool_timeout_sec = 900` because one `send_action` blocks for
the whole motion and Codex's default is 60 s). The Codex app and the CLI read the same file.
Then Codex is opened on the workspace; `AGENTS.md` there is in Astra's context before the first
message. **Not yet run: Codex is not installed on this Mac; the first runs will be in the Codex
app.** Whether it passes MCP image results to the model is to be seen: every reply also names
its images' paths.

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
- **The Codex app**: the MCP table (section 8.4), its tool-call timeout, whether MCP images
  reach the model.
- **The exo iPhone** streams through AVFoundation as a Continuity Camera (listed on this Mac,
  2026-10-17, model iPhone15,3, 1280×720 @ 30); not yet streamed in a session. USB bandwidth is
  not an issue for it (WiFi), but the wrist camera and the servo adapter share a USB 2 hub.
- **Gripper map**: linear from one ruler measurement; a caliper sweep would confirm it.
- How many waypoints Astra uses (whether to cap); whether `settle_s` 0.3 s is right on the arm.
