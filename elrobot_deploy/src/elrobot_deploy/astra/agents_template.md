# Driving the ElRobot

You drive a real robot arm through four MCP tools of the `elrobot` server: `start`,
`send_action`, `inspect_observation`, `finish`. This folder is your workspace: every episode you
run gets its own folder under `episodes/`, with every reply you got, every camera frame and every
measurement. Read files with `ls`, `cat`, `jq`, `sqlite3`, and open any image by its path.

(This file is written by `elrobot_deploy astra` each time the robot runtime starts. If a tool
says the robot is not reachable, ask the person to start it: `uv run elrobot_deploy astra` in
`ElRobot/elrobot_deploy/`, then Home.)

## How an episode goes

1. The person gives you a task in the chat.
2. **Call `start(task)`** with that task in their words. The robot begins an episode from its home
   pose; the reply names the episode's folder and gives the current observation (an image of
   every camera and the arm's pose).
3. Call `send_action` as often as you need. Each reply is the next observation.
4. **Call `finish(success, summary, lessons?)`** when the task is done or you cannot go on. The
   robot goes home; a new task means a new `start`.

## The robot

ElRobot: one 7-joint arm on a table, with a two-finger parallel gripper. Cameras, in the order
every image shows them, left to right: $camera_list. (The `start` reply says which are running.)

## Frames and units

- **Base frame**: fixed to the arm's base, **x forward** (away from the base, into the
  workspace), **y left**, **z up**. Metres.
- **Virtual TCP** (the point you move): between the two fingertips. Its own axes: **x points
  out of the jaws** (the approach direction), **y is the direction the fingers open along**,
  **z = x × y**. When the arm points straight forward, level, the TCP axes are the base axes
  (roll = pitch = yaw = 0); pitch > 0 tips the jaws down.
- **Rotation**: roll, pitch, yaw in degrees about the fixed x, y, z axes:
  `R = Rz(yaw) · Ry(pitch) · Rx(roll)` (scipy `Rotation.from_euler("xyz", [roll, pitch, yaw],
  degrees=True)`).
- **Gripper**: the opening between the fingers' inner faces, in millimetres:
  $gripper_closed_mm mm (closed) to $gripper_open_mm mm (fully open).
- **Time**: `tick` = the frame index of the $anchor camera, 30 per second, counted from the
  episode's start and through every pause. `t` = seconds since the episode's start. Every camera
  and every measurement of a tick is presented as taken at that `t`.

Every episode starts at the home pose: $home_pose.

## The tools

### `start(task)`

Begins an episode with this task. Refused while one is running (`finish` it first) or while the
robot is not ready (it says why).

### `send_action(waypoints, speed?)`

```json
{"waypoints": [
   {"xyz": [0.04, 0, 0],     "rpy": [0, 0, 0], "gripper_mm": 50},
   {"xyz": [0.04, 0, -0.03], "rpy": [0, 0, 0], "gripper_mm": 50},
   {"xyz": [0.04, 0, -0.03], "rpy": [0, 0, 0], "gripper_mm": 0}
 ],
 "speed": "normal"}
```

- Every waypoint is an **offset from the arm's last commanded pose (the reference), in the
  reference's TCP frame**: `xyz` metres along the TCP's x / y / z, `rpy` degrees, then the
  gripper opening it should have there. All waypoints are offsets from the same reference,
  not from each other. `xyz = [0,0,0], rpy = [0,0,0]` is "stay where you are".
- The reference is the pose the previous action's last waypoint commanded (its last
  "commanded" row in the result); for an episode's first action, the home pose.
- All three fields are required on every waypoint.
- **The timing is chosen for you.** The arm moves through the waypoints in order **without
  stopping at them**, on a smooth path that starts and ends at rest and never overshoots between
  two waypoints. Each waypoint has a **deadline tick**: when the command reaches it. The
  duration follows from the distances and the arm's speed limits (peaks about $max_speed_m_s m/s,
  $max_rot_deg_s °/s, gripper $max_gripper_mm_s mm/s; slower where a joint would be too fast).
  `"speed": "slow"` takes twice as long.
- A waypoint rotated $max_rotation_deg° or more from the reference is refused: turn further in
  two actions.
- The reply arrives when the motion is over: the current observation (image + pose), the
  action's history (the start, ⅓ and ⅔ of the motion, every camera), and per waypoint the
  commanded pose next to the measured pose at its deadline tick, and the largest IK residual
  (how far the joints could not reach what was commanded).
- A refused action (malformed, out of range, sent while another runs) moves nothing; the reply
  lists every problem.
- The arm holds still between actions, however long you think.

### `inspect_observation(ticks, cams?)`

Labelled images of past ticks of the current episode: 1 tick → every camera at that tick; 2 to
4 ticks → one column per tick; more → several images of up to 4 columns. `cams` picks a subset
(fewer rows, larger views). A single raw frame needs no tool: open its JPEG by path (below).

### `finish(success, summary, lessons?)`

Ends the episode: your verdict (`success`: do you think the task is done), what you did, and
anything you learnt that you would want to know next time. The robot then goes home.

**Images.** Every image a tool returns is also a file, and the reply's text gives its path
(`[image episodes/.../observation_now.jpg: ...]`). If an image did not come through, open it by
that path.

## The folder

```
AGENTS.md                    this file
episodes/current             the newest episode
episodes/<episode>/
  episode.json               the task, the robot, the limits, the home pose, your finish verdict
  timeline.sqlite            one row per tick (below)
  actions.jsonl              one line per action: its folder, tick range, waypoints
  actions/NNNN/              decision step N:
    observation_now.jpg        what you saw when deciding action N
    observation_history.jpg    ... the motion of action N-1 (absent in 0001)
    observation.txt            the exact text you had (action N-1's reply; `start`'s in 0001)
    action.json                what you sent for action N
    execution_outcome.json     how action N went: deadline ticks, commanded vs measured
    inspections/               the inspect_observation images you looked at while deciding N
  frames/<camera>/tNNN/tickNNNNNN.jpg   every camera frame, named by its tick (t000 = 0-999)
  record/                    the raw recording (bus reads at 250 Hz, targets, every frame)
```

Paths in replies are relative to this workspace. Earlier episodes stay: compare them, reuse what
worked.

## timeline.sqlite

```sh
sqlite3 episodes/current/timeline.sqlite "SELECT * FROM column_docs"     # what every column means
sqlite3 episodes/current/timeline.sqlite "SELECT * FROM timeline WHERE tick = 120"
sqlite3 episodes/current/timeline.sqlite "SELECT tick, t, meas_z_m, cmd_z_m, meas_gripper_mm FROM timeline WHERE action_id = 3"
sqlite3 episodes/current/timeline.sqlite "SELECT tick, json_extract(meas_load_frac, '$$[7]') FROM timeline WHERE tick BETWEEN 300 AND 400"
```

`action_id` is the action whose commands drive the tick (NULL while holding, which includes
while you think); `phase` is `executing` or `holding`. `meas_*` is measured (from the joint
encoders), `cmd_*` the commanded path. The joints, in `joints` order, are the 7 arm joints from
the base, then the gripper; `meas_load_frac` is each joint's load as a fraction of its stall
torque (the gripper's rises when it squeezes something). Frame paths in the timeline are relative
to the episode's folder.
