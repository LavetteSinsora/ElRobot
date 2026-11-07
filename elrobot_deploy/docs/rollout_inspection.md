# Rollout inspection

From `elrobot_deploy/`:

```bash
uv run elrobot_deploy inspect
# One recording, or another root:
uv run elrobot_deploy inspect ../logs/records/20261018-120025
```

Open **http://127.0.0.1:8770**. The inspector reads files only and can run after the robot
session has ended. It never opens a serial port, attaches to the live arena, or sends commands.
`--port` changes its port; `--episodes DIR` selects an Astra episodes directory.

This ports the core workflow of g1d's rollout inspector to `elrobot-record/1`: select a record,
scrub/play synchronized camera views and a 3D arm, compare measured joints, servo commands and
IK joint targets, then inspect the requested TCP and position/orientation residuals. Camera
images use their recorded capture timestamps. Joint sampling uses the latest measurement at
or before the playhead. The camera timestamp offset is displayed so a stalled camera is visible.

The joint chart can show the whole rollout or a 10/2-second window around the playhead. Click
the chart to seek. Select any of the eight joints, including the gripper. State charts are
downsampled for display; the playhead values and JPEG selection use full-resolution indices.
The orientation residual is computed independently from the recorded IK joints and pose,
using the local model; older action summaries report orientation only at the worst position
error and can hide large orientation errors elsewhere.

Astra episodes are associated by their `episode.json` record path. Action markers jump to
each action; the panel shows waypoints, planned/actual duration, results and interruptions.
Session events expose recorded transitions such as `RUNNING → RECOVERING → EXITING`.
Existing recordings do **not** persist the guard's fault text; the inspector does not invent it.

The model is read from the URDF path saved in the recording. This format does not embed a
URDF snapshot, so the page labels that provenance. If the model is missing, camera and joint
inspection still work. Use Reload to refresh a recording that was still being written.
The UI and three.js assets are served locally; no network services are required.

g1d's policy-checkpoint replay and alternative-IK benchmarking modes are not ported here.
The reference implementation remains in `legacy_g1d/`.
