# Upstream ElRobot follower description

These files are copied unchanged from NormaCore (Apache-2.0, see `LICENSE`):

- Repo: https://github.com/norma-core/norma-core
- Commit: `3aa88d4ade5a1ed287f2a8bdc0677db208572b86` (2026-10-08)
- Path: `hardware/elrobot/simulation/` (`elrobot_follower.urdf` and `assets/`)

Do not edit them here. Our customised description (camera mount, TCP frame, measured
limits) will live next to this folder and point at these meshes.

## Notes on the upstream URDF

- Exported from CAD: every joint `rpy` is 0, so all link frames share the base orientation
  at q = 0. z is up, and the arm reaches along +y.
- Zero pose: upper arm vertical, wrist and gripper horizontal along +y.
- Joint axes are tilted 0.5 to 1 degree from the coordinate axes (CAD export artefact).
- The gripper jaw prismatic joints (`rev_motor_08_1`, `rev_motor_08_2`) put their frames
  about 0.3 m from the gripper; the mesh origins cancel this out. They render correctly but
  are not usable as kinematic frames. There is no TCP frame.
- `effort` and `velocity` limits are the same placeholder on every joint.
