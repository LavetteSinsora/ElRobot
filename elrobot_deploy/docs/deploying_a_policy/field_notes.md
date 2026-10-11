# Field notes: things learned on the robot

Know-how collected from real runs, written down as it is learned. Not yet
organised; entries will move into the other docs once there are enough of
them to see where they belong. Newest first.

- **Read first:** `docs/deploying_a_policy/running_and_live_dashboard.md`.
- **Code:** `src/g1d_deploy/runtime/launcher/startup_checks.py` (the startup
  checks whose refusals these notes are about).

---

## A refusal right after the robot was switched on: often just "not up yet"

*2026-09-30.* Several startup refusals describe a service on the robot board
that is not answering — most visibly the gripper one:

> Note P1 pairs the two topics, so ONE silent gripper stops the latch: this
> refusal does not say which. … Check, in order: 1. the Dex-1 service is
> running on the robot board, for BOTH grippers …

The message lists what can be broken, but the commonest cause is simpler:
**the robot was only just started, and the board's services (the Dex-1
gripper server, the camera server) have not finished coming up.** Wait
half a minute and launch again; the refusal usually does not come back. Only
if it does, work through the checklist in the message.

The same goes for a camera that "connected, but no frame" or a state stream
that is silent in the first minute after power-on: relaunch before debugging.

(Code: `src/g1d_deploy/runtime/launcher/startup_checks.py`, the gripper-state
refusal.)
