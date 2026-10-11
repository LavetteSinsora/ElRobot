"""The fisheye image server — code that runs on the ROBOT BOARD, not here.

This package is shipped here so it is diffable and reviewable, then copied
to the robot file's `[image_server] dir` on the G1-D's Jetson board (by P0, at
every CHECK: `../image_server_launch.py`) and run there under the robot file's
conda environment. Nothing in `g1d_deploy` imports it, and it
imports nothing from `g1d_deploy` — the two run on different machines, under
different Python versions (3.10 on the board, the runtime's own on the deploy box).

Every board-only dependency (`uvc`, `zmq`, `yaml`) is imported INSIDE the
function that needs it, never at module scope. That is not style: the tests
import these modules on the deploy box, where `uvc` (pyuvc) does not exist. A
top-level import here would fail on every machine that is not the robot.

See `README.md` beside this file for the design, the wire format and the board
facts it rests on. The `.py` and `.yaml` files here are what the board runs:
any change to them is synced to the board and restarts its server at the next
launch.
"""
