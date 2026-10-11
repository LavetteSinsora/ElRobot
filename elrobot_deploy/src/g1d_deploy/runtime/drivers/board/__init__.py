"""Things `g1d_deploy` does to the ROBOT BOARD over SSH, rather than to itself.

One module so far: `image_server_launch.py`, which starts the fisheye image
server (`fisheye_image_server/`, shipped here and synced to the board) and
stops the vendor's, because the two cannot both own the cameras. Called by P0
at CHECK (`runtime/launcher/startup_checks.py`); the robot-file side is
`docs/setup/robot_toml.md`, "`[image_server]`".

Nothing in the control path lives here. These are startup actions, they all
happen before CHECK confirms the cameras produce, and every one of them is a
refusal rather than a retry — a board that cannot be reached is a run that
cannot happen, and finding that out in one second beats finding it out after
a policy's warmup.
"""

from g1d_deploy.runtime.drivers.board.image_server_launch import (BoardImagerError,
                                                                  ensure_running, stop)

__all__ = ["BoardImagerError", "ensure_running", "stop"]
