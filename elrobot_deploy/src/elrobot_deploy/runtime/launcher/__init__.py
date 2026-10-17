"""P0: the launcher. Creates the arena, starts every process, CHECKs that they produce, homes
the arm, hands it to the operator, and brings everything back down.

    state_machine.py    `Launcher`: INIT -> CHECK -> HOME -> READY <-> ACTIVE -> EXITING -> STOPPED
    children.py         starting the child processes by module path
    ramps.py            the homing, reset and exit ramps
"""

from elrobot_deploy.runtime.launcher.state_machine import BusFault, Launcher, run

__all__ = ["BusFault", "Launcher", "run"]
