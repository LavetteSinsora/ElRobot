"""P0: the launcher. Starts every process, CHECKs the robot and the streams,
homes the arm, hands over to the operator, and brings everything back down.

    state_machine.py    `Launcher`: INIT -> CHECK -> HOME -> WARMUP -> READY
                        <-> ACTIVE -> EXIT, and the damp
    startup_checks.py   the CHECK verifiers (`StartupChecks`, mixed in)
    children.py         starting the child processes by module path
    ramps.py            homing and exit ramps

What an operator sees of it: `docs/deploying_a_policy/running_and_live_dashboard.md`,
"A launch, start to finish".
"""

from g1d_deploy.runtime.launcher.state_machine import Launcher, run

__all__ = ["Launcher", "run"]
