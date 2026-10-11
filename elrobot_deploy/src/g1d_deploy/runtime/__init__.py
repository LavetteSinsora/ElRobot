"""Everything that runs and controls the robot.

    launcher/         P0: start every process, CHECK, home, hand over, exit
    processes/        P1-P11, one module each, spawned by P0 by module path
    shared_memory/    the arena every process reads and writes
    ik/               pose targets -> joint targets (P6): the four shipped solvers
    execution/        joint targets -> the 500 Hz motor commands: chunk blending,
                      the servo reference filter, segment interpolation
    drivers/          the hardware: DDS topics, cameras, the robot's board
    teleop/           PICO takeover: the maths and the PC service
    timing.py         the monotonic clock and pacing
    cpu_pinning.py    which process runs on which core

Deliberately empty of imports: P5 runs under the policy's own interpreter and
imports `shared_memory`, `timing` and `processes.p5_policy` from here.

The processes and one chunk's path through them: `docs/overview.md`.
"""
