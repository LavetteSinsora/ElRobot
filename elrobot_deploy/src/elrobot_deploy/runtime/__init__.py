"""Everything that runs and controls the robot.

    launcher/         P0: start every process, CHECK, home, hand over, exit
    processes/        P6 (target), P7 (the bus), P11 (the phone), one module each
    shared_memory/    the arena every process reads and writes
    ik/               TCP poses -> joint targets (P6), behind one interface
    execution/        joint targets -> 250 Hz servo goals: the segment walk, the filter, clamps
    drivers/          the servo bus (real and simulated)
    teleop/           the phone: packet format and the arithmetic of driving from it
    timing.py         the monotonic clock and pacing

Deliberately empty of imports: P0 spawns each process by module path, and the bus process
must not pull in the IK stack.
"""
