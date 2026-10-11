"""The drivers: DDS, ZMQ and the board, and nothing above them.

    arm_command.py    `rt/lowcmd` and gravity compensation       (P7, P0's damp)
    robot_state.py    `rt/lowstate` and the hand state topics    (P1)
    gripper.py        the hand command topics                    (P8)
    column.py         the lift column's height                   (P1)
    cameras.py        ZMQ, V4L2 and RealSense camera sources     (P2…)
    fisheye_circle_fit.py   a fisheye lens circle and its square (P0, P2)
    dds/              the DDS domain, channels, CRC and IDL types
    board/            what is started on the robot's board over SSH

Every module here is a thin binding to a device or a topic. They hold no
scheduling logic, no rings and no notion of a tick — a process loop decides
*when*, a driver decides only *how to say it*.

That split is what lets the whole system run on a laptop: each binding is
constructed behind an interface the loops depend on, and the tests substitute a
fake that records what was published.
"""
