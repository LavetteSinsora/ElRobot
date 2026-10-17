"""What the robot IS.

    profiles/*.toml     the robot profiles (`config_loading/config.py` reads them)
    elrobot_joints.py   the 8 joints, their order and limits, read from our URDF (no MuJoCo)
    elrobot_model.py    the MuJoCo model of our URDF (the IK and FK)
    transforms.py       quaternion and pose algebra in numpy

The URDF itself is the repository's `description/elrobot_ours/elrobot.urdf`.
"""
