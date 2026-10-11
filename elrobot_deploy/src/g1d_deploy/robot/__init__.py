"""What the robot IS: its profiles, its model files, and the maths on them.

    profiles/*.toml     the shipped robot files (cameras, end effector, TCP,
                        servo model, image server); `config_loading.robot_file`
                        reads them, and yours may `extends` one
    assets/             the G1-D URDF and meshes, the Dex-1 meshes
    end_effectors.py    the built-in end-effector tables (Dex-1, BrainCo)
    kinematics.py       forward kinematics and trunk placement (numpy only)
    mujoco_model.py     the generic MuJoCo backend the IK and the viewers pose
    g1d_model.py        the G1-D on that backend: base, column, waist, arms
    asset_paths.py      where `assets/` is

Deliberately empty of imports: P5 imports `kinematics` and `end_effectors`
under the policy's own interpreter, which has no MuJoCo.

Robot files for a user: `docs/setup/robot_toml.md`.
"""
