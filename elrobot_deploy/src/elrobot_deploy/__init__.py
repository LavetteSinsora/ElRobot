"""elrobot_deploy: the deployment runtime for the ElRobot follower arm (one 7-DoF arm and a
gripper on one ST3215 servo bus), driven from an iPhone now and by policies later.

    cli/              the `elrobot_deploy` command: run, config, executor-test
    config_loading/   the robot profile -> one frozen `Config` every process is handed
    robot/            what the robot is: profiles/*.toml, the joints and limits (from our URDF),
                      the MuJoCo model, numpy pose algebra
    runtime/          everything that runs: launcher/ (P0), processes/ (P6, P7 bus, P11 phone),
                      shared_memory/ (the arena), ik/, execution/ (the executor), drivers/
                      (the servo bus), teleop/ (the phone maths), timing

The map of the whole system: `docs/architecture.md`. The g1d code not ported yet (policies,
recording, dashboards, cameras) is in `../legacy_g1d/`.
"""
