"""g1d_deploy — a deployment runtime for the Unitree G1-D.

    cli/              the `g1d_deploy` command: run, check, inspect
    policy_contract/  what a policy author codes against (Policy, actions,
                      Observation, Reader, loading a class, checkpoints)
    config_loading/   reads deploy.toml, robot files and machine.toml and
                      resolves them into the settings every process reads
    robot/            what the robot is: profiles/*.toml, assets/ (URDF,
                      meshes), end effectors, kinematics, the MuJoCo model
    runtime/          everything that runs and controls the robot: launcher/
                      (P0), processes/ (P1-P11), shared_memory/, ik/,
                      execution/, drivers/, teleop/, timing, cpu_pinning
    rollout_record_utils/
                      code for the record format, not records: builds the
                      manifest P9 writes and reads a record back
    dashboards/       the live dashboard (P10), rollout inspection, and the
                      web code they share

Processes P0-P11 share one shared-memory arena (`runtime/shared_memory/`),
where every field has exactly one writing process. A policy's own interpreter
imports only `policy_contract` and the numpy-only core it needs.

The map of the whole system, and of the documentation, is `docs/overview.md`.
"""

#: The policy surface, re-exported so a policy can write `g1d_deploy.Policy`.
#: Lazy, because P7 imports this package and must not pay for — or be able to
#: reach — anything beyond what it names: the 500 Hz loop imports no policy
#: code and no IK (`tests/test_p7_body.py` asserts its import set).
_POLICY_NAMES = ("Policy", "EefActions", "JointActions", "Observation",
                 "Reader", "RobotSpec")


def __getattr__(name):
    if name in _POLICY_NAMES:
        from g1d_deploy import policy_contract as policy
        return getattr(policy, name)
    raise AttributeError(f"module 'g1d_deploy' has no attribute {name!r}")


__all__ = [*_POLICY_NAMES]
