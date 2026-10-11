"""The two web apps and what they share.

    live/      P10's dashboard while the robot runs: Start/Reset/Exit, the
               prompt, deploy settings, the cameras and the posed robot
    inspect/   `elrobot_deploy inspect`: recorded rollouts offline, with modes
               (IK solvers, policy checkpoint) and the producers they spawn
    web/       the shared HTTP server, packing, jobs, GPU lock and the WebGL
               renderer both apps draw with

For the operator: `docs/deploying_a_policy/running_and_live_dashboard.md`
("The live dashboard") and `docs/deploying_a_policy/rollout_inspection.md`.
"""
