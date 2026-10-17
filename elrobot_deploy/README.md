# elrobot_deploy

The deployment runtime for our ElRobot follower: one 7-DoF arm and a gripper, 8 Feetech ST3215
servos on one serial bus, driven from an iPhone (policies later). Ported from g1d_deploy
(`UPSTREAM.md`); the plan and its decisions are `PORTING_PLAN.md`.

```bash
uv sync --extra runtime                  # once (includes ../elrobot, the bus and calibration)
uv run pytest -q                         # no hardware needed

uv run elrobot_deploy view               # the URDF and the home pose in the browser; no robot
uv run elrobot_deploy phone-bench        # check how you hold the phone; no robot
uv run elrobot_deploy run --sim          # the whole chain on simulated servos
uv run elrobot_deploy run --dry-run      # the real arm, torque OFF: only the ghost moves
uv run elrobot_deploy run                # the real arm
```

A session: the hardware check, then the arm held where it is while the dashboard
(http://127.0.0.1:8080) shows the home pose; **Home**, then **Start** (dashboard buttons or Enter
in the terminal); then tap ENGAGE on the phone to drive (tap again to hold). Ctrl-C (or Exit) ramps to rest.

How it works (processes, the arena, the session, the teleop mapping, timing, safety, what is
pluggable): **`docs/architecture.md`**. The robot profile, with every setting commented:
`src/elrobot_deploy/robot/profiles/elrobot.toml`. The phone app: `../phone_teleop/README.md`.

`legacy_g1d/` holds the g1d code not ported yet (policies, recording, rollout inspection, cameras),
unchanged, as reference; it is not packaged, linted or tested.
