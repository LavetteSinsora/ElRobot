# elrobot_deploy

The deployment runtime for our ElRobot follower: one 7-DoF arm and a gripper, 8 Feetech ST3215
servos on one serial bus, driven from an iPhone (policies later). Ported from g1d_deploy
(`UPSTREAM.md`); the plan and its decisions are `PORTING_PLAN.md`.

```bash
uv sync --extra runtime                  # once (includes ../elrobot, the bus and calibration)
uv run pytest -q                         # no hardware needed

uv run elrobot_deploy phone-bench        # check how you hold the phone; no robot
uv run elrobot_deploy run --sim          # the whole chain on simulated servos
uv run elrobot_deploy run                # the real arm
```

How it works (processes, the arena, the session, the teleop mapping, timing, safety, what is
pluggable): **`docs/architecture.md`**. The robot profile, with every setting commented:
`src/elrobot_deploy/robot/profiles/elrobot.toml`. The phone app: `../phone_teleop/README.md`.

`legacy_g1d/` holds the g1d code not ported yet (policies, recording, dashboards, cameras),
unchanged, as reference; it is not packaged, linted or tested.
