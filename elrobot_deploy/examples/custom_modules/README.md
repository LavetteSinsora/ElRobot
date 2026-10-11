# Examples: replacing parts of the infra

Three parts of elrobot_deploy can be replaced by a function or class of your own,
named in the deploy file. Nothing in the framework is edited.

| file | replaces | deploy-file key | explained in |
|---|---|---|---|
| `ik_solver.py` | the IK solver: target poses to joint angles | `[control] ik` | `docs/infra_logic/ik_solvers.md` |
| `blend.py` | how overlapping chunks merge in async mode | `[control] blend` | `docs/infra_logic/chunk_blending.md` |
| `servo.py` | the 500 Hz smoother and the torque feedforward | `[control.servo] smoother`, `feedforward` | `docs/infra_logic/500hz_execution.md` |

    elrobot_deploy check-policy examples/custom_modules/deploy.toml     # offline

`deploy.toml` uses all three at once on the sine policy
(`examples/policies/sine_eef.py`); each line works on its own in any deploy
file. A module is named `"file.py:name"` (relative to the deploy file) or
`"package.module:name"`, and is imported at launch, so a typo stops the
launch before anything moves.

These run inside elrobot_deploy's own processes and its own Python, not the
policy's: the IK and the blend in the target process (once per control tick),
the smoother and feedforward in the 500 Hz servo. Keep them fast.

Each example is deliberately plain: `ik_solver.py` wraps the shipped solver
and times it, `blend.py` averages only the chunks that agree with the newest
one, `servo.py` is a textbook second-order filter. They show the shape of
each slot, not a better algorithm.

A fourth extension point, a comparison mode for the inspection dashboard, is
`examples/inspect_mode/`.
