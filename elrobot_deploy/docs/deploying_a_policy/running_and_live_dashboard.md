# Running a policy, and the live dashboard

How a launch goes from `elrobot_deploy run` to rollouts, what the operator sees,
and what can change between rollouts.

- **Read first:** `docs/overview.md`.
- **Code:** `src/elrobot_deploy/cli/main.py` (the commands and flags),
  `src/elrobot_deploy/runtime/launcher/` (P0: `state_machine.py`,
  `startup_checks.py`, `ramps.py`), `src/elrobot_deploy/dashboards/live/` (P10).
- **Related:** `docs/deploying_a_policy/deploy_toml.md`, `docs/setup/machine_toml.md`,
  `docs/setup/robot_toml.md`, `docs/deploying_a_policy/rollout_inspection.md`.

## The commands

| command | robot? | what |
|---|---|---|
| `elrobot_deploy check [--robot R] [--no-robot]` | read-only | is this box ready (packages, machine file, GPU, `/dev/shm`, network card, CPU table), and does the robot answer (every camera the robot file lists, with a preview of each fisheye's lens fit; the state streams)? Never publishes anything |
| `elrobot_deploy check-policy deploy.toml [--no-run]` | no | level 1: resolve the deploy file exactly as `run` would, applying every startup check, then construct the class and judge the prompt in its own Python with the GPU hidden (seconds, no GPU). Level 2: under the adapter's Python, construct it, judge the prompt, `warmup()` and one `infer` on a dummy observation — the policy loads and an inference completes. Level 2 **takes the GPU** like a warmup and is refused while a deployment is live; it fails on exceptions, a wrong action type or an unusable action count, and only *notes* action values (meaningless on black frames). `--no-run` stops after level 1 |
| `elrobot_deploy run deploy.toml` | **moves the arm** | the deployment |
| `elrobot_deploy inspect [DIR]` | no | the rollout inspection dashboard (`docs/deploying_a_policy/rollout_inspection.md`) |

Activate the venv first (`source .venv/bin/activate` in the checkout), or
prefix commands with `uv run`.

## A launch, start to finish

`elrobot_deploy run my_task/deploy.toml`:

1. **Resolve** the deploy file (layers, the adapter's declarations read in its
   own Python), print a banner of every non-default value and where it came
   from, and refuse anything inconsistent — all before anything starts.
2. **INIT.** For a fisheye rig: check the robot is on the subnet, start the
   board's image server if the robot file says so, and fit every fisheye's
   lens circle from ~1 s of live frames (the square around it is what that
   camera's ring holds). Then create the arena in `/dev/shm` (after checking
   it fits).
3. **CHECK.** Start the state and camera processes; wait for the state
   stream, the hands and every camera; verify rates.
4. **HOME.** Ramp the arm to `[home]` — **the arm moves here, before Start is
   pressed.** P5 starts loading and warming the policy in parallel. After
   homing, a posture further than `tol_arm` / `tol_waist` from home refuses the
   run.
5. **WARMUP → READY.** When P5 reports warm, the terminal and dashboard say
   READY.
6. **Start** (dashboard) → **ACTIVE**: a rollout. P5 infers, P6 solves, P7
   servos, P9 records.
7. **Reset** ends the rollout and homes the arm again without reloading the
   policy; **Start** begins the next rollout.
8. **Exit** (dashboard) or the first **Ctrl-C** runs the exit ramp to
   `[exit]`, then damps. A **second Ctrl-C** skips straight to the damp,
   which drops the arm into gravity — keep a hand ready.

Recording: each rollout is one directory under the record root
(`docs/deploying_a_policy/deploy_toml.md`, `[record]`), with a `manifest.json` holding the
resolved config, the robot file chain verbatim, the adapter entry and
checkpoint, both repos' git revisions, the crop decisions, and each rollout's
settings and prompt.

## The live dashboard

Served by P10 on the machine file's `[dashboard] port` (8080), on loopback.
From a laptop: `ssh -L 8080:localhost:8080 <deploy box>`, then open
`localhost:8080`. One page:

- **The bar** — Start / Reset / Exit; the prompt field (Enter submits; the
  adapter accepts or refuses it within about one inference); the **Deploy
  settings** button; the phase, pending request and `k_exec` chips; the
  experiment, step and checkpoint.
- **The status line** — phase, IK solver and mode, inference mode
  (sync/async), replan period, rate and prompt, always of the rollout that is
  running (or last ran), never of a pending request.
- **The cameras** — one pane per robot-file camera as the robot sent it; the
  panes the adapter uses come first. A fisheye pane outlines the lens circle
  and the square the policy sees: check the square hugs the disc, and that the
  lens looks like the one the data was collected with.
- **The 3-D scene** — switches for the measured robot, the commanded flange
  pose, the commanded joints, the virtual TCP, and the base and column. The
  commanded overlays exist only during a rollout; the pose overlay is greyed
  for joint-mode adapters.

IK residuals, servo tracking and timing are not shown live; look at them over
the recording (`docs/deploying_a_policy/rollout_inspection.md`).

## Changing things between rollouts

Nothing changes mid-rollout: both controls are disabled while one runs, and
what you submit takes effect at the next accepted Start.

- **Prompt:** type and press Enter; the adapter judges it (only it knows its
  tokenizer). A Start whose newest prompt was refused is refused.
- **Deploy settings:** sync/async, replan period (2 … H), control rate, IK
  solver (the four shipped ones, plus the launch's own `module:factory` if it
  named one; waist solvers in sync only), the blend (shown when async is selected), sample
  steps, and the servo numbers. The panel closes when a request is accepted. **Request** checks them at once and stores them as *requested*;
  the panel shows requested beside *active*.

At Start, P0 validates the request again (and, for async, checks the
measured inference latency fits: `docs/deploying_a_policy/deploy_toml.md`, `sync`), P5 hands the
adapter any new prompt and settings (`set_prompt`, `on_settings`) and runs
one dummy inference so a recompile is paid now, and only then does the
rollout begin. A refusal leaves the robot at READY with the reason on the
dashboard and the terminal.

**Fixed per launch** (relaunch to change): robot file, adapter, checkpoint,
`action_horizon`, history lengths, the camera set, a `module:factory` IK
solver, a `module:function` blend, and custom servo modules.

## Run flags

`--robot`, `--machine`, `--prompt`, `--checkpoint`, `--step`, `--out`,
`--no-record`, `--set KEY=VALUE`, `--ik`, `--sample-steps`,
`--replan-period`, `--action-horizon`, `--control-hz`, `--home-tol-arm`,
`--home-tol-waist`, the servo flags (`--servo-filter-hz`, `--servo-no-lead`,
`--velocity-ff`, `--torque-ff`, `--arm-gain-scale`), and
`--intervene`. `elrobot_deploy run --help` describes each.

### `--intervene`: PICO takeover

An operator with a PICO headset can take over the arms mid-rollout and hand
back (a DAgger-style correction; the record marks every takeover). It needs a
one-time build (`scripts/install.sh --intervene`) and runs with the `arm_qp`
IK and the plain linear servo. The controller drives the **flange**, not the
TCP: turning the wrist pivots at the flange and swings the Dex-1's fingertips
through ~26 cm per quarter turn. `[intervene]` in the deploy file tunes it.

## Known limits

- **The DDS layer** is elrobot_deploy's own over CycloneDDS (vendored message
  types and CRC, tested byte-identical to Unitree's SDK) and has not yet been
  checked on the robot. Do one supervised run first; if the state arrives but
  the arm does not move at homing, suspect the command, not the network.
- **No policy runs on a BrainCo hand** until its contract is decided.
- **The IK drives the TCP** when the robot file gives one (every Dex-1
  profile); planner benchmarks from before that should be re-run.
