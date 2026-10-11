# `deploy.toml`, field by field

A deploy file describes one deployment: which robot, which adapter and
checkpoint, where the arm starts and ends, the default settings of each
rollout, and where the recordings go. It sits beside the adapter; every
relative path in it is relative to the file itself, not to where a command is
run.

    g1d_deploy check-policy my_task/deploy.toml     # offline, no robot
    g1d_deploy run my_task/deploy.toml               # on the robot

- **Read first:** `docs/deploying_a_policy/policy_contract.md` (the adapter this file names).
- **Code:** `src/g1d_deploy/config_loading/deploy_file.py` (parsing, layers,
  `--set`), `resolved.py::DeployConfig` (every field and its default),
  `validate.py` (the refusals).
- **Related:** `examples/deploy.toml` (a complete commented file),
  `docs/setup/robot_toml.md`, `docs/setup/machine_toml.md`.

## Contents

1. [How a value is decided](#how-a-value-is-decided)
2. [Per launch or per rollout](#per-launch-or-per-rollout)
3. [Top level: `robot`, `prompt`](#top-level)
4. [`[policy]`](#policy-which-class-in-which-python)
5. [`[checkpoint]`](#checkpoint-which-weights)
6. [`[home]` and `[exit]`](#home-and-exit-where-the-arm-starts-and-ends)
7. [`[control]`](#control-the-rollout-settings)
8. [`[control.servo]`](#controlservo-the-500-hz-servo)
9. [`[intervene]`](#intervene-the-pico-takeover)
10. [`[record]`](#record-where-recordings-go)
11. [`[labels]`](#labels)

## How a value is decided

Four layers, later winning:

    infra defaults  <  the adapter class  <  deploy.toml  <  command line

- **The class** contributes `trained_control_hz` and `trained_action_horizon`.
- **The command line** is named flags (`--prompt`, `--step`, `--ik`, ...) and
  `--set section.key=value` for any key here (the value parsed as TOML).
- **Which layer won** is printed at startup for every non-default value, and
  every rollout's `manifest.json` records the whole resolved result.
- **Unknown keys are refused by name**, so `replan_peroid = 10` stops the
  launch instead of silently running the default.

## Per launch or per rollout

**Per launch** values size shared memory, pick the adapter and checkpoint, or
place the robot; changing one means restarting `g1d_deploy run`. **Per
rollout** values are only the first rollout's default here: the live dashboard
can change them between rollouts without reloading the model
(`docs/deploying_a_policy/running_and_live_dashboard.md`):

- `prompt`;
- `[control]` `sync`, `replan_period`, `control_hz`, `ik` (among the four
  shipped solvers), `blend` (between the two shipped ones), `sample_steps`;
- the numbers in `[control.servo]`.

Each field below says which it is.

---

## Top level

### `robot`

`robot = "g1d_fisheye_dex1"` — a shipped profile name, or a path to a robot
file of your own (`docs/setup/robot_toml.md`). The robot file says what the machine
is: cameras (names, ports, resolution), end effector and TCP, joints, gains,
limits, the servo model, the image server. Camera names an adapter uses must
exist in it.

Shipped: `g1d_fisheye_dex1` (two fisheye wrists + binocular head),
`g1d_fisheye_dex1_exo` (+ a RealSense exo camera), `g1d_normal_dex1` (head
only, vendor image server), `g1d_normal_brainco` (BrainCo hands; no policy
contract yet).

Per launch. Default `g1d_fisheye_dex1`. Flag `--robot`.

### `prompt`

`prompt = "put the cup on the plate"` — the first rollout's instruction:
passed to the constructor and `obs.prompt`. The adapter judges it
(`check_prompt_length`); too long is refused, never truncated.

Per rollout. Flag `--prompt`.

---

## `[policy]`: which class, in which Python

| key | meaning | default |
|---|---|---|
| `entry` | the adapter class: `"policy.py:MyPolicy"` (path relative to this file) or `"my_pkg.adapter:MyPolicy"` | empty = the built-in `EchoPolicy`, which holds the arm still |
| `python` | the interpreter P5 runs under: usually the model's own venv, with g1d_deploy's core installed (`docs/deploying_a_policy/policy_contract.md`, "The adapter's own Python") | g1d_deploy's own |
| `anchor_camera` | overrides the class's declaration: whose frame time is `t_anchor` | the class's |
| `uses_cameras` | overrides the class's declaration: exactly the cameras delivered in `obs.frames` (all are still recorded); must include the anchor | the class's |

`entry` is only *described* at resolution (declarations read in its own
interpreter, GPU hidden); P5 constructs it later.

### `[policy.env]`

```toml
[policy.env]
XLA_PYTHON_CLIENT_PREALLOCATE = "false"   # JAX: allocate on demand (a shared GPU)
# XLA_PYTHON_CLIENT_MEM_FRACTION = "0.5"  # ... or preallocate, but only half
```

Environment P5 sets before it imports anything, merged over the machine
file's `[env]` (these win). Allocator flags such as XLA preallocation only take
effect when set before the first `import jax`, which is why they live here and
not in the adapter.

All per launch.

---

## `[checkpoint]`: which weights

Give **either** `path` **or** `run` + `step`.

| key | meaning |
|---|---|
| `path` | the step directory itself; `""` = this adapter loads nothing |
| `run` | a training run directory |
| `step` | a number; `"latest"` (the highest numbered subdirectory); or an alias the class's `resolve_checkpoint(run, step)` understands (`"best"`) |

The resolved directory is the constructor's `checkpoint` (a `Path`, or
`None`). A checkpoint that does not exist is refused before anything starts.

Per launch. Flags `--checkpoint PATH` (replaces the table), `--step N`.

---

## `[home]` and `[exit]`: where the arm starts and ends

```toml
[home]
arm = [ ...14 radians: left arm then right, shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw ]   # optional
waist = [0.0, 0.0, 0.0]     # yaw, bow, (no motor)
hand = "open"               # "open", "closed", a number (both), or [left, right]
tol_arm = 0.15
tol_waist = 0.05
```

At the start of the run and on every Reset the arm ramps to `home`; `[exit]`
is where it goes when the session ends (keys left out take the home value).

- **Units:** radians in the robot file's joint order
  (`docs/deploying_a_policy/policy_contract.md`, "Joint order"); `hand` in the end effector's
  units (Dex-1 gear radians, 0 = closed).
- **`arm` is optional.** Left out, the arm goes to the infra's default home
  (`src/g1d_deploy/config_loading/deploy_file.py::DEFAULT_HOME_ARM`): forearms
  forward and a little down in front of the torso, a posture suited to
  tabletop work. Set it when a task needs a different start (the first
  observation is taken there). `waist` defaults to zeros, `hand` to open.
- **Tolerances:** after homing, a measured posture further than `tol_arm` /
  `tol_waist` from home refuses the run. The arm's allows for sag under
  gravity at the commanded stiffness; the waist's is tighter because every
  pose is expressed in the frame the waist carries.
- Rarely changed: `min_s` (shortest homing ramp, 3.0 s), `velocity_margin`
  (ramp speed as a fraction of the joint limit, 0.5).

Per launch. Flags `--home-tol-arm`, `--home-tol-waist`.

---

## `[control]`: the rollout settings

### `control_hz`

The action rate: consecutive actions of a chunk are executed `1 / control_hz`
apart, and action `j` is due `(j + 1) / control_hz` after `t_anchor`. The
training-consistent value is the spacing of actions in the training data,
which the class declares as `trained_control_hz`. It **may differ**: faster
plays the same trajectory faster, slower plays it slower; the infra warns and
runs it, so speed is an A/B knob.

Per rollout. Default: `trained_control_hz`, else 30. Flag `--control-hz`.

### `action_horizon` (H)

How many actions of each chunk the infra uses: a longer chunk is cut to its
first H. May not exceed `trained_action_horizon` — a chunk cannot contain
rows the model never produces. Setting it lower is how a model's chunk is used
only partly (with `P ≤ H`).

Per launch (it sizes the chunk ring). Default: `trained_action_horizon`, else
50. Flag `--action-horizon`.

### `replan_period` (P)

How many ticks run between inferences. `2 ≤ P ≤ H`. Smaller reacts sooner and
costs more GPU; in sync the rest of each chunk is discarded.

Per rollout. Default 25. Flag `--replan-period`.

### `sync`

- **`true` (sync):** the robot executes P actions of a chunk, then waits for
  the next inference, which is triggered at the end of the P ticks. Latency
  becomes a pause at every seam, never a missed action. Every IK solver works.
- **`false` (async):** the arm keeps moving while the next chunk is inferred.
  Inference still runs once per P ticks, triggered P ticks after the last;
  each new chunk takes over from the tick being executed when it lands, plus
  two, and where chunks overlap they are merged by `blend`
  (`docs/infra_logic/chunk_blending.md`). It needs the inference latency L (measured at
  warmup, trigger to chunk written) to fit: **`P / control_hz > L`** and
  **`H − P ≥ ceil(L · control_hz) + 2`**. A Start that breaks either is refused
  with the range of P that would work, because otherwise every chunk lands
  after its ticks have passed and the arm silently holds. Arm-mode IK only.

Per rollout. Default `true`.

### `sample_steps`

A number the infra never interprets, handed to `on_settings` as
`settings.sample_steps`: meant for the sampler's step count (flow or diffusion
steps). It lives here so the operator can change it per rollout and every
record says which value ran.

Per rollout. Default 10. Flag `--sample-steps`.

### `ik` (eef adapters only)

Which solver turns flange poses into joint angles (`docs/infra_logic/ik_solvers.md`).
Unused for joint adapters.

| value | mode | |
|---|---|---|
| `arm_planner` | arm | default; plans each chunk's elbow branch; waist upright |
| `arm_qp` | arm | one QP per tick from the last solution; the only one `--intervene` runs |
| `waist_planner` | waist | arm planner on a planned trunk (yaw and bow); sync only |
| `waist_qp` | waist | whole-body QP, waist tier; sync only |
| `"pkg.module:factory"` or `"path/file.py:factory"` | the factory's | your own solver (a file path is relative to this file) |

`"arm"` / `"waist"` mean each mode's default.

Per rollout among the four shipped names; a `module:factory` is per launch.
Flag `--ik`.

### `blend` (async only)

How overlapping chunks become one action per tick (`docs/infra_logic/chunk_blending.md`):

- `"temporal_ensemble"` (**default**): ACT's temporal ensembling, an
  age-weighted average of every chunk's prediction of the tick; smooth seams;
- `"newest"`: the newest chunk takes over outright; for real-time chunking or
  the fastest reaction;
- your own, as `"pkg.module:function"` or `"path/file.py:function"` (relative
  to this file), imported at launch in g1d_deploy's Python (P6 runs it).

`temporal_ensemble_m` (default 0.01, ACT's): the ensemble's `m`, per tick of a
prediction's age; 0 is a plain mean, larger leans on older predictions.

Per rollout between the two shipped names (the dashboard's Deploy settings,
shown when async is selected); a function of your own is per launch: a
rollout can switch away from it, not to it. Inert in sync, where each tick has
one chunk.

### Rarely changed

| key | default | what it does |
|---|---|---|
| `state_history_s` | 1.0 | seconds of the ~1 kHz state stream kept in shared memory; the ceiling for a class's `history["state"]`. Per launch |
| `solve_lead_ticks` | 5 | how many ticks ahead of execution P6 may solve; keeps the measured trunk each solve uses fresh |
| `plan_lookahead_ticks` | 0 | sync only: extra decoded rows past P the trunk planner may look at; `P + this ≤ H` |
| `ik_posture_cost`, `ik_collision_min_dist`, `ik_early_stop`, `ik_max_iters` | 0.02, 0.005 m, 1e-4 m, 20 | QP tuning shared by the solvers |

---

## `[control.servo]`: the 500 Hz servo

P7 turns the per-tick joint targets into 500 Hz motor commands
(`docs/infra_logic/500hz_execution.md`). The numbers are per rollout and editable on the
dashboard; the defaults are the tuned ones.

| key | default | what it does |
|---|---|---|
| `filter_hz` | 4.0 | cutoff of the critically damped smoother on the target line, Hz; `0` = straight lines between targets (and turns a custom `smoother` off) |
| `filter_lead` | true | read the target line ahead by the smoother's own lag, so smoothing adds no delay |
| `velocity_ff` | 1.0 | scale on the published velocity feedforward |
| `torque_ff` | 0.5 | scale on the model torque feedforward (inertia and friction); needs the robot file's `[servo_model]` unless `feedforward` is set; `0` = off (and turns a custom `feedforward` off) |
| `arm_gain_scale` | 1.0 | scale on the arm joints' kp; kd by its square root |
| `smoother` | — | `"pkg.module:factory"` or `"path/file.py:factory"`: replaces the critically damped smoother. Per launch |
| `feedforward` | — | the same spellings: replaces the model torque feedforward. Per launch |

`--intervene` runs the plain straight-line servo (`filter_hz = 0`,
`torque_ff = 0`), so custom modules are off there too.

---

## `[intervene]`: the PICO takeover

Used only with `g1d_deploy run --intervene`, where an operator with a PICO
headset can take over the arms mid-rollout and hand back. Defaults are this
rig's measured values.

| key | default | what it does |
|---|---|---|
| `lead_s` | 0.035 | how far ahead of the operator's instant a target is written: jitter margin and perceived lag |
| `stale_budget_s` | 0.5 | how long the headset may go silent before control returns to the policy |
| `scale` | 1.0 | metres of arm per metre of hand, translation only |
| `hand_tcp` | measured | controller-to-operator TCP per hand, `[[x,y,z,qw,qx,qy,qz], [...]]` |
| `gripper_rate` | 3.0 | how fast a held thumbstick drives the jaw, rad/s |
| `squeeze_margin_rad` | 0.6 | how far below the measured jaw a command may sit (bounds squeeze force) |
| `up_closes` | true | whether pushing the thumbstick up closes the jaw |

---

## `[record]`: where recordings go

Every real rollout is recorded: one directory per rollout holding every camera
frame as the robot sent it, the state stream, the chunks, P6's targets, P7's
commands and a `manifest.json` saying exactly what ran. About 1 GB per rollout
with three cameras. `g1d_deploy inspect` opens them
(`docs/deploying_a_policy/rollout_inspection.md`).

| key | default | what it does |
|---|---|---|
| `enabled` | `true` | `false` (or `--no-record`) records nothing |
| `out` | — | the record root, relative to this file; wins over everything below. Flag `--out` |
| `task` | this file's folder name | the folder under the machine's records root |

Where, first found wins:

1. `out` / `--out`;
2. the machine file's `[paths] records` / `<task>` (`docs/setup/machine_toml.md`);
3. `rollout_record/` beside this file.

Per launch.

## `[labels]`

```toml
[labels]
ab_arm = "baseline"
```

String keys to string values, copied into every record and read by nothing:
tags for an experiment.
