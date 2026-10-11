---
name: deploy-a-policy
description: Guide for getting a trained policy to run on a real Unitree G1-D through the g1d_deploy infra. Use it whenever the user wants to deploy, run, roll out or test a policy or checkpoint on the G1-D; needs the policy adapter or the deploy.toml written or changed; or has an adapter that fails `g1d_deploy check-policy` or behaves wrongly on the robot. Covers how the infra works, what has to be set up first, how to write and verify the adapter, and how to fill in the deploy file.
---

# Deploy a policy on the G1-D with g1d_deploy

This skill is for helping a user run a policy they trained on a real Unitree
G1-D. It explains how the deployment infra (g1d_deploy) works, what has to be
in place before a policy can run, and how to write and check the two files a
policy needs:

1. **the policy adapter**: a Python class that wraps the user's model;
2. **the deploy file** (`deploy.toml`): the settings for one deployment.

The skill is written for the common case, a trained model with a checkpoint.
A policy that is not a neural network (a scripted controller, a bridge to an
external planner or language model that emits waypoints) is deployed the same
way: its adapter returns actions in the same contract, the deploy file's
`[checkpoint]` is left empty, and the sections about training below reduce to
"what does this policy expect as input, and what do its outputs mean".

The work is finished when the user can run `g1d_deploy run path/to/deploy.toml`
on the robot. That command moves the robot, so the user runs it, supervised.
Do not run it yourself. Everything in this skill is done without the robot
moving.

Contents: [how g1d_deploy works](#how-g1d_deploy-works) ·
[where the files are](#where-the-files-are) · [setup](#setup) ·
[the policy adapter](#part-1-the-policy-adapter) ·
[the deploy file](#part-2-the-deploy-file) ·
[verifying](#part-3-verify-without-the-robot) ·
[handing over](#part-4-hand-over) · [when something fails](#when-something-fails)

## How g1d_deploy works

g1d_deploy is a set of processes on the deploy computer that share one block
of memory, called the **arena**. Every stream of data has its own region in
the arena (a ring buffer), and each region has exactly one writer.

```
 robot cameras ──► camera rings ─┐
 robot joints  ──► state ring ───┤
                                 ▼
                    policy process: calls the ADAPTER's infer(obs)
                                 │
                                 ▼
                         action chunk ring
                                 │
                                 ▼
          target process: merges chunks, solves IK ──► joint targets
                                 │
                                 ▼
          500 Hz servo ──► motor commands ──► robot
```

- **The infra writes the observations.** Camera processes decode every frame
  into a camera ring. A state process writes joint angles, joint velocities,
  the pose of both wrists and the gripper openings into the state ring, about
  1000 times a second. Everything is timestamped on arrival.
- **The policy reads observations and writes action chunks.** The policy
  process decides when an inference is due, picks the camera frames and the
  state sample for that instant, and passes them to the adapter as one
  observation. The adapter returns a chunk of future actions. The policy
  process checks the chunk and writes it to the **action chunk ring**.
- **The infra turns chunks into motion.** Once per control tick (typically
  30 Hz) the target process takes the action due at that tick, runs inverse
  kinematics if the actions are end-effector poses, and writes a joint target.
  A 500 Hz servo follows those targets smoothly and sends motor commands.
- **Everything is recorded.** Each rollout is saved to disk and can be
  replayed in an inspection dashboard afterwards.

So the adapter has one job: given an observation, return actions in the format
the infra expects. Timing, synchronisation, safety checks, IK, the servo and
recording are the infra's and need no code from the user.

A launch (`g1d_deploy run`) goes through these stages: check the robot and
cameras, move the arms to a home posture, load and warm up the policy, then
wait at READY. The operator presses Start on a web dashboard to begin a
rollout, Reset to end it and return home, and Start again for the next one.
The model is loaded once per launch.

For more depth: `docs/overview.md` (the processes and one chunk's life) and
`docs/deploying_a_policy/running_and_live_dashboard.md` (a launch, step by step).

## Where the files are

This skill refers to documents, examples and templates by paths such as
`docs/deploying_a_policy/policy_contract.md` and `examples/policies/sine_eef.py`. Those paths are relative to
the **root of the g1d_deploy checkout**: the directory that contains
`README.md`, `docs/`, `examples/`, `skills/` and `src/`. Find that directory
once, at the start, so the paths can be opened.

- This skill normally lives at `skills/deploy-a-policy/` inside the checkout,
  or is a symlink to it. Then the root is two directories above the real
  location of this file.
- If the skill was copied somewhere else, ask the installed package:

  ```bash
  python -c "import g1d_deploy, pathlib; print(pathlib.Path(g1d_deploy.__file__).resolve().parents[2])"
  ```

  Run it with a Python that has g1d_deploy installed (the checkout's
  `.venv/bin/python`). If the answer is a `site-packages` directory, the
  package was not installed from a checkout; ask the user where the checkout
  is.

What to read there, and when:

| file | read it |
|---|---|
| `docs/deploying_a_policy/policy_contract.md` | before writing an adapter, in full. It is the reference for everything in Part 1 |
| `docs/deploying_a_policy/deploy_toml.md` | when filling in the deploy file: every key and its default |
| `docs/setup/robot_toml.md` | to see which cameras and gripper a robot profile has |
| `docs/deploying_a_policy/running_and_live_dashboard.md` | to explain to the user what a launch does and how to operate it |
| `docs/deploying_a_policy/rollout_inspection.md` | to replay an adapter on a recorded rollout |
| `docs/infra_logic/arena_layout_and_reads.md` | only if the adapter must read the arena directly (rare) |
| `docs/deploying_a_policy/field_notes.md` | when a launch on the robot fails in a way the message does not explain |
| `examples/policies/sine_eef.py`, `examples/policies/sine_joint.py` | small complete adapters with their deploy files |
| `examples/policies/history.py`, `examples/rtc/` | adapters that use past observations, or read the arena |
| `skills/deploy-a-policy/templates/` | the files to start a new adapter from |
| `skills/deploy-a-policy/references/adapter_patterns.md` | solutions to situations that come up with real checkpoints (several runs sharing one adapter, launch-time switches, a missing camera, checkpoint self-checks) |

## Setup

Check these before writing anything. Each is a one-time step per computer; if
any is missing, the `setup-g1d-deploy` skill
(`skills/setup-g1d-deploy/SKILL.md`) walks through it.

1. **g1d_deploy is installed.** The checkout has a `.venv` and
   `.venv/bin/g1d_deploy` exists. If not: `scripts/install.sh` from the
   checkout root (`docs/setup/install.md`). Activate with
   `source .venv/bin/activate`.
2. **A robot profile fits the user's robot.** The shipped profiles
   (`docs/setup/robot_toml.md`) cover the standard camera rigs with the
   factory board address and login. A robot that differs gets its own robot
   file extending one. `machine.toml` at the checkout root is optional: it
   only overrides this computer's defaults, such as where recordings are
   stored (`docs/setup/machine_toml.md`).
3. **`g1d_deploy check` passes** (optional before writing; required before a
   launch). It is read-only: it reports whether this computer is ready and
   whether the robot's cameras and state streams answer. `--no-robot` skips
   the robot half.
4. **The model's Python can import g1d_deploy.** The adapter runs in the
   model's own Python environment (its torch or JAX versions), not in
   g1d_deploy's. That environment needs g1d_deploy's core, which depends only
   on numpy. Either install it there
   (`<model venv>/bin/python -m pip install -e <checkout root>`), or write a
   small wrapper script that sets `PYTHONPATH` and runs the model's Python
   (`docs/deploying_a_policy/policy_contract.md`, "The adapter's own Python").

## Part 1: the policy adapter

Most of the work, and nearly all of the risk, is here. The adapter is one
class, a subclass of `g1d_deploy.policy_contract.Policy`. It has three parts:

- **declarations**: class attributes that tell the infra what the policy
  needs (which cameras, the action type, the action rate);
- **`__init__` and `warmup`**: load the checkpoint;
- **`infer(obs, reader)`**: one inference. Preprocess the observation the way
  training did, run the model, and convert its output to the infra's action
  format.

The smallest complete one is in `docs/deploying_a_policy/policy_contract.md`, "The smallest
adapter". Start a real one from `skills/deploy-a-policy/templates/policy.py`.

### What `infer` receives

| | |
|---|---|
| `obs.frames[camera].pixels` | one image per camera, `uint8 (height, width, 3)`, **RGB**, at the camera's native size. A fisheye camera's image is already cut to the square around its lens circle; do not crop it again |
| `obs.state.q`, `obs.state.dq` | measured joint angles and velocities, `(29,)`. Index with `self.robot.joints_arm` and `self.robot.joints_waist`, not literal numbers |
| `obs.state.flange` | measured pose of both wrists, `(2, 7)`, in the same format as the pose actions below |
| `obs.state.hand` | measured gripper openings, `(2,)`, in the same units as the gripper actions below |
| `obs.prompt` | the task instruction, a string |
| `obs.t_anchor` | the time this observation depicts |
| `reader` | past frames and states, for models that take history; unused otherwise |

Camera names come from the robot profile (`docs/setup/robot_toml.md`). On the
standard fisheye rig they are `head_left`, `head_right` (the two eyes of the
head camera), `lwrist` and `rwrist`.

### What `infer` returns

A chunk of `n` future actions. Action `j` is executed `(j + 1) / control_hz`
seconds after `obs.t_anchor`. Return the whole chunk the model produces; the
infra cuts it to the length in use.

For a model that predicts end-effector poses (`action_mode = "eef"`):

```python
EefActions(pose=..., hand=...)    # pose (n, 2, 7), hand (n, 2)
```

- `pose[:, 0]` is the left arm, `pose[:, 1]` the right.
- Each pose is `[x, y, z, qw, qx, qy, qz]`: metres, and a unit quaternion
  with the scalar **first** (wxyz).
- It is the pose of the **flange** (the last wrist link), not of the gripper
  tip.
- It is **absolute**, in the robot's **base frame**: x forward, y left, z up,
  origin at the waist.
- `hand` is the gripper opening in Dex-1 motor radians: **0 is fully closed,
  about 5.4 is fully open**.

For a model that predicts joint angles (`action_mode = "joint"`):

```python
JointActions(arm_q=..., waist_q=..., hand=...)   # (n, 2, 7), (n, 3), (n, 2)
```

in radians, in the joint order tabled in `docs/deploying_a_policy/policy_contract.md`, "Joint
order and limits". `waist_q` is required; zeros keep the trunk upright.

**The infra refuses a bad chunk; it does not repair it.** A quaternion that is
not unit length, a gripper value outside its range, a joint beyond its limit
or a NaN is refused with a `ChunkError`. Normalise quaternions and clip
(`self.robot.hand_min`, `self.robot.hand_max`) in the adapter.

### Find out how the model was trained

The adapter must reproduce the training pipeline's input processing and
reverse its output processing. Read the training configuration, the dataset
and transform code, the model code and any existing inference script, and
answer each question below. Where the code does not settle a question, ask the
user. A wrong guess here usually passes every automatic check and shows up
only as the robot moving wrongly.

**Inputs**

| find out | what to do in the adapter |
|---|---|
| Which cameras the model takes, and which image slot each one fills | Map the robot's camera names to the model's slots. List them in `uses_cameras`. Set `anchor_camera` to the one the model relies on most for fine motion (usually a wrist camera) |
| How images were preprocessed: resize method, padding, value range, channel order | Apply the same steps to `obs.frames[...]`. Call the training code's own functions where possible: two resize helpers with the same name can differ by a pixel value |
| Which robot state the model takes (joints, wrist poses, gripper), in which frame, units and order | Build it from `obs.state` in that same layout |
| Whether the model takes past frames or states | Declare `history` and read them from `reader` (`examples/policies/history.py`). Otherwise leave `history` out |
| The tokenizer and the prompt template | Tokenize `obs.prompt` as training did. Set `max_prompt_tokens` to the length prompts are padded to, and implement `count_prompt_tokens` (see [the prompt](#the-prompt)) |
| The normalisation statistics: where they are stored, which keys they cover | Load them in `__init__`; apply them to the inputs and invert them on the outputs |

**Outputs**

| find out | what to do in the adapter |
|---|---|
| Whether actions are end-effector poses or joint angles | `action_mode = "eef"` or `"joint"` |
| Whether poses are absolute or relative, and relative to what (the current pose, the first pose of the chunk) | Convert to absolute poses, using `obs.state.flange` as the reference |
| Which frame the poses are in | Convert to the robot base frame |
| Which point the poses describe: the flange, or a tool point such as the gripper tip | If a tool point, convert to the flange with `self.robot.tcp` |
| The rotation format: quaternion order, rotation vector, 6D, Euler angles | Convert to a unit wxyz quaternion |
| The order of the arms and the layout of the action vector | Reorder to `[left, right]` |
| The gripper's units and direction (radians, metres, 0 to 1; which end is open) | Convert to Dex-1 radians with 0 = closed, then clip |
| The number of actions per chunk, and the time between consecutive actions in the training data | `trained_action_horizon` and `trained_control_hz`. The second is the data's action rate, which may differ from the camera frame rate |

`skills/deploy-a-policy/templates/conversions.py` has tested functions for the
rotation, tool-point and relative-to-absolute conversions; copy it beside the
adapter.

**Loading**

| find out | what to do in the adapter |
|---|---|
| The checkpoint's directory layout (numbered step folders, names like `best`, a separate EMA folder) | The deploy file names a run directory and a step. If the layout is not `<run>/<step number>/`, implement `resolve_checkpoint` |
| Which Python environment the model needs, and any environment variables its code reads | Goes in the deploy file ([Part 2](#part-2-the-deploy-file)) |
| Sampler settings (for example the number of denoising steps) | Read `self.settings.sample_steps`, so the operator can change it between rollouts; rebuild the sampler in `on_settings` |
| Whether the checkpoint records how it was trained (a config file, a hash, its own statistics) | Compare it in `__init__` with what the adapter assumes, and stop with a clear message on a mismatch |

### Write it

- **Keep `__init__` cheap and off the GPU**: configuration, tokenizer,
  statistics. It is constructed in places that only need to check a prompt.
- **Load the weights and compile in `warmup()`**, then call
  `super().warmup()`, which runs one inference on a dummy observation so that
  compilation is paid before the first rollout.
- **Put the adapter inside the training package if you can.** An adapter file
  in a package is imported as a member of that package, so it can import the
  training code's transforms directly.
- **`uses_cameras` is exactly what `obs.frames` contains.** A camera not
  listed is not delivered (it is still recorded).

To build the adapter in a script exactly as the infra builds it, for your own
checks, run this under the model's Python:

```python
from g1d_deploy.policy_contract.offline import build
policy, cfg, hw = build("path/to/deploy.toml", warmup=True)   # warmup=False: no weights
obs = policy.dummy_observation()
actions = policy.infer(obs, policy.reader())
```

### The prompt

A model that compiles its forward pass (JAX, or anything specialised to input
shapes) pads every prompt to one fixed length. `max_prompt_tokens` is that
length. It is a property of how the adapter is compiled, not a limit of the
model: a model trained on prompts of 42 tokens can be served at 60 so that the
operator has room for longer instructions. `count_prompt_tokens(text)` returns
how many tokens a prompt takes, template included, and the infra refuses a
prompt that would not fit instead of letting it be cut short silently. If the
template includes values that change during a rollout (for example a
discretised gripper state), return the largest count those values can produce.

### Mistakes that no automatic check catches

- Quaternion written xyzw where wxyz is expected (both have unit length).
- Tool-point poses returned as flange poses: every target is off by the
  length of the gripper, 18.5 cm on the Dex-1.
- Relative actions returned as absolute, or accumulated from the wrong
  reference.
- Left and right arms swapped.
- Gripper inverted or in the wrong units (clipping makes it look valid).
- Images resized differently from training.
- Normalisation statistics applied to the wrong keys, or not inverted on the
  output.
- State passed in a different frame or joint order than training used.
- `trained_control_hz` set to the camera frame rate instead of the data's
  action rate.

[Part 3](#part-3-verify-without-the-robot) describes the checks that do catch
these.

## Part 2: the deploy file

`deploy.toml` configures one deployment. It says which robot, which adapter,
which checkpoint, and how the infra should execute the policy's actions. Start
from `skills/deploy-a-policy/templates/deploy.toml`; `docs/deploying_a_policy/deploy_toml.md`
documents every key. Paths in it are relative to the file itself.

| key | set it to |
|---|---|
| `robot` | the robot profile whose cameras match the training setup, e.g. `"g1d_fisheye_dex1"` (`docs/setup/robot_toml.md`) |
| `prompt` | the task instruction, worded as in training |
| `[policy] entry` | the adapter class: `"policy.py:Policy"` |
| `[policy] python` | the model's Python (a venv's `bin/python`, or the wrapper script from [Setup](#setup)) |
| `[policy.env]` | environment variables for the policy process, e.g. `XLA_PYTHON_CLIENT_PREALLOCATE = "false"` so a JAX model does not claim the whole GPU |
| `[checkpoint] run`, `step` | the training run directory and the step to serve; or `path` for one step directory |
| `[home]` | leave `arm` out: the infra has a default home posture for tabletop tasks. Set it only if the user asks for a different one |
| `[control]` | how actions are executed; see below |

`[control]` has sensible defaults taken from the adapter's declarations
(`control_hz` from `trained_control_hz`, `action_horizon` from
`trained_action_horizon`). The two settings worth a decision:

- **`replan_period`**: how many actions of each chunk are executed before the
  next inference. Use the value the user evaluates with, or half the chunk.
- **`sync`**: `true` (the default choice) executes `replan_period` actions,
  then holds still while the next chunk is inferred. `false` (async) keeps the
  arm moving on the previous chunk during inference; it needs inference to be
  faster than one replan period, and the infra refuses a Start that cannot
  keep up.

These, the IK solver, the sampler steps and the prompt can all be changed
from the dashboard between rollouts without relaunching, so the deploy file
only sets the starting values. Any key can also be overridden for one launch
on the command line: `--step 60000`, `--prompt "..."`, or
`--set section.key=value` for any key.

## Part 3: verify without the robot

**Check 1: the files are consistent.** Resolves the deploy file as a launch
would, constructs the adapter in its own Python with the GPU hidden, and
checks the prompt length. Takes seconds and no GPU; run it after every change.

```bash
g1d_deploy check-policy path/to/deploy.toml --no-run
```

**Check 2: the policy loads and an inference completes.** Runs `warmup()`
and one `infer` on a dummy observation (black images, zero joints).

```bash
g1d_deploy check-policy path/to/deploy.toml
```

This one uses the GPU as a real launch does. Run `nvidia-smi` first; if
another compute job is on the GPU, ask the user before running it. It is
refused while a deployment is running.

**Check 3: the actions mean the right thing.** Checks 1 and 2 prove the
adapter runs. They cannot tell whether its actions are correct, because a
dummy observation has no right answer. Do every one of these that the user's
data allows, strongest first, and keep them in a script beside the adapter so
they can be re-run:

1. **Round-trip the training labels.** Take recorded poses from the training
   data, build the model's action labels from them with the training code,
   then decode those labels with the adapter's output conversion. The result
   must equal the recorded poses to about 1e-6. This catches every output
   convention mistake without a GPU.
2. **Run held-out training samples through `infer`.** Feed real frames and
   states from the validation split and compare the returned actions with the
   recorded ones, next to a "stay still" baseline. A convention mistake shows
   as errors of centimetres to tens of centimetres (about 18 cm for a
   tool-point/flange mix-up, about 30 cm for swapped arms).
3. **Replay a recording from this robot**, if one exists: the inspection
   dashboard's "policy checkpoint" mode draws the adapter's actions over what
   the robot actually did (`docs/deploying_a_policy/rollout_inspection.md`).
4. **Sanity-check absolute poses**: with all joints at zero, the flanges are
   at about (0.20, ±0.149, 0.095) m with identity rotation.

If the user already serves this model another way, read that code and compare
its preprocessing and decoding with the adapter's.

## Part 4: hand over

Report to the user:

- the files written and where;
- every assumption made in Part 1 and every question left open;
- which checks ran and their results, and which did not run and why;
- the command to launch, and what it does:

  ```bash
  source <checkout root>/.venv/bin/activate
  g1d_deploy run path/to/deploy.toml
  ```

  The arms move to the home posture **during startup, before Start is
  pressed**. The dashboard is at `localhost:8080` on the deploy computer (the port is
  in `machine.toml`).
  Start begins a rollout, Reset ends it, Exit (or Ctrl-C once) returns the
  arms and shuts down.
- **before the first Start on this robot, ask the user to look at the camera
  panes on the dashboard** and confirm that the pane named for the left wrist
  shows the left arm's view and the right one the right. Identical cameras
  cannot be told apart in software, so only a person can check this, and a
  policy fed crossed wrists moves wrongly. If they are crossed, write a robot
  file that extends the profile and exchanges the two cameras' ports
  (`docs/setup/robot_toml.md`, "Check the wrists by eye"), and point the
  deploy file's `robot` at it. Suggest a short first rollout with a hand near the
  emergency stop.

## When something fails

| symptom | where to look |
|---|---|
| `check-policy` or the launch stops with `PolicyError` or `ChunkError` | the message names the cause; `docs/deploying_a_policy/policy_contract.md`, "Errors", lists each one with its fix |
| A launch is refused before the arms move | the message is a startup check; if the robot was switched on moments ago, relaunch first (`docs/deploying_a_policy/field_notes.md`) |
| A prompt is refused as too long | `max_prompt_tokens` is the compiled length; see [the prompt](#the-prompt) |
| The checkpoint is refused by the adapter's own self-check | the checkpoint and the code disagree about how it was trained; `skills/deploy-a-policy/references/adapter_patterns.md` |
| The robot moves, but wrongly | a convention mistake: redo Check 3, then replay the recorded rollout in `g1d_deploy inspect` (`docs/deploying_a_policy/rollout_inspection.md`) to compare the policy's actions with what was executed |
| The arm pauses between chunks | expected in sync mode: the pause is the inference time. Async removes it |
