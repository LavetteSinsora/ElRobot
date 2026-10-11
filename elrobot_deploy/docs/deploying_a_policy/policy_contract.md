# The policy contract

What a policy adapter is, what it declares, what it receives, and exactly what
it must return. This is the reference an adapter is written against.

**An adapter need not wrap a neural network.** Anything that returns actions
in the contract below is a policy: a trained model, a scripted controller, a
bridge that turns an external planner's or a language model's waypoints into
actions. The words "model", "checkpoint", "trained" and "inference" in this
document describe the common case; for anything else `[checkpoint]` is left
empty (`checkpoint` is then `None`), the `trained_*` attributes are left unset,
and `infer` is simply "compute the next actions" (`examples/policies/hold.py`,
`examples/policies/sine_eef.py`).

- **Read first:** `docs/overview.md` (what the infra does around the adapter).
- **Code:** `src/g1d_deploy/policy_contract/` — `base.py` (the `Policy`
  class), `actions.py` (`EefActions`, `JointActions`, validation),
  `observation.py` (`Observation`, `Reader`), `deadline_shift.py`,
  `anchor.py`.
- **Related:** `docs/deploying_a_policy/deploy_toml.md` (the file that names the adapter),
  `docs/infra_logic/arena_layout_and_reads.md` (reads beyond `obs` and `reader`),
  `examples/` (working adapters).

## Contents

1. [What an adapter is](#what-an-adapter-is)
2. [The smallest adapter](#the-smallest-adapter)
3. [The adapter's own Python](#the-adapters-own-python)
4. [Declarations](#declarations)
5. [Methods](#methods)
6. [Lifecycle](#lifecycle)
7. [The prompt](#the-prompt)
8. [The observation](#the-observation-obs)
9. [Cameras](#cameras)
10. [History: `reader`](#history-reader)
11. [The actions](#the-actions)
12. [What happens to the actions](#what-happens-to-the-actions)
13. [The waist](#the-waist)
14. [Joint order and limits](#joint-order-and-limits)
15. [Errors](#errors)

## What an adapter is

One Python class, a subclass of
`src/g1d_deploy/policy_contract/base.py::Policy`. It sits between the infra and
whatever decides the actions (in the common case, a trained model):

| the infra owns | the adapter owns |
|---|---|
| when to infer, and which instant is "now" (the anchor); cutting each fisheye to the square around its lens circle | preparing the observation the way the policy expects it (for a model, exactly as training did it): resizes, normalisation, any crop beyond the lens square |
| reading shared memory, converting frames to RGB | computing the actions: a model call with its sampler and inverse transform, a scripted rule, a request to an external planner |
| validating actions, re-timing them onto real deadlines | converting the policy's action space into the contract (frame, units, order, TCP to flange, gripper units) |
| blending chunks, IK, the waist, the 500 Hz servo, the hand | interpreting the prompt (tokenising it and knowing its length limit, if it has one) |
| recording everything | loading whatever it needs (a checkpoint, if any) |

The line is "what the policy decided" versus "what the robot needs". Anything
the policy's maker chose — image size, stats, action representation — is the
adapter's, because only it can know. Anything about when and how the robot
moves is the infra's, so every policy gets the same, tested behaviour.

## The smallest adapter

```python
import numpy as np
from g1d_deploy.policy_contract import EefActions, Policy

class MyPolicy(Policy):
    action_mode = "eef"            # required: "eef" | "joint"
    end_effector = "dex1"          # required: the hand's units
    anchor_camera = "head_left"    # required: whose frame time is "now"

    def __init__(self, checkpoint, prompt, robot):
        super().__init__(checkpoint, prompt, robot)   # always call this
        self.norm = load_norm_stats(checkpoint)       # cheap things only

    def warmup(self):
        self.model = restore(self.checkpoint)         # the expensive part
        super().warmup()                              # one dummy infer: compile

    def infer(self, obs, reader):
        rgb = obs.frames["head_left"].pixels          # uint8 (480, 640, 3), RGB
        pose, grip = self.model(rgb, obs.state, obs.prompt)
        return EefActions(pose=pose, hand=grip)       # (n, 2, 7), (n, 2)
```

Working adapters, each with a `deploy.toml` that runs it:

| example | shows |
|---|---|
| `examples/policies/hold.py` | eef; holds a latched pose (bring-up, and the default of `examples/deploy.toml`) |
| `examples/policies/sine_eef.py` | eef; both flanges trace a 2 cm sinusoid |
| `examples/policies/sine_joint.py` | joint; both elbows bend on a sinusoid |
| `examples/policies/history.py` | `history` + `reader`: stacked frames and a state window |
| `examples/rtc/policy.py` | read-only arena access: which ticks of its last chunk will already have run (real-time chunking) |

`skills/deploy-a-policy/templates/` has annotated skeletons to start from.

## The adapter's own Python

P5, the policy process, runs under the adapter's interpreter, so the model's
own stack (torch, JAX, numpy 2, a tokenizer) never has to share a venv with the
robot side (which pins numpy 1.x and carries MuJoCo, pinocchio, CycloneDDS):

```toml
[policy]
entry  = "policy.py:MyPolicy"
python = "~/projects/my_model/.venv/bin/python"   # default: g1d_deploy's own
```

A relative path is from the deploy file's folder, a bare name from `PATH`.
Symlinks are not followed (a venv's `bin/python` is a symlink out of the venv).

**That venv must have g1d_deploy's core installed**, with no extras:

    ~/projects/my_model/.venv/bin/python -m pip install -e /path/to/g1d_deploy

The core needs only numpy (1.x or 2.x). Install the same checkout into both
venvs: the arena's schema version is checked on attach, and a mismatch is
refused at launch.

- **The runtime never imports the adapter.** Before anything starts, the
  deploy file's resolution asks the adapter's interpreter for the class's
  declarations (`python -m g1d_deploy.policy_contract.describe`, GPU hidden),
  so an adapter that fails to import fails the launch with its own traceback.
- **Configuration reaches P5 as JSON**
  (`src/g1d_deploy/config_loading/json_transport.py`), never pickled, so the
  two numpy versions never meet.
- **Don't import the robot side.** Nothing in `g1d_deploy.policy_contract`
  needs MuJoCo; `obs.state.flange` is computed by P1, not in your process.
  `tests/test_policy_interpreter.py` pins the modules P5 may load.
- **Replay needs one extra.** Replaying the policy on a recording
  (`docs/deploying_a_policy/rollout_inspection.md`) decodes JPEGs, so that venv also needs the
  `replay` extra (`pip install -e '/path/to/g1d_deploy[replay]'`).
- **Or a wrapper instead of an install.** `[policy] python` may be a script
  that sets `PYTHONPATH` (the model's repo, and g1d_deploy's `src/`) and any
  environment variables the model's code needs **at import time**, then
  `exec`s a venv's python with `"$@"`. The runtime keeps using the script, not
  the python it runs. Variables in `[policy.env]` are set by P5 only, so
  anything a module reads while being imported belongs in the wrapper (or the
  shell) instead.
- **Your adapter can live inside the training package.** An entry file inside
  a package (every folder up to it has an `__init__.py`) is imported under its
  dotted package name, so relative imports into the training code
  (`from ..data import transforms`) work. Calling the training pipeline's own
  transforms from there is the surest way to match its preprocessing.

## Declarations

Class attributes. They are read from the class, before anything is
constructed, so a wrong one costs no slow load. Each is used for
something specific:

| attribute | required | what the infra does with it |
|---|---|---|
| `action_mode` | yes | `"eef"`: the chunk carries flange poses and P6 runs the IK. `"joint"`: it carries joint angles and P6 skips only the IK. Decides which action type `infer` must return. |
| `end_effector` | yes | What `hand` means. `"dex1"`: the Dex-1 jaw in gear radians, 0 = closed. Checked against the robot file's `[end_effector] type`; a mismatch is refused, so a policy made for one hand never drives another. |
| `anchor_camera` | yes | Whose frame time is `obs.t_anchor`, the instant the observation depicts and action timing counts from. The state is matched to that instant. Pick the camera the model relies on most for fine motion (a wrist camera for a wrist-driven policy): a half-period skew between that image and the state is ~5 mm at p99. |
| `uses_cameras` | no | Exactly the cameras anchored and delivered in `obs.frames` (and the dummy observation); any other camera is not in `obs.frames` at all. Also what the dashboard marks and the manifest lists. Recording is unaffected: every camera is always recorded. Must include `anchor_camera`. Unset: every camera. |
| `history` | no | Seconds of past data per stream, e.g. `{"state": 1.0, "head_left": 4.0}`. Sizes each camera's ring in shared memory at launch, and gates `reader`: an undeclared stream raises. See [History](#history-reader). |
| `max_prompt_tokens` | no | The prompt length the adapter's forward pass is built for, in the model's own tokens: the fixed length it pads prompts to and compiles at in `warmup`. Not a training limit (see [The prompt](#the-prompt)); a longer prompt is refused rather than silently truncated. When the limit comes from the checkpoint, set `self.max_prompt_tokens` in `__init__`: every check reads the instance. See [The prompt](#the-prompt). |
| `trained_control_hz` | no | The action rate the policy was made for (for a model, that of its training data). Becomes the default `control_hz`; running at another rate is allowed and warned about. Unset: the infra's default. |
| `trained_action_horizon` | no | Actions per chunk the policy produces. Becomes the default `action_horizon` and its ceiling. Unset: the infra's default, no ceiling. |
| `deadline_shift` | no | Default `True`: the infra re-times each chunk onto its real deadlines ([below](#what-happens-to-the-actions)). Turn off only to A/B that correction. |
| `deadline_shift_point` | no | Default `(0, 0, 0)`. Flange-frame point (m) where eef poses are interpolated during the shift. Set it to your TCP if the model's poses are TCP poses converted to flange. |

**Mental model for `control_hz`.** Consecutive actions of a chunk are executed
`1 / control_hz` apart. The training-consistent value is the spacing of
consecutive actions in the training data (`trained_control_hz`). A faster rate
plays the same trajectory faster, a slower one slower; both run, with a
warning, so speed is one flag to A/B. It does not need to equal the camera
rate or the data's recording rate.

## Methods

| method | required | default behaviour |
|---|---|---|
| `__init__(checkpoint, prompt, robot)` | yes; call `super().__init__` | Stores the three. `checkpoint` is a `Path` or `None`; `robot` is `src/g1d_deploy/config_loading/resolved.py::Hardware`: `joints_arm`, `joints_waist`, `urdf`, `hand_min`, `hand_max`, `tcp`. Load only cheap things (config, tokenizer, norm stats) and never the GPU: the inspection dashboard constructs the class just to check a prompt. |
| `infer(obs, reader)` | yes | Returns `EefActions` or `JointActions`. |
| `warmup()` | no | One `infer` on the infra's dummy observation. Override to restore weights, then call `super().warmup()` to compile. |
| `count_prompt_tokens(text)` | no | `None`: never refused for length. Return the model's token count, template included. |
| `check_prompt_length(text)` | no | Raises `PromptTooLong` when the count exceeds `max_prompt_tokens`. Rarely overridden. |
| `set_prompt(text)` | no | Checks, then stores `self.prompt`. Called at a Start whose prompt changed. |
| `on_settings(settings)` | no | Ignores them. Called at a Start whose rollout settings changed; `settings` is `src/g1d_deploy/config_loading/rollout_settings.py::RolloutSettings` (`sync`, `replan_period`, `control_hz`, `ik`, `sample_steps`, `servo`). Raise to refuse the Start. A JAX policy compiles a sampler per `sample_steps` here. Read the argument: `self.settings` still holds the previous settings until this returns. |
| `resolve_checkpoint(run, step)` | no (classmethod) | `None`: the infra's rule (numbered step folders, `"latest"` = highest). Return a `Path` for your own layout or aliases like `"best"`. |

Bound by the infra, for the adapter to read:

| attribute | what |
|---|---|
| `self.checkpoint`, `self.prompt`, `self.robot` | from the constructor; `self.prompt` follows `set_prompt` |
| `self.cameras` | per camera view, a `CameraInfo` ([Cameras](#cameras)); empty inside `__init__` |
| `self.settings` | the current rollout's `RolloutSettings` (`replan_period`, `control_hz`, `sync`, ...): the launch's until a Start changes them. `on_settings` is the hook for a change; this is for reading them any time. `None` inside `__init__` |
| `self.dummy_observation()`, `self.reader()` | what `warmup` uses |

**Building an adapter offline**, for your own checks, exactly as P5 builds it
(in the adapter's interpreter): `policy, cfg, hw =
g1d_deploy.policy_contract.offline.build("deploy.toml")`, with `warmup=True`
to load the weights (`src/g1d_deploy/policy_contract/offline.py::build`).

## Lifecycle

All of it runs in P5, under `[policy] python`.

1. **Load.** The class named by `[policy] entry` is imported
   (`"pkg.module:Class"` or `"path/file.py:Class"`), its declarations
   validated and checked against the robot's cameras and ring sizes.
2. **Construct.** `Class(checkpoint, prompt, robot)`. Keep it cheap.
3. **Warm up.** The infra binds a dummy observation, a dummy reader,
   `self.cameras` and `self.settings`, then calls `warmup()` once. Restore weights and compile
   here, never lazily in `infer` (a lazy 40 s compile looks like a hang). The
   dummy reader serves every declared history at full length and rate —
   4 s of a 30 Hz camera is 121 black frames — so the shapes compiled at
   warmup are the shapes seen live.
   The dummy observation is black frames at each camera's shape, zero
   joints and velocities, both flanges at the base origin with identity
   orientation, a closed hand, and `t_anchor = 0` — valid poses, so quaternion
   normalisation and inter-arm features do not divide by zero.
4. **Measure.** P5 keeps calling `infer` on the dummy until its latency
   settles, publishes the measurement (async rollouts are checked against
   it), and only then reports warm. It keeps inferring on the dummy through
   homing until a rollout starts.
5. **Between rollouts.** P5 judges prompts the operator submits
   (`check_prompt_length`). At an accepted Start it calls `set_prompt` and
   `on_settings` if they changed, then runs one dummy `infer`, so a recompile
   is paid before the rollout.
6. **Run.** `infer(obs, reader)` once per replan, and at the start of every
   rollout and after every intervention handback. One object lives for the
   whole launch, across rollouts.

## The prompt

The adapter owns its tokenizer, so it owns the length limit:
`max_prompt_tokens` and `count_prompt_tokens(text)`. `check_prompt_length`
compares them and raises `PromptTooLong` with the sentence the dashboard
shows: "Prompt exceeds the policy's maximum token length (N tokens > M)."

If the model's prompt includes anything that changes per observation (a
digitised gripper state, a counter), `count_prompt_tokens` must return the
**worst case** over those values: the check runs on the operator's text before
any observation exists, and a prompt that fits at one gripper value and not at
another would truncate mid-rollout.

**What the limit is, and is not.** It is not how long a prompt the model was
trained on: a model trained on prompts of at most 42 tokens may do fine with 60.
It is the input shape the adapter compiles its forward pass at: a JAX (or any
shape-specialised) model pads every prompt to one fixed length in `warmup`, so
every later inference has exactly that shape and needs no recompile. A prompt
longer than that length would be cut to fit, and the robot would act on half an
instruction with nothing saying so. So set `max_prompt_tokens` to the length
you pad and compile at, choosing it as long as the prompts you expect need (the
training length is a fine default, not a ceiling), and the check refuses only
what would not fit.

Where it is checked: live when the operator submits a prompt and again at
Start; before a policy replay; and by the inspection dashboard's "Check
prompt" button, which constructs the class with the GPU hidden and never
calls `warmup` or `infer`. That button is as fast as `__init__`, which is the
second reason `__init__` loads only the tokenizer and stats.

## The observation (`obs`)

Built at the instant the anchor rule picks: every observed camera takes its
frame nearest the scheduled trigger, the anchor camera's frame time becomes
`t_anchor`, and the state sample nearest `t_anchor` completes it.

| field | type | meaning |
|---|---|---|
| `obs.t_anchor` | float | the anchor frame's receive time: the instant this observation depicts |
| `obs.frames[cam].pixels` | uint8 `(H, W, 3)` | **RGB** at native resolution, a private copy (the wire is BGR; the infra converts once): the whole view, or for a fisheye the square around its lens circle (see [Cameras](#cameras)) |
| `obs.frames[cam].t_recv` | float | when that frame reached the box |
| `obs.frames[cam].frame_id` | int | the camera's frame counter; what `reader.frames_by_id` takes |
| `obs.state.t_recv` | float | when the state sample reached the box |
| `obs.state.q`, `obs.state.dq` | `(29,)` | measured joint angles (rad) and velocities (rad/s), wire order; index with `self.robot.joints_arm` / `joints_waist`, never literals |
| `obs.state.flange` | `(2, 7)` | `[left, right]` flange poses from FK of this `q`, in the [base frame](#the-actions) |
| `obs.state.hand` | `(2,)` | measured hand, in the units `hand` actions take (Dex-1 gear radians) |
| `obs.prompt` | str | the current prompt |
| `obs.t_trigger` | float | the scheduled instant (logging; the re-timing already accounts for `t_anchor − t_trigger`) |

**Clock.** Every time here is box-monotonic seconds (`time.monotonic()` on
the deploy box), stamped when the data arrives. The camera-to-box transport
delay is not corrected.

## Cameras

Camera names come from the robot file (`docs/setup/robot_toml.md`), with one
exception: a camera declared `layout = "side_by_side"` (the G1-D head, one
480×1280 frame holding both eyes) is split into two views, **`head_left`**
(columns 0–639, the left eye) and **`head_right`** (640–1279), each 480×640
with its own ring. Use the two view names everywhere (`anchor_camera`,
`uses_cameras`, `history`, `obs.frames`, `reader`); `head` is refused with the
two names it meant. Both halves of one frame share `t_recv` and `frame_id`.

**Fisheye frames arrive square-cropped.** A fisheye (`kind = "fisheye180"`)
view is the square around its lens circle, cut by the infra: at launch the
lens circle `(cx, cy, r)` is fitted from live frames, the square is the circle
plus 2 px of air (side `2 (r + 2)`, so the disc nearly fills it), and P2 cuts
every frame to it before it enters the ring (off-sensor parts are black).
`obs.frames`, the reader and the replay all see the same square, so an adapter
applies no crop of its own; resize it as training did. The live dashboard
outlines the square and the circle on each fisheye pane; whether the lens is
the one the data was collected with is for a human to see there.

`self.cameras[view]` (`src/g1d_deploy/policy_contract/base.py::CameraInfo`):

| field | meaning |
|---|---|
| `name`, `source` | the view, and the robot-file camera it is cut from (`head_left` ← `head`) |
| `height`, `width`, `x0` | the view's size (a fisheye's square side, both), and its first column in the wire frame |
| `hz` | the frame rate the robot file states |
| `kind` | `"normal"` or `"fisheye180"` |
| `crop`, `crop_source` | for information only (the pixels are already cut): where a fisheye's square sits on the sensor, `(x0, y0, x1, y1)`, `x1`/`y1` exclusive, possibly running off the sensor (`src/g1d_deploy/runtime/drivers/fisheye_circle_fit.py::crop_for`). Live: the launch's fit (`"fitted"`). In check-policy, where nothing is fitted: the robot file's `fallback_lens_circle` (`"fallback_lens_circle"`), else a sensor-height square (`"placeholder"`, right in shape only). `None` for a normal lens. |
| `lens_circle` | the lens circle `(center_x, center_y, radius)`, full-sensor pixels, the square was cut around, when known |
| `intrinsics`, `extrinsics` | the robot file's optional calibration, passed through untouched |

Why the lens is fitted every launch: its circle sits a few pixels differently
on each unit and after each remount, and a model trained on circle-centred
crops is sensitive to that offset.

## History: `reader`

Declare what you read, then read everything captured in a window. Each item
carries its own timestamp; aligning and subsampling are yours.

```python
class MyPolicy(Policy):
    history = {"state": 1.0, "head_left": 4.0}

    def infer(self, obs, reader):
        t = obs.t_anchor
        w = reader.states_between(t - 1.0, t)                  # every state sample, ~1 kHz
        ft = reader.frame_ids_between("head_left", t - 4.0, t) # ids + times, no pixels
        picks = nearest_to_times(ft, t - np.array([3.0, 2.0, 1.0, 0.0]))
        frames = reader.frames_by_id("head_left", picks.ids)   # copy only these
```

| call | returns |
|---|---|
| `reader.states_between(t0, t1)` | `StateWindow`: `t_recv (m,)`, `q (m, 29)`, `dq (m, 29)`, `flange (m, 2, 7)`, `hand (m, 2)` |
| `reader.frame_ids_between(cam, t0, t1)` | `FrameTimes`: `ids (m,)`, `t_recv (m,)`; metadata only |
| `reader.frames_by_id(cam, ids)` | `list[Frame]` for those ids, in order asked, RGB, private copies |
| `reader.frames_between(cam, t0, t1)` | every frame in the window |
| `nearest_to_times(x, times)`, `subsample_every(x, dt)` | helpers over any of the above (`from g1d_deploy.policy_contract import ...`) |

Rules, and why:

- **Windows are closed** (`t0 <= t_recv <= t1`), ascending, and return what
  exists: early in a rollout you get less, never padding and never an error.
- **Items can be missing.** Each is copied out of shared memory and verified
  afterwards; one overwritten during the copy is dropped, never returned torn.
  So `frames_by_id` may return fewer frames than asked: compare `frame_id`s if
  every one matters. Nothing returned aliases shared memory.
- **Cost:** one copy per frame (0.9 MB for a 480×640 view). Pick ids with
  `frame_ids_between`, then copy only those.
- **Undeclared streams raise `LookupError`**, because the declaration is what
  sized the ring: a camera ring holds `ceil(seconds × fps × 1.25) + 3`
  frames. 4 s of a 30 Hz 480×640 view is 153 slots, ~140 MB of `/dev/shm`. The
  launch refuses if `/dev/shm` cannot hold the arena, and prints the numbers.
  `"state"` may not exceed the deploy file's `[control] state_history_s`.

**Beyond `reader`.** `reader.arena` gives read-only access to every region of
shared memory — where execution is now (`k_exec`), the chunks P5 published,
P6's targets, P7's commands — for adapters that need what the windows above
don't offer, such as real-time chunking. It is `None` whenever there is no live
rollout: warmup, the dummy inference at each Start, check-policy and a policy
replay. Its rules and clocks are in `docs/infra_logic/arena_layout_and_reads.md`.

## The actions

**Action `j` is due `(j + 1) / control_hz` seconds after `obs.t_anchor`** —
the timing training labelled. Return the whole chunk the model produces: P5
cuts it to `action_horizon` (H) when it is longer, and in sync mode to `P`
(so in sync return at least `P`). Never cut it yourself; the adapter need not
know H.

### The base frame

Every pose (actions, `obs.state.flange`) is in the robot's **base frame**:

- **Attached to** `LZ_it_Link`, the link on top of the lift column
  (`src/g1d_deploy/robot/g1d_model.py::BASE_BODY`). It rides the column and
  the wheeled base, so column height never changes a pose; the waist (yaw and
  bow) sits above it, so bowing moves the flanges *within* the frame.
- **Origin** at the waist-yaw axis, where the Unitree humanoid's pelvis sits
  (`g1d_model.py::PELVIS_FROM_TORSO`), so poses match the G1 humanoid's
  pelvis frame at waist zero.
- **Axes:** x forward, y left, z up.
- **For scale:** at zero joint angles the flanges are at about
  (0.20, ±0.149, 0.095) m with identity orientation.

### `EefActions` (`action_mode = "eef"`)

| field | shape | convention |
|---|---|---|
| `pose` | `(n, 2, 7)` | `[left, right]`; `[x, y, z, qw, qx, qy, qz]`; metres; **wxyz** unit quaternion; the **flange** (`{left,right}_wrist_yaw_link`); **absolute** in the base frame (not deltas) |
| `hand` | `(n, 2)` | `[left, right]`, the end effector's units. Dex-1: jaw gear radians, **0 = fully closed … 5.4 = fully open** |

**Flange, not TCP.** The pose is the wrist-yaw link's origin. If the model
predicts a tool-centre point, convert: `T_flange = T_tcp @ inv(T_flange_tcp)`.
The robot file's TCP for the mounted hand is `self.robot.tcp`, a `(7,)`
flange-frame pose (Dex-1: `[0.184759, 0, 0, 1, 0, 0, 0]`, the jaw-tip
midpoint 18.5 cm along the flange's +x), or `None`. The IK then solves your
flange target composed with that TCP, so an unreachable pose is missed as
little as possible at the fingertips rather than at the wrist
(`docs/infra_logic/ik_solvers.md`).

**Gripper units.** The contract accepts Dex-1 gear radians and nothing else.
If the model's gripper is a width, a normalised 0–1 value, or inverted,
convert in the adapter and clip to `[robot.hand_min, robot.hand_max]`.

**No waist** in eef mode; see [The waist](#the-waist).

### `JointActions` (`action_mode = "joint"`)

| field | shape | convention |
|---|---|---|
| `arm_q` | `(n, 2, 7)` | radians, `[left, right]`, per-arm order in the [table](#joint-order-and-limits) |
| `waist_q` | `(n, 3)` | **required**, radians, `[yaw, bow, 0]`. For an upright trunk return zeros |
| `hand` | `(n, 2)` | as eef |

Construct both by keyword. In joint mode no IK runs; the dashboard and banner
say `ik: not used (joint actions)`.

## What happens to the actions

In P5, in this order (`src/g1d_deploy/runtime/processes/p5_policy.py`):

1. **Cut to H, then validate.** A chunk longer than `action_horizon` keeps
   its first H actions. Then: the type matches `action_mode`; shapes match;
   every value is finite; quaternions are unit length; `hand` is within the end effector's
   travel; joint angles are within URDF limits; `n ≥ 1`, and `n ≥ P` in
   sync. A failing chunk is refused. **The infra refuses, it never clips:**
   clip and normalise in the adapter.
2. **Deadline shift.** The anchor frame is the one nearest the trigger, so
   `t_anchor` differs from the trigger by up to half a camera period, and
   every action is due that much earlier than the chunk's timing says. The
   chunk is resampled by that fraction of a tick (at most one tick): positions
   and joints linearly, rotations by slerp, row 0 being the measured state at
   `t_anchor`.
3. **Clip** only what the shift itself pushed past a limit.
4. **Truncate** to `P` in sync (after the shift, so the extrapolated tail is
   never commanded).
5. **Publish** to the chunk ring. From there: P6 blends overlapping chunks
   (`docs/infra_logic/chunk_blending.md`), runs the IK (eef) and a velocity clamp, and
   writes one joint target per tick; P7 follows the targets at 500 Hz
   (`docs/infra_logic/500hz_execution.md`); P8 drives the hand; P9 records everything.

A chunk's action 0 is never executed: P6 starts at the tick after next
(`k_exec + 2`), leaving itself one tick to solve.

## The waist

| policy | waist commanded | trunk the arm IK solves against |
|---|---|---|
| eef, arm-mode IK (`arm_planner`, `arm_qp`) | 0 (upright) | the **measured** trunk |
| eef, waist-mode IK (`waist_planner`, `waist_qp`; sync only) | the planner's trajectory | measured + the plan's increment |
| joint | `waist_q` as returned | no IK |

The IK always solves against the measured trunk: on the reference robot the bow rests
against a stop 0.039 rad forward of zero, and solving against the commanded
zero puts every flange 8–11 mm off.

## Joint order and limits

`arm_q[:, side, i]`. `b` is the index in the 14-joint arm vector (left 0–6,
right 7–13); `w` is the index in `obs.state.q`, from the robot file's
`[joints] arm`. Limits from `src/g1d_deploy/robot/assets/unitree_g1d/g1_d.urdf`,
radians.

| i | joint | left b / w / URDF | right b / w / URDF | axis | left limits | right limits |
|---|---|---|---|---|---|---|
| 0 | shoulder pitch | 0 / 15 / `left_shoulder_pitch_joint` | 7 / 22 / `right_shoulder_pitch_joint` | y | [−3.0892, 2.6704] | [−3.0892, 2.6704] |
| 1 | shoulder roll | 1 / 16 / `left_shoulder_roll_joint` | 8 / 23 / `right_shoulder_roll_joint` | x | [−1.5882, 2.2515] | [−2.2515, 1.5882] |
| 2 | shoulder yaw | 2 / 17 / `left_shoulder_yaw_joint` | 9 / 24 / `right_shoulder_yaw_joint` | z | [−2.618, 2.618] | [−2.618, 2.618] |
| 3 | elbow | 3 / 18 / `left_elbow_joint` | 10 / 25 / `right_elbow_joint` | y | [−1.0472, 2.0944] | [−1.0472, 2.0944] |
| 4 | wrist roll | 4 / 19 / `left_wrist_roll_joint` | 11 / 26 / `right_wrist_roll_joint` | x | [−1.9722, 1.9722] | [−1.9722, 1.9722] |
| 5 | wrist pitch | 5 / 20 / `left_wrist_pitch_joint` | 12 / 27 / `right_wrist_pitch_joint` | y | [−1.6144, 1.6144] | [−1.6144, 1.6144] |
| 6 | wrist yaw | 6 / 21 / `left_wrist_yaw_joint` | 13 / 28 / `right_wrist_yaw_joint` | z | [−1.6144, 1.6144] | [−1.6144, 1.6144] |

Waist, `waist_q[:, s]`:

| s | motion | wire w | URDF name | axis | limits |
|---|---|---|---|---|---|
| 0 | yaw about the column | 12 | `torso_Joint` | z | [−2.7053, 2.7053] |
| 1 | forward bow | 13 | `Yaw_Joint` | y | [−0.04363, 2.3562] |
| 2 | (no motor) | 14 | — | — | must be 0 |

The G1-D's waist joint names are swapped relative to their motions
(`Yaw_Joint` bows, `torso_Joint` yaws): map by slot, never by name.

**Signs.** Radians from the URDF zero; positive is right-handed about the
joint axis in its parent frame. The shoulder-roll limits are mirrored, so the
same positive roll lifts the left arm out and moves the right arm in: an
adapter mirroring one arm onto the other flips roll and yaw.

## Errors

| error | cause |
|---|---|
| `PolicyError: ...action_mode ... REQUIRED` | missing or not `"eef"`/`"joint"` |
| `PolicyError: ...end_effector ...` | missing, or not the robot file's hand |
| `PolicyError: ...anchor_camera '...' is not in uses_cameras ...` | the class's anchor is missing from its own `uses_cameras` |
| `PolicyError: ...anchor_camera '...' is not among the observed cameras ...` | the anchor is not delivered in this run (a deploy file's `uses_cameras` left it out, or it is not a camera of this robot) |
| `... 'head' is a side_by_side camera and is split into ['head_left', 'head_right']` | name an eye |
| `PolicyError: ...history asks for ... but this run's camera rings hold ...` | launch through a deploy file, which sizes rings from the class |
| `LookupError: reader....() needs history = {...}` | declare the stream |
| `... /dev/shm has N MB free ...` | declared history does not fit; shorten it or grow `/dev/shm` |
| `ChunkError: ... has shape (...), the contract wants (...)` | wrong array shape (e.g. `(n, 7)` for a two-arm `(n, 2, 7)`) |
| `ChunkError: ... non-finite value(s)` | NaN or inf in the actions |
| `ChunkError: ...hand[...] = ... is outside ...` | clip `hand` to `[hand_min, hand_max]` (a joint angle outside its URDF limits reads the same way) |
| `ChunkError: EefActions.pose[...] has a quaternion of norm ...` | normalise the quaternion |
| `ChunkError: sync needs at least replan_period_ticks = P actions` | return at least `P` actions (ideally `H`) |
| `ChunkError: action_mode = 'eef' but infer() returned JointActions` | return the declared type |
| `ChunkError: JointActions.waist_q is None` | return the waist; zeros for upright |
