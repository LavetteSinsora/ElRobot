# elrobot_deploy: overview

elrobot_deploy runs a policy on a Unitree G1-D. A **policy** here is anything that
turns observations into actions: a trained neural network, a scripted
controller, or a bridge to an external planner or language model that emits
waypoints. It is plugged in through a small **adapter** class and a **deploy
file**. The infra does everything between "something that maps observations to
actions" and "motors following a smooth, safe trajectory": synchronising
cameras and state, scheduling inference, re-timing actions onto real
deadlines, blending chunks, inverse kinematics, the 500 Hz servo, the hands,
recording every rollout, and dashboards to run and to inspect it. It places no
requirement on what the policy is or how it was made, only on the format of
the actions it returns.

This page is the map. It gives the whole system in brief and says which
document holds the detail; read further only where you need to.

- **Code:** `src/elrobot_deploy/` (the table under "Source layout" below says what
  each folder holds; every module's docstring names the document that explains
  it). Working adapters and plug-ins: `examples/`.

## The documentation, by folder

**`docs/setup/`**: once per computer and per robot.

| doc | contains |
|---|---|
| `docs/setup/install.md` | installing on a new computer, the first checks, connecting a policy's Python environment |
| `docs/setup/machine_toml.md` | the optional machine file: overriding this computer's defaults (record location, dashboard port, network card, CPU pinning) |
| `docs/setup/robot_toml.md` | robot files: the board's address and login, cameras, end effector, tool point, the shipped profiles and how to extend one |

**`docs/deploying_a_policy/`**: everything needed to get a policy onto the robot and look at what it did.

| doc | contains |
|---|---|
| `docs/deploying_a_policy/policy_contract.md` | the adapter: what it declares, the observation it receives, the exact action format it returns |
| `docs/deploying_a_policy/deploy_toml.md` | the deploy file, key by key |
| `docs/deploying_a_policy/running_and_live_dashboard.md` | the commands, a launch from start to finish, the live dashboard, what can change between rollouts |
| `docs/deploying_a_policy/rollout_inspection.md` | the inspection dashboard: recorded rollouts, IK re-solves, policy replays, writing a comparison mode |
| `docs/deploying_a_policy/field_notes.md` | know-how from real runs that is not yet filed elsewhere |

**`docs/infra_logic/`**: how the infra works inside, for changing or extending it.

| doc | contains |
|---|---|
| `docs/infra_logic/arena_layout_and_reads.md` | the shared memory: regions, records, clocks, and reading it from an adapter or a tool |
| `docs/infra_logic/ik_solvers.md` | the shipped IK solvers and how to add one |
| `docs/infra_logic/chunk_blending.md` | how overlapping action chunks merge in async mode, and writing a blend |
| `docs/infra_logic/500hz_execution.md` | how joint targets become 500 Hz motor commands: the smoother and feedforward |
| `docs/infra_logic/development.md` | working on the infra itself: project layout, dependencies, tests |

A coding agent starts from the two skills: `skills/setup-g1d-deploy/` (install,
network, robot file, first checks), then `skills/deploy-a-policy/` (the adapter
and the deploy file). Both point into these documents.

## The processes

One launch is a dozen processes sharing one block of memory (the *arena*,
`docs/infra_logic/arena_layout_and_reads.md`). Each region has exactly one writer, and no
process ever waits on another's lock.

| process | code (`src/elrobot_deploy/runtime/`) | does |
|---|---|---|
| P0 launcher | `launcher/` | resolves the config, creates the arena, starts the others, checks the robot, homes the arm, runs the session (Start / Reset / Exit), exit ramp and damp |
| P1 state | `processes/p1_state.py` | reads `rt/lowstate` (~1 kHz), the hands and the column; computes flange FK; writes the state ring |
| P2… cameras | `processes/p2_camera.py` | one per robot-file camera: receives, decodes, splits the binocular head, writes camera rings |
| P5 policy | `processes/p5_policy.py` | runs the adapter under its own Python: schedules each inference, builds the observation, calls `infer`, validates and re-times the actions, publishes chunks |
| P6 target | `processes/p6_target.py` | per tick: blends the chunks covering it, runs the IK (eef), clamps, writes one joint target with its deadline |
| P7 body | `processes/p7_body.py` | 500 Hz: walks the target line, shapes it, adds gravity and feedforward torque, publishes `rt/lowcmd` |
| P8 hand | `processes/p8_hand.py` | 200 Hz: the same walk for the hands |
| P9 record | `processes/p9_record.py` | writes every stream of every rollout to disk |
| P10 dashboard | `processes/p10_dashboard.py` | the live dashboard |
| P11 PICO | `processes/p11_pico.py` | only with `--intervene`: an operator's headset targets |

## One chunk's life

Sync mode, 30 Hz, `P = 25`:

1. **Trigger.** P5 schedules the inference at the deadline of the last tick
   the previous chunk runs.
2. **Anchor.** Every observed camera takes its frame nearest that instant; the
   anchor camera's frame time becomes `t_anchor`; the state sample nearest it
   completes `obs`. Frames are copied out RGB.
3. **Infer.** `adapter.infer(obs, reader)` returns up to H actions, action `j`
   due `(j + 1) / 30` s after `t_anchor`.
4. **Validate and re-time.** P5 refuses non-finite values, non-unit
   quaternions, out-of-range hands or joints; shifts the chunk by the sub-tick
   gap between trigger and anchor; cuts it to P (sync); publishes it.
5. **Solve.** P6 adopts it, starts at the tick after next, and per tick blends
   (async), solves the IK against the measured trunk, clamps joint speed, and
   writes the target with its deadline.
6. **Servo.** P7 interpolates toward each target's deadline at 500 Hz,
   smooths, adds gravity and model feedforward torque, and publishes. P8 does
   the hands.
7. **Record.** P9 writes the frames, state, chunk, targets and commands.

**Sync vs async.** In sync, the arm finishes P actions, then waits (holding)
for the next chunk: latency becomes a pause at each seam. In async the arm
keeps moving on the previous chunk while the next is inferred, and the new one
takes over when it lands; inference must be shorter than a replan period
(checked at Start).

## What you can customize

| part | how | runs in | doc | example |
|---|---|---|---|---|
| the policy adapter | `[policy] entry` + `python` | P5, the adapter's own Python | `docs/deploying_a_policy/policy_contract.md` | `examples/policies/sine_eef.py`, `examples/policies/sine_joint.py` |
| observations beyond `obs` | `history` + `reader` | P5 | `docs/deploying_a_policy/policy_contract.md` | `examples/policies/history.py` |
| anything in shared memory (read-only) | `reader.arena` | P5 | `docs/infra_logic/arena_layout_and_reads.md` | `examples/rtc/` |
| checkpoint layout / aliases | `resolve_checkpoint` classmethod | resolution | `docs/deploying_a_policy/policy_contract.md` | — |
| the robot's cameras, hand, TCP | a robot file extending a profile | everywhere | `docs/setup/robot_toml.md` | `examples/robot_custom.toml` |
| the IK solver | `[control] ik = "pkg.mod:factory"` | P6, elrobot_deploy's Python | `docs/infra_logic/ik_solvers.md` | `examples/custom_modules/ik_solver.py` |
| how overlapping chunks merge | `[control] blend = "pkg.mod:fn"` | P6 | `docs/infra_logic/chunk_blending.md` | `examples/custom_modules/blend.py` |
| the 500 Hz smoother / torque feedforward | `[control.servo] smoother`, `feedforward` | P7 | `docs/infra_logic/500hz_execution.md` | `examples/custom_modules/servo.py` |
| an inspection comparison | `--mode pkg.mod:Class` | the inspection server | `docs/deploying_a_policy/rollout_inspection.md` | `examples/inspect_mode/` |

Every plug-in is named `"pkg.module:attr"` (importable) or
`"path/file.py:attr"` (relative to the deploy file; for `--mode`, to the
working directory), and imported at launch, so a typo fails before anything
moves. Plug-ins for P6 and P7 run under elrobot_deploy's own Python, not the
adapter's.

**Deliberately not customizable:**

- **P7's walk loop** (segment selection, stalls, the slew clamp): the same
  path runs homing, rollouts, interventions and the exit ramp; one tested loop
  is what makes them safe. Shape motion through the smoother instead.
- **The anchor rule and the deadline shift**: they define what "action `j` is
  due `(j+1)/control_hz` after `t_anchor`" means for every policy; an adapter
  trained on other timing converts inside `infer`.
- **P5's validation**: the infra refuses bad actions and never clips; clip in
  the adapter.
- **End effectors**: a hand's topics and units are product facts, built in.

## Capabilities, in brief

- Adapter in its own venv (numpy 2, torch, JAX) beside a numpy-1 robot stack.
- Two action contracts: absolute flange poses (IK in the loop) or joint angles.
- Sync and async inference; per-rollout switching of sync, replan period,
  control rate, IK solver, sample steps and servo, without reloading the policy.
- Prompt changes between rollouts, checked against the adapter's tokenizer.
- Four IK solvers with TCP-aware solving and collision limits; the waist
  planned or held.
- A smoothed 500 Hz servo with velocity and model torque feedforward.
- Full recording of every rollout, portable across machines.
- An inspection dashboard: timeline, cameras, 3-D, IK re-solves, checkpoint
  replays on recorded observations.
- `check` (box and robot, read-only) and `check-policy` (the adapter,
  offline) before anything moves.
- Optional PICO takeover with DAgger-ready records.

## Source layout

| folder (`src/elrobot_deploy/`) | holds |
|---|---|
| `cli/` | the `elrobot_deploy` commands |
| `policy_contract/` | what an adapter subclasses and returns; loading, checking, the anchor, the deadline shift |
| `config_loading/` | reading deploy, robot and machine files into one resolved config; validation |
| `robot/` | profiles, the G1-D model and assets, kinematics, end effectors |
| `runtime/` | the running system: `launcher/`, `processes/`, `shared_memory/`, `ik/`, `execution/`, `drivers/`, `teleop/` |
| `rollout_record_utils/` | the manifest and reading records |
| `dashboards/` | `live/`, `inspect/`, and the shared `web/` |

Limits known today are listed in `docs/deploying_a_policy/running_and_live_dashboard.md`,
"Known limits".
