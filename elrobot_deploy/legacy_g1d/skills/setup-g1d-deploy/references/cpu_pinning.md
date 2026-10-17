# CPU pinning: writing a table by hand

Read this when `elrobot_deploy check` refuses to derive a CPU table for the
user's computer, or when the user asks to tune one. On a computer the
derivation accepts, there is nothing to do.

This is a task to work out yourself from the machine, not one to interview
the user about. The user needs two sentences at the end: what you wrote, and
what the computer will be slower at as a result.

- [Why processes are pinned](#why-processes-are-pinned)
- [The processes, by priority](#the-processes-by-priority)
- [What the automatic table needs](#what-the-automatic-table-needs)
- [The rules a hand-written table must satisfy](#the-rules-a-hand-written-table-must-satisfy)
- [Procedure](#procedure)
- [Trade-offs when cores are short](#trade-offs-when-cores-are-short)
- [What to tell the user](#what-to-tell-the-user)

The code is `src/elrobot_deploy/runtime/cpu_pinning.py`; the user-facing
description is `docs/setup/machine_toml.md`, "CPU pinning". A complete worked
table for a 24-core hybrid CPU is `tests/machine.toml`.

## Why processes are pinned

elrobot_deploy is about a dozen processes. One of them sends a motor command
every 2 ms. If the operating system moves a camera decoder or the policy onto
that process's core for a few milliseconds, the command is late and the arm
jitters. Pinning gives each process fixed cores so that this cannot happen. A
physical core with hyperthreading is two logical CPUs that share one set of
hardware; a time-critical process is given one of them and the other is left
empty ("reserved"), so that nothing shares the physical core.

Nothing here has been measured on a small computer. The priorities below come
from how each process works, and the reference machine has 24 physical cores.

## The processes, by priority

| pin name | what it does | load | if it is delayed |
|---|---|---|---|
| `p7_body` | the 500 Hz arm servo: interpolates targets, adds gravity compensation, publishes motor commands | busy-waits; a full core | the arm jitters. **Must have a whole physical core** |
| `p8_hand` | the same loop for the grippers, 200 Hz | part of a core | the grippers lag. **Must have a whole physical core** |
| `p6_target` | once per control tick: merges action chunks, solves IK, writes the joint target | spins without sleeping during a rollout; a full core | targets miss their deadlines and the servo stalls |
| `p1_state` | reads the robot's state at about 1 kHz, computes wrist poses | light, but must be prompt | stale state; at startup a low state rate refuses the launch |
| `p5_policy` | builds the observation and runs the policy | heavy during each inference; uses every core it is given for preprocessing | slower inference: a longer pause between chunks in sync mode, or async refused |
| `p2_camera`, `p3_camera`, `p4_camera` | one per camera of the robot file (right wrist, left wrist, head on the fisheye profile): decodes each JPEG frame | steady, moderate | older frames in the observation |
| a camera's own name (e.g. `exo`) | any further camera is pinned by its name | as above | as above |
| `p9_record` | writes every stream to disk | busy-waits; a full core | nothing: being late is harmless |
| `p10_dashboard` | the web dashboard | light | nothing |
| `p0_launcher` | supervises the others | light | nothing |
| `p11_pico` | only with `--intervene`: polls the headset | light | operator lag |

Order of importance when cores are short: `p7_body`, `p8_hand`, then
`p6_target`, `p1_state`, `p5_policy`, then the cameras, then the rest.

## What the automatic table needs

Derivation never uses physical core 0 (left to interrupts) and refuses when
it cannot give:

- `p7_body` and `p8_hand` a hyperthreaded physical core each, both threads;
- `p6_target`, `p5_policy` and `p1_state` one further performance core each;
- on a CPU without efficiency cores, four more cores for the cameras and the
  recorder.

So it needs about 6 hyperthreaded performance cores on a hybrid CPU, or about
10 hyperthreaded cores on a uniform one, and it always refuses a CPU without
hyperthreading. `python -m elrobot_deploy.runtime.cpu_pinning` prints what this
computer has and either the table it derives or why it cannot.

## The rules a hand-written table must satisfy

A table stated in `machine.toml` skips derivation and is checked at launch:

1. No two processes on the same logical CPU.
2. No process on a CPU listed in `[cpu] reserved`.
3. `p7_body` and `p8_hand` each have exactly one logical CPU, and every
   sibling of that CPU is in `[cpu] reserved`. On a CPU without
   hyperthreading there are no siblings, so `reserved` is empty.
4. `p7_body` and `p8_hand` must be pinned. Every other process may be left
   out of the table; it then runs unpinned, on whatever the operating system
   picks among all CPUs.
5. The CPUs named must exist on this machine with the siblings the machine
   really has (a table copied from another computer is refused).

`p5_policy` may list several CPUs; everything else takes one.

## Procedure

1. Read the topology: `python -m elrobot_deploy.runtime.cpu_pinning` and
   `lscpu -e` (columns CPU and CORE show which logical CPUs share a physical
   core).
2. Leave physical core 0 alone if there are cores to spare.
3. Give `p7_body` the fastest free physical core and `p8_hand` the next.
   Put one logical CPU of each in `[cpu.pin]` and its sibling in `reserved`.
4. Give `p6_target` and `p1_state` a physical core each.
5. Give `p5_policy` what performance cores remain (a list).
6. Give each camera a core. On a hybrid CPU use efficiency cores, each camera
   in a different cluster of four where possible.
7. If cores remain, pin `p9_record`, `p10_dashboard`, `p0_launcher`. If not,
   leave them out.
8. Write it into `machine.toml`:

   ```toml
   [cpu]
   reserved = [9, 11]        # the siblings of p7_body's and p8_hand's CPUs

   [cpu.pin]
   p7_body   = 8
   p8_hand   = 10
   p6_target = 2
   p1_state  = 6
   p5_policy = [4, 12, 14]
   p2_camera = 16
   p3_camera = 20
   p4_camera = 24
   ```

9. Run `elrobot_deploy check --no-robot`; it validates the table against this
   machine and names any rule that is broken.

## Trade-offs when cores are short

Apply in this order; each frees a core without touching the motor loops.
None needs a code change.

| free a core by | what it costs |
|---|---|
| leaving `p10_dashboard`, `p0_launcher`, `p9_record` unpinned | they may briefly land on a busy core; a small risk of a late state sample |
| running without recording (`elrobot_deploy run … --no-record`) | no recordings, so nothing to inspect afterwards |
| a robot file with fewer cameras | the policy must not need the camera that is dropped |
| giving `p5_policy` one CPU instead of several | slower inference. Use sync mode, or a longer replan period |
| a lower control rate, or the lighter `arm_qp` IK solver in the deploy file | less work per tick for `p6_target` (it still occupies its core) |
| putting `p1_state` on the sibling thread of `p6_target`'s core | the two contend a little; acceptable when nothing else is left |

Do not put anything on the siblings of `p7_body` and `p8_hand`, and do not
let those two share a physical core.

Roughly: 8 physical cores can run the three-camera rig with a hand-written
table and the compromises above. Below about 6, expect to drop cameras or
recording. These numbers are estimates, not measurements.

## What to tell the user

Keep it to the outcome: that this computer has fewer cores than elrobot_deploy
lays out automatically, that you wrote a core assignment into `machine.toml`,
and which compromises it contains in practical terms ("recording may
occasionally be delayed", "the policy will infer more slowly than on a larger
computer", "the head camera is not used"). Ask the user only when a
compromise removes something they may need: a camera, or recording.
