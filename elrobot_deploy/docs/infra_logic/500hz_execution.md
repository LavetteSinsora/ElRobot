# 500 Hz execution: from joint targets to motor commands

P6 writes one joint target per tick (30 Hz at the default rate). The motors
want a command every 2 ms. P7 fills the gap: it walks the line through the
targets, optionally smooths it and adds a torque feedforward, and publishes
position, velocity and torque at 500 Hz. The smoother and the feedforward can
be your own; the walk itself is fixed.

- **Read first:** `docs/overview.md`.
- **Code:** `src/elrobot_deploy/runtime/processes/p7_body.py` (`BodyServo`),
  `src/elrobot_deploy/runtime/execution/servo_segment.py` (`Segment`: which
  target is in flight), `src/elrobot_deploy/runtime/execution/servo_filter.py`
  (`Shaping`, `Smoother`, `Feedforward`, `CritDamped`, `ModelFeedforward`,
  `build`, `load`).
- **Related:** `docs/deploying_a_policy/deploy_toml.md` (`[control.servo]`),
  `docs/setup/robot_toml.md` (`[servo_model]`, gains), `examples/custom_modules/servo.py`.

## What P7 does every 2 ms

1. **Find the segment.** The targets form a line of `(deadline, q)` points.
   P7 interpolates from where it stands toward the next target with a future
   deadline (`Segment.phase`); when that deadline passes it moves to the next
   one and publishes that tick as `control.k_exec`. If there is no future
   target it **holds** its last command (a stall) — which is how the arm
   waits through a sync inference, a hole, or the end of a chunk.
2. **The line.** The linear walk gives a position and its slope (the
   velocity the line implies).
3. **Shape** (optional): a smoother turns the line into a smooth
   `(q, v, a)`, reading it slightly ahead so smoothing adds no lag.
4. **Clamp.** One slew limit per joint per period — the last defence; in
   normal operation it never binds, because P6 already clamped each target's
   step.
5. **Publish** `q`, `dq = v × velocity_ff` (clamped to the joint speed limit),
   and `tau = gravity compensation (measured joints) + feedforward(v, a)`,
   with the robot file's gains (`kp` scaled by `arm_gain_scale`, `kd` by its
   square root).
6. **Log** the command (`COMMAND` ring: `q`, `dq`, `tau`, `tau_ff`, and
   `q_lin`, the plain linear walk at that instant), which P9 records.

The same walk runs everything the arm does: homing, every rollout, the
intervention takeover and handback, and the exit ramp. P7 knows nothing about
chunks, policies or sessions, and never sends a damp (that is P0's). P8 runs
the same segment walk for the hands at 200 Hz.

Why a velocity is published at all: without it the firmware's `kd` term reads
every commanded motion as velocity error and brakes it, and the arm trails its
target by `kd/kp` seconds (measured 37–42 ms).

## The shipped shaping

`[control.servo]` (per rollout, editable on the dashboard):

| knob | default | effect |
|---|---|---|
| `filter_hz` | 4.0 | a critically damped filter (`CritDamped`) per joint: the fastest response that never overshoots. At 4 Hz it halved 4–40 Hz judder on the robot with no loss of hand accuracy. `0` = the straight-line walk |
| `filter_lead` | true | read the line ahead by the filter's own lag (`2 / (2π·filter_hz)`, 80 ms at 4 Hz), capped at the last written target, so smoothing costs no delay |
| `velocity_ff` | 1.0 | scale on published `dq` |
| `torque_ff` | 0.5 | scale on the model feedforward (`ModelFeedforward`): `inertia·a + viscous·v + coulomb·tanh(v/eps_v)` on the 14 arm joints, clipped to `±tau_max`, from the robot file's `[servo_model]`. 50% halved tracking lag on the robot (12 → 6 ms). `0` = off |
| `arm_gain_scale` | 1.0 | arm stiffness scale |

`--intervene` forces the plain servo (`filter_hz = 0`, `torque_ff = 0`).

## Writing your own smoother or feedforward

Two slots, each named by module path in the deploy file:

```toml
[control.servo]
filter_hz = 6.0                         # > 0: the smoother is on (your factory may read it)
torque_ff = 0.5                         # > 0: the feedforward is on (your factory may read it)
smoother = "servo.py:make_smoother"           # relative to the deploy file,
feedforward = "servo.py:make_feedforward"       # or "pkg.module:factory"
```

```python
def make_smoother(servo, hw, q0):
    """servo: this rollout's ServoShaping (filter_hz, filter_lead, ...);
    hw: the robot (Hardware); q0: (17,) the command to start from."""
    return MySmoother(q0)

class MySmoother:
    lead_s = 0.05          # optional: read the line this far ahead (used when filter_lead)
    def step(self, reference, h):
        """reference (17,) the target line at t + lead_s; h: seconds since the
        last call. -> (q, v, a), each (17,): position, velocity, acceleration."""

def make_feedforward(servo, hw):
    return MyFeedforward(scale=servo.torque_ff)

class MyFeedforward:
    def tau(self, v, a):
        """(17,) commanded velocity and acceleration -> (17,) N·m, added to
        gravity compensation."""
```

The 17 joints are the 14 arm joints (left then right) and the 3 waist slots.

- **The same switches still apply.** Yours replaces the shipped piece only
  while `filter_hz > 0` / `torque_ff > 0`, so setting either to 0 on the
  dashboard still gives the plain straight-line servo for an A/B, and
  `--intervene` is unaffected.
- **It runs in P7**, under elrobot_deploy's Python, inside a 2 ms period, at
  500 Hz: pure numpy, allocation-light, no imports of the IK stack. It is
  imported at launch (a bad path is refused before anything moves).
- **It shapes everything**, including the first homing move and the exit
  ramp — nobody is at the dashboard for those. The factory is called when P7
  starts and again at every rollout's first tick (with that rollout's
  settings and the current command as `q0`), so the smoother state starts
  fresh each rollout.
- **Per launch** (the module); its switches are per rollout.

## What is not pluggable, and why

The walk itself — segment selection, the stall, abandoning a segment at a
takeover, the slew clamp — is fixed. The same code path runs the unattended
first move of the day and the last, every rollout and every intervention;
keeping it one short, well-tested loop is what makes those safe. A different
*shape* of motion goes in the smoother; a different *torque* in the
feedforward.

## Checking a change on recordings

Every rollout records P7's command next to the plain linear walk (`q` vs
`q_lin` in `command.bin`) and the measured joints, so tracking error and the
shaping's effect are in the record. The inspection dashboard's optional
**executor** mode (`docs/deploying_a_policy/rollout_inspection.md`) charts measured against
commanded per joint and can list replays of a rollout through other servos
produced by a simulator you plug in.
