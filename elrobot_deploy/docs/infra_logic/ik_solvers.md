# IK solvers

For an eef adapter, P6 turns each tick's flange targets into 14 arm joint
angles (and, in waist mode, a trunk). Four solvers ship; a solver of your own
plugs in by module path.

- **Read first:** `docs/overview.md` (where P6 sits).
- **Code:** `src/elrobot_deploy/runtime/ik/` — `__init__.py` (the registry,
  `SOLVERS`, `build`, the interface), `arm_qp.py`, `arm_planner.py`,
  `waist_planner.py`, `waist_qp.py`, `collision.py`, `velocity_caps.py`;
  the caller is `src/elrobot_deploy/runtime/processes/p6_target.py`.
- **Related:** `docs/deploying_a_policy/deploy_toml.md` (`[control] ik`), `docs/setup/robot_toml.md`
  (`[end_effector] tcp`), `docs/deploying_a_policy/rollout_inspection.md` (the IK mode, where
  solvers are compared on recordings), `examples/custom_modules/ik_solver.py`.

## Where the IK sits

Per tick `k`, P6 (`p6_target.py::Target`):

1. **Blends** the chunks covering `k` into one flange target per arm
   (`docs/infra_logic/chunk_blending.md`).
2. **Solves**: `solver.solve(pose, waist_ik, seed, budget=dt)`.
   - `pose` — `(2, 7)` flange targets, base frame.
   - `waist_ik` — the **measured** trunk (yaw, bow, 0), because the arm hangs
     off the trunk as it is, not as commanded. On the reference robot the bow rests on
     a stop 0.039 rad forward of zero; solving against the commanded zero put
     every flange 8–11 mm off.
   - `seed` — the previous tick's *published* target, never the measurement
     (a measured seed would feed tracking error back into the next target).
   - `budget` — one tick.
3. **Clamps** each joint's step against the seed to the robot's joint speed
   limit over the tick span.
4. **Writes** the target with its deadline, the residual `err` (metres, at the
   TCP or flange), the blended `pose` and `waist_ik`, so every record can
   re-check every solve offline.

A solve that reports `ok = False` writes nothing: the tick becomes a hole that
P7 skips. P6 solves up to `solve_lead_ticks` (5) ahead of execution, so each
solve uses a trunk measurement at most that old.

**The TCP.** With `[end_effector] tcp` in the robot file (every shipped Dex-1
profile), every solver targets the TCP: the flange target is composed with the
TCP, the QP's task frame is a site at the TCP, and residuals are measured
there. A reachable target lands the same joints either way; an unreachable
one is missed at the fingertips as little as possible, rather than at the
wrist. Without a `tcp`, all of it is at the flange.

**Modes.** A solver's *mode* is a property of the solver:

- **arm** — solves the 14 arm joints on the measured trunk; the waist is
  commanded upright (0) and never moved.
- **waist** — also plans the trunk (yaw and bow) per chunk and commands it,
  for reach the arms alone lack. **Sync only**: planning a chunk's trunk needs
  one chunk per tick, and async overlaps them.

## The four shipped solvers

| `ik` | mode | in one line | when |
|---|---|---|---|
| `arm_planner` | arm | per chunk, plans which elbow branch to take and escapes joint-limit traps | **default**; `ik = "arm"` |
| `arm_qp` | arm | one QP per tick from the last answer | reproducing old runs, ruling the planner out, and `--intervene` (the only one it runs) |
| `waist_planner` | waist | a slewed, settled, lag-aware trunk plan + the arm planner | a task that needs the trunk's reach; `ik = "waist"` |
| `waist_qp` | waist | the whole-body QP's trunk tier | comparison |

### `arm_qp` — the per-tick QP (`arm_qp.py::DualArmSolver`)

A mink differential-IK QP: two frame tasks (the targets), a posture task
toward the seed, joint position and velocity limits, and a collision limit
between the arms and the torso. Per tick it runs QP iterates from the seed
until the worst position error is under `ik_early_stop` (1e-4 m) or
`ik_max_iters` (20) — about 3.7 iterates in practice. The trunk is placed at
`waist_ik` before the solve and held there.

Its weakness: it answers one tick from the last. When a target needs another
elbow *branch* (the arm folded the other way), the QP slides onto a joint
stop and misses, and nothing ever swings it across.

### `arm_planner` — the streamed branch planner (`arm_planner.py`)

Looks at the chunk instead of the tick:

1. **Families.** Four candidate arms are chained through the chunk's knots
   (every few poses and the last): from the seed, from the joint-range
   midpoint, and from the seed swung ±0.6 rad along each elbow's self-motion.
   One (knot, family) solve runs per tick, so the plan streams in as the chunk
   plays.
2. **A DP over the knots** picks a branch per knot: TCP miss, plus travel,
   plus joint-limit margin, plus a penalty for winding toward a shoulder-roll
   stop, over switches the arm can make within a knot span.
3. **Per tick:** a few QP iterates pulled toward the planned posture, then a
   plain polish from there (the pull escapes the trap, the polish removes its
   bias).
4. **Escape.** When another branch clearly beats the current one at an
   upcoming knot, reaches it, and the arm is already missing, the tick pulls
   hard toward that posture at a capped joint speed.

Every solve carries the same motion caps inside the QP (`velocity_caps.py`:
a per-joint speed box and a cap on elbow self-motion) and a collision limit
that does not trust MuJoCo's occasional phantom-zero distances
(`collision.py`), so what P6 publishes is what the QP checked. On an offline
replay of 32 recorded rollouts it roughly halved the per-tick QP's mean TCP
error (3.99 → 1.94 mm) and cut ticks inside the body from 93 to 11. Those
numbers predate TCP-driven solving: re-bench on your recordings.

### `waist_planner` — trunk plan + arm planner (`waist_planner.py`)

1. **Trunk plan** per chunk: the whole-body solver's trunk passes, with the
   bow floored at its physical stop and a pull back toward upright, so the
   trunk bends only while the chunk needs it.
2. **Slewed** against the last commanded trunk (0.2 rad/s bending, 0.1
   unbending, 0.3 yaw), and **settled**: the chunk's last 6 ticks hold the
   trunk, so it is not mid-swing at a seam.
3. **Lag-aware:** the arm is planned on the trunk the servo will actually
   reach (a fitted model: each tick the trunk moves 0.7 of the way to its
   command, at most 1 rad/s), not the one commanded.

On one set of recordings every waist mode was worse than arm mode (the bow
crept forward): turn it on for a task that needs the reach.

### `waist_qp` — the whole-body QP (`waist_qp.py`)

The original whole-body solver restricted to its trunk tier: a chunk-at-a-time
optimisation over arm and trunk with a banded re-timing pass. Kept for
comparison.

## Writing your own

A solver is a factory, named by module path:

```toml
[control]
ik = "my_ik.py:my_solver"              # a file, relative to the deploy file
# ik = "my_pkg.my_ik:my_solver"        # or an importable module; --ik works too
```

```python
def my_solver(cfg, *, urdf=None, qdot_max=None, tcp=None):
    """cfg: the DeployConfig. urdf: the robot's model. qdot_max: P6's joint
    speed clamp (rad/s), so you can hold it inside your optimisation. tcp: the
    robot file's (7,) flange->TCP pose, or None. Omit `tcp` from the signature
    and it is not passed."""
    return MySolver(...)

my_solver.mode = "arm"                  # REQUIRED: "arm" or "waist"


class MySolver:
    def solve(self, pose, waist, seed, budget):
        """pose (2, 7) flange targets, base frame, [x y z qw qx qy qz];
        waist (3,) the trunk to solve in (measured, or measured + plan);
        seed (14,) the previous published arm target; budget: seconds.
        -> (q_arm (14,), err_m, ok). ok=False: write nothing for this tick."""

    # Optional: plan per chunk.
    plans_chunks = True
    def plan_chunk(self, poses, waist_meas, seed, *, budget):
        """poses (K, 2, 7) the chunk's blended targets from its first executed
        tick; waist_meas (3,); seed (14,). -> (K, 3) trunk commands, or None
        (plan nothing; the chunk runs on the measured trunk)."""
```

- **The mode is the solver's**, and the run is held to it: a `"waist"`
  factory is refused with `sync = false`, and a factory without `.mode` is
  refused at launch.
- **It runs in P6, under elrobot_deploy's own Python**, not the adapter's: the
  module must import there (MuJoCo and mink are available). It is imported at
  launch, so a typo fails before anything moves.
- **Budget.** `solve` has one tick (33 ms at 30 Hz) and should take a few
  ms: P6 solves ticks one after another. `plan_chunk` runs once when a chunk
  lands.
- **Joint order** of `q_arm` and `seed`: left 7 then right 7, the order in
  `docs/deploying_a_policy/policy_contract.md`. `err_m` is recorded as the tick's residual.
- **Fixed per launch.** The dashboard can switch between it and the shipped
  solvers between rollouts, but cannot name another module.

`examples/custom_modules/ik_solver.py` is a working one (it wraps `arm_qp`).

## Comparing solvers on recordings

The rollout inspection dashboard's **ik** mode re-solves a recording's
commanded poses with each shipped solver and draws each as a ghost, with its
residual on a strip under the timeline (`docs/deploying_a_policy/rollout_inspection.md`). A
solver of your own is benched from the command line, then shows up there too:

    python -m elrobot_deploy.dashboards.inspect.producers.ik_bench RECORD_DIR --methods my_pkg.my_ik:my_solver

Each tick is solved the way P6 solved it (chunk by chunk, on the trunk P6
recorded, seeded with the previous solution, TCP included) and scored at the
recorded robot's TCP; results merge into `RECORD_DIR/ik_bench.npz`.
