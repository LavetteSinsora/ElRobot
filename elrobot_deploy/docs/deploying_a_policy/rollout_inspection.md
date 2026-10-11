# Rollout inspection

    g1d_deploy inspect [ROOT-OR-RECORD] [--port 8770]

A local web app over recorded rollouts: scrub a rollout's timeline with every
camera, the robot in 3-D (measured, commanded pose, commanded joints), and
compare it against re-solves by other IK solvers or replays through other
checkpoints. It never talks to a robot. With no argument it opens the machine
file's `[paths] records`. On the deploy box it binds loopback: reach it with
`ssh -L 8770:localhost:8770 <box>`.

- **Read first:** `docs/overview.md`.
- **Code:** `src/g1d_deploy/cli/inspect.py`, `src/g1d_deploy/dashboards/inspect/`
  (`options.py`, `discover.py`, `record_robot.py`, `loader.py`, `scene.py`,
  `server.py`, `modes/`, `producers/ik_bench.py`, `static/`); shared with the
  live dashboard: `src/g1d_deploy/dashboards/web/`.
- **Related:** `docs/infra_logic/ik_solvers.md`, `docs/deploying_a_policy/policy_contract.md`,
  `docs/infra_logic/500hz_execution.md`.

## What it opens

A **record** is any directory holding a `manifest.json` that P9 wrote. The
root is searched recursively; nothing about names or layout is assumed. The
dropdown groups records by their path under the root:

    <root>/pick_cup/20260926-101010       -> group pick_cup
    <root>/20260925-090000                -> no group

Pointing it at one record opens just that. A record's own results
(`policy_replays/`, `executor_replays/`, `ik_bench.npz`) live inside it and are
not listed as records. `?rec=<path under root>&load=1` in the URL loads one.

**The robot is the record's**: joints, the hand's travel, the TCP and the
URDF come from the robot file text in the manifest, not from this machine. A
record whose robot names a custom URDF is reported as unsupported.

## The view

- **Record chip**, beside Load: checkpoint and step, adapter class, and that
  rollout's settings (sync/async, replan period, rate, IK, sample steps).
  Click it for the rest: prompt, rollout generation, entry, full checkpoint
  path, anchor camera, servo, robot file chain, code revisions.
- **Status line** after loading: robot, actions, duration, replans,
  inference latency, and warnings (stalls, robot or TCP assumed).
- **Timeline** with the playhead; every camera pane follows it.
- **3-D scene**, three sources:
  - **measured** — the robot at its measured joints;
  - **commanded pose** — the blended flange target P6 gave the IK, drawn as a
    gripper (greyed for joint-mode rollouts except where a PICO operator
    intervened);
  - **commanded joints** — what P6 published, posed on the trunk the IK solved
    in.

## Modes

Pick one from **compare**. Each also draws the **mode strip**, a second
timeline under the scrubber with that mode's per-tick metric, sharing the
playhead.

### ik

Re-solves the recording's commanded poses with the shipped solvers and draws
each as a ghost, with its residual on the strip (mm at the TCP, or degrees of
orientation; one line for what flew, `deployed`, and one per solver switched
on). Opening the mode solves the four shipped solvers the recording lacks,
once, skipping the one the rollout flew; **Solve** re-runs one. Solving is a
CPU subprocess:

    python -m g1d_deploy.dashboards.inspect.producers.ik_bench RECORD --methods arm_planner,arm_qp

which re-solves every tick the way P6 did (per chunk, on the recorded trunk,
TCP included) and merges results into `RECORD/ik_bench.npz`. A solver of your
own (`--methods pkg.mod:factory`) is drawn too once benched
(`docs/infra_logic/ik_solvers.md`).

### policy checkpoint (`replay`)

Re-runs the rollout's own observations — the frames and state the adapter saw
at each replan, rebuilt from the record — through an adapter class and a
checkpoint, via the ordinary `Policy` interface, and draws each result as a
ghost against what flew; the strip shows each replay's distance from the
recorded actions (mm, worst hand). **New replay** opens a form defaulting to
the rollout's own:

| field | meaning |
|---|---|
| deploy file | the `deploy.toml` that names the policy adapter and the Python it runs under; default: the one the rollout ran from. **Browse** opens the desktop's file dialog to pick another |
| checkpoint folder | the folder the policy loads from; default: the rollout's. **Browse** opens the desktop's folder dialog, starting at the recorded run. Viewed over an SSH tunnel (no display on the box), Browse shows a folder list in the page instead |
| prompt | checked by the adapter first; **Check prompt** asks without replaying (GPU hidden, only `__init__`, so it is as fast as the constructor) |
| sample steps | handed to `on_settings` |
| repeats | independent replays: their spread is the sampler's noise |

This is the check that an adapter's actions **mean** the right thing — frame,
units, arm order, TCP, gripper — on real observations, without a robot: a
replay whose ghost tracks the recorded motion is speaking the contract the
way the recorded policy did. The replay runs under the interpreter the
rollout recorded (or the one given), which needs the `replay` extra. It is
**refused while a live deployment runs** on the box (it would compete for the
GPU). By hand:

    g1d_deploy replay-policy --record DIR [--entry E] [--checkpoint STEP | --run RUN --step N]
        [--prompt TEXT] [--sample-steps N] [--seed N]

Results land in `<record>/policy_replays/<id>/`.

### executor (off by default)

Mounted with `--mode executor` or `--executor-producer my_pkg.simrig`. Lists
replays of this rollout through other 500 Hz servos (records under
`<record>/executor_replays/`) and charts measured against commanded per joint;
making new ones needs a producer module you provide, run as
`python -m my_pkg.simrig --source RECORD --variants a,b --out RECORD/executor_replays`.

## Writing a mode

A mode is a class, mounted by module path or file (relative to the working
directory):

    g1d_deploy inspect --mode my_pkg.my_mode:MyMode
    g1d_deploy inspect --mode examples/inspect_mode/mode.py:LatencyMode

or in the machine file:

```toml
[inspect]
modes = ["my_pkg.my_mode:MyMode"]
runs = ["/data/runs"]
port = 8770
```

```python
from pathlib import Path
from g1d_deploy.dashboards.inspect.modes import describe

class MyMode:
    name = "mine"                                  # ?mode=mine, and mode_mine.js
    title = "my comparison"
    blurb = "tooltip"
    static_dir = Path(__file__).parent / "static"  # holds mode_mine.js

    def __init__(self, options):                   # dashboards.inspect.options.Options
        self.options = options

    def mount(self, app, state):
        state.on_load(lambda loaded, record_dir: ...)

        @app.get("/api/mine/thing")
        def thing(req):
            req.json({...})

    def describe(self):
        return describe(self)
```

`mode_mine.js` exports `activate(ctx)` and `deactivate()`; `ctx.hooks` gives
the viewer, `onLoaded` / `onFrame` callbacks, `setOverlay` for ghosts, `strip`
for the mode strip (`strip.show({title, unit, series: [{label, color, t, y}]})`),
and the loaded record. A mode that fails to import is reported and left out.
Anything needing a GPU or heavy CPU runs as a subprocess
(`src/g1d_deploy/dashboards/web/jobs.py`'s `RUNNER`), never in the server.
`examples/inspect_mode/` is a working one.
