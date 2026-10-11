"""An IK solver of your own: `ik = "ik_solver.py:timed_arm_qp"`.

    g1d_deploy check-policy examples/custom_modules/deploy.toml
    python -m g1d_deploy.dashboards.inspect.producers.ik_bench RECORD \\
        --methods /abs/path/to/examples/custom_modules/ik_solver.py:timed_arm_qp

The smallest complete solver: it wraps the shipped per-tick QP and records how
long each solve took and how far it missed, which is the skeleton any research
solver starts from. What P6 asks of it is the whole interface
(`docs/infra_logic/ik_solvers.md`, "Writing your own"):

* a FACTORY `f(cfg, *, urdf, qdot_max, tcp)` carrying `f.mode` ("arm" here:
  the trunk is read, never commanded);
* an object with `solve(pose (2, 7), waist (3,), seed (14,), budget) ->
  (q_arm (14,), err_m, ok)`;
* optionally `plans_chunks = True` and `plan_chunk(...)` (not used here).

It runs in P6, under g1d_deploy's own Python, so it may import MuJoCo and
mink; it is imported at launch, so a typo fails before anything moves.
"""

from __future__ import annotations

import time

import numpy as np


class TimedSolver:
    """Delegates to `inner`, keeping per-solve timing and residuals."""

    plans_chunks = False

    def __init__(self, inner):
        self.inner = inner
        self.solve_ms: list[float] = []
        self.err_m: list[float] = []

    def solve(self, pose, waist, seed, budget):
        t0 = time.perf_counter()
        q_arm, err, ok = self.inner.solve(pose, waist, seed, budget=budget)
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.err_m.append(float(err))
        if len(self.solve_ms) % 300 == 0:           # every ~10 s at 30 Hz
            ms = np.asarray(self.solve_ms[-300:])
            print(f"[timed_arm_qp] p50 {np.median(ms):.1f} ms, "
                  f"max {ms.max():.1f} ms, worst miss "
                  f"{max(self.err_m[-300:]) * 1e3:.1f} mm", flush=True)
        return q_arm, err, ok


def timed_arm_qp(cfg, *, urdf=None, qdot_max=None, tcp=None):
    """The factory P6 calls (`g1d_deploy.runtime.ik.build`)."""
    from g1d_deploy.runtime.ik.arm_qp import arm_qp    # MuJoCo: imported here
    return TimedSolver(arm_qp(cfg, urdf=urdf, qdot_max=qdot_max, tcp=tcp))


timed_arm_qp.mode = "arm"
