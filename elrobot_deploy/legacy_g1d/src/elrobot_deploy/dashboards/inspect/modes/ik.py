"""Mode `ik`: which IK solver would have reached this rollout's poses.

What it draws is the recording's own sidecar — `ik_bench.npz`, the infra's
solvers benched against it (`producers/ik_bench.py`). `scene.build` reads it through
`overlays.py` and packs one section per name into the transform buffer, so a
method arrives in the viewer by being in a file, and `mode_ik.js` builds a
switch, a swatch and a readout row for each.

## Solving

Opening this mode on a recording nobody has benched SOLVES it: the mode spawns
the bench for the infra's solvers the sidecar is missing
(every name in `elrobot_deploy.runtime.ik.SOLVERS` — `arm_planner`, `arm_qp`,
`waist_planner`, `waist_qp` today — resolved by name in the producer, so a
solver that will not import here shows as unavailable with the reason), merges them
into `ik_bench.npz`, and reloads the rollout so the switches appear.

Two halves that never import each other, meeting at that one file:

  * **this mode** — reads `ik_bench.npz` to say what is already solved, and
    spawns the bench. It imports numpy and no solver.
  * **the bench** — `python -m elrobot_deploy.dashboards.inspect.producers.ik_bench`,
    mujoco and every solver, a minute of CPU per method on a long recording,
    run as a subprocess so the dashboard stays responsive.

**Joint-mode rollouts.** A joint policy's ticks carry no Cartesian target, so
there is nothing to re-solve — except where an operator intervened (PICO
commands are EEF poses). The mode says which: `applicable` is false, with the
reason, when the record has no Cartesian tick at all.

Outputs, all beside the recording:

    <record>/ik_bench.npz                      the solved methods (merged)
    <record>/ik_bench_jobs/<id>.log            this run's stdout
    <record>/ik_bench_jobs/<id>.progress.json  which method of how many

For the user: `docs/deploying_a_policy/rollout_inspection.md`, "ik"; the
solvers: `docs/infra_logic/ik_solvers.md`, "Comparing solvers on recordings".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from elrobot_deploy.dashboards.web.jobs import RUNNER
from elrobot_deploy.dashboards.inspect.overlays import BENCH_NAME
from elrobot_deploy.dashboards.inspect.modes import describe

#: The producer, by module path; run with the dashboard's own interpreter.
PRODUCER = "elrobot_deploy.dashboards.inspect.producers.ik_bench"

#: Where a recording's bench job logs go, relative to the recording.
JOBS_DIR = "ik_bench_jobs"

#: How long `--list-json` may take: it imports every solver module.
LIST_TIMEOUT_S = 90


def default_methods() -> tuple[str, ...]:
    from elrobot_deploy.dashboards.inspect.producers.ik_bench import methods
    return methods()


def jobs_dir(record_dir: Path) -> Path:
    return Path(record_dir) / JOBS_DIR


def solved_methods(record_dir: Path | None) -> list[str]:
    """The methods already in this recording's `ik_bench.npz`, in file order
    (which is the order the viewer colours them in)."""
    if record_dir is None:
        return []
    path = Path(record_dir) / BENCH_NAME
    if not path.exists():
        return []
    try:
        with np.load(path, allow_pickle=False) as z:
            return [str(m) for m in z["methods"]]
    except Exception:                                     # noqa: BLE001
        return []


def applicable(rollout) -> tuple[bool, str]:
    """Has this rollout any Cartesian target to re-solve? `(ok, why not)`."""
    body = rollout.body_targets
    if not len(body) or "pose" not in (body.dtype.names or ()):
        return False, "this record stores no commanded pose (recorded before P6 did)"
    if bool(np.isfinite(body["pose"]).all(axis=(1, 2)).any()):
        return True, ""
    if rollout.action_mode == "joint":
        return False, ("a joint-mode rollout: the policy commanded joint angles, "
                       "so there is no pose for an IK to re-solve, and no "
                       "operator intervened (PICO commands are poses)")
    return False, "no tick of this record carries a Cartesian target"


def _r(a, scale: float, nd: int = 3) -> list:
    a = np.asarray(a, dtype=np.float64) * scale
    return [round(float(v), nd) if np.isfinite(v) else None for v in a]


def strip(loaded, record_dir) -> dict:
    """The timeline strip: the IK residual per commanded tick, one series for
    the configuration that flew and one per method the sidecars carry.

    Per tick `k` that carries a Cartesian target, worst hand of the two:

      * `tcp_mm` — distance between the commanded and the solved TCP, where
        the TCP is the recorded robot's virtual TCP (`point == "virtual TCP"`)
        or its flange when the robot file names none (`point == "flange"`);
      * `rot_deg` — the angle between the commanded and the solved flange
        orientation.

    `t` is the tick's deadline in seconds from the display window's start
    (`timeline.t0`), the axis the main timeline is drawn on. The `deployed`
    series is recomputed from the recorded joints on the trunk they were
    solved in (`ik_bench.recorded_residuals`), so it exists without a bench.
    """
    from elrobot_deploy.dashboards.inspect import overlays
    from elrobot_deploy.dashboards.inspect.scene import tcp_point
    from elrobot_deploy.dashboards.inspect.producers.ik_bench import recorded_residuals

    rollout = loaded.rollout
    facts = rollout.model
    point = "virtual TCP" if np.any(tcp_point(facts)) else "flange"
    ok, why = applicable(rollout)
    if not ok:
        return {"applicable": False, "why": why, "point": point, "t": [],
                "k": [], "series": []}
    ticks, _err, tcp, rot = recorded_residuals(rollout)
    k = np.asarray(ticks["k"], np.int64)
    t = np.asarray(ticks["deadline"], np.float64) - float(loaded.timeline.t0)
    series = [{"name": "deployed", "source": "deployed",
               "tcp_mm": _r(tcp, 1e3), "rot_deg": _r(rot, 180 / np.pi, 2)}]
    ov = overlays.load(record_dir) if record_dir is not None else None
    if ov is not None:
        row = {int(kk): i for i, kk in enumerate(np.asarray(ov["k"], np.int64))}
        idx = np.array([row.get(int(kk), -1) for kk in k], dtype=np.int64)
        have = idx >= 0
        for name in ov.tiers:
            src = ov.meta.get("names", {}).get(name, {}).get("source", "")
            out = {}
            for field, key, scale, nd in (("tcp_err", "tcp_mm", 1e3, 3),
                                          ("rot_err", "rot_deg", 180 / np.pi, 2)):
                vals = np.full(len(k), np.nan)
                arr = ov.get(f"{name}/{field}")
                if arr is not None:
                    vals[have] = np.asarray(arr, np.float64)[idx[have]]
                out[key] = _r(vals, scale, nd)
            series.append({"name": name, "source": src, **out})
    return {"applicable": True, "why": "", "point": point,
            "t": _r(t, 1.0, 4), "k": k.tolist(), "series": series}


class IkMode:
    name = "ik"
    title = "IK solvers"
    blurb = ("Re-solved arms from the sidecars beside this recording: the infra's "
             "IK solvers benched against it, and the whole-body tiers, one ghost "
             "per method. Solves the missing ones when the mode opens, "
             "except the one the rollout flew (its `deployed` line is that).")

    def __init__(self, options=None):
        self.options = options
        self._lock = threading.Lock()
        self._registry: dict | None = None          # `--list-json`, cached
        self._state = None
        self._strip: tuple | None = None           # (key, answer)

    def _python(self) -> str:
        return getattr(self.options, "python", None) or sys.executable

    # --- what can be solved ------------------------------------------------

    def registry(self, refresh: bool = False) -> dict:
        """The infra's solvers and whether this install has each. Cached for
        the life of the process; `refresh` asks again."""
        with self._lock:
            if self._registry is not None and not refresh:
                return self._registry
        out: dict = {"methods": [], "error": ""}
        try:
            proc = subprocess.run([self._python(), "-m", PRODUCER, "--list-json"],
                                  capture_output=True, text=True,
                                  timeout=LIST_TIMEOUT_S, cwd=os.getcwd())
            if proc.returncode != 0:
                out["error"] = (proc.stderr or proc.stdout or "").strip()[-400:]
            else:
                out.update(json.loads(proc.stdout.strip().splitlines()[-1]))
        except subprocess.TimeoutExpired:
            out["error"] = f"the solvers did not list in {LIST_TIMEOUT_S}s"
        except (OSError, ValueError, IndexError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._registry = out
        return out

    def methods(self, refresh: bool = False) -> dict:
        """The picker's whole state: the registry, what this recording already
        has, and which defaults are therefore worth solving."""
        record_dir = self._state.record_dir if self._state else None
        cur = self._state.current if self._state else None
        ok, why = applicable(cur.rollout) if cur is not None else (False, "")
        reg = self.registry(refresh)
        solved = solved_methods(record_dir)
        defaults = default_methods()
        rows = [{**m, "solved": m["name"] in solved,
                 "default": m["name"] in defaults}
                for m in reg.get("methods", [])]
        known = {m["name"] for m in rows}
        # A method in the sidecar that the registry does not have (solved by
        # an older build, or a solver since renamed) is still drawn, so the
        # picker says it is there rather than pretending it is not.
        for name in solved:
            if name not in known:
                rows.append({"name": name, "mode": "", "describe": "",
                             "available": False, "why": "not an infra solver in "
                             "this install; it came from the file",
                             "solved": True, "default": False})
        # The solver this rollout flew is already on screen as `deployed`
        # (its recorded joints), so re-solving it is not needed to compare.
        deployed = ""
        manifest = getattr(cur.rollout, "manifest", None) if cur is not None else None
        if manifest and getattr(cur.rollout, "action_mode", "eef") != "joint":
            from elrobot_deploy.dashboards.inspect import discover
            deployed = str(discover.specifics(manifest).get("ik") or "")
        missing = [n for n in defaults if ok and n not in solved and n != deployed
                   and any(m["name"] == n and m["available"] for m in rows)]
        return {
            "methods": rows, "defaults": list(defaults), "solved": solved,
            "deployed": deployed,
            "missing": missing, "error": reg.get("error", ""),
            "applicable": ok, "why": why,
            "sidecar": str(Path(record_dir) / BENCH_NAME) if record_dir else "",
            "record": str(record_dir) if record_dir else "",
            "busy": RUNNER.busy(),
        }

    def strip(self) -> dict:
        """`strip(...)` for the loaded record, cached until the record or its
        sidecars change (a solve lands, a reload)."""
        cur = self._state.current if self._state else None
        if cur is None:
            raise RuntimeError("nothing loaded")
        record_dir = self._state.record_dir
        from elrobot_deploy.dashboards.inspect import overlays
        stamps = []
        for name in (overlays.BENCH_NAME,):
            p = Path(record_dir) / name if record_dir else None
            stamps.append(p.stat().st_mtime_ns if p is not None and p.exists() else 0)
        key = (id(cur), str(record_dir), tuple(stamps))
        with self._lock:
            if self._strip is not None and self._strip[0] == key:
                return self._strip[1]
        out = strip(cur, record_dir)
        with self._lock:
            self._strip = (key, out)
        return out

    # --- solving -----------------------------------------------------------

    def solve(self, req: dict) -> dict:
        """Spawn the bench for `methods` on the loaded recording."""
        if self._state is None or self._state.current is None:
            raise RuntimeError("load a rollout first")
        ok, why = applicable(self._state.current.rollout)
        if not ok:
            raise RuntimeError(why)
        if RUNNER.busy():
            raise RuntimeError("a job is already running; this box takes one at a time")
        names = [str(n) for n in (req.get("methods") or []) if str(n).strip()]
        if not names:
            raise RuntimeError("no methods asked for")
        record_dir = Path(self._state.record_dir)
        job = RUNNER.submit(
            kind=self.name,
            label=f"{len(names)} method(s): {', '.join(names)}",
            cmd=[self._python(), "-m", PRODUCER, str(record_dir),
                 "--methods", ",".join(names)],
            cwd=os.getcwd(), jobs_dir=jobs_dir(record_dir),
            meta={"record": record_dir.name, "methods": names})
        return {"job": job.as_json()}

    # --- mounting ----------------------------------------------------------

    def mount(self, app, state) -> None:
        self._state = state

        @app.get("/api/ik/methods")
        def methods(req):
            req.json(self.methods(refresh=req.arg("refresh", "") == "1"))

        @app.get("/api/ik/strip")
        def ik_strip(req):
            try:
                req.json(self.strip())
            except RuntimeError as exc:
                req.error(exc, 409)

        @app.get("/api/ik/jobs")
        def ik_jobs(req):
            req.json({"jobs": [j.as_json() for j in RUNNER.jobs(kind=self.name)],
                      "busy": RUNNER.busy()})

        @app.post("/api/ik/solve")
        def solve(req):
            try:
                req.json(self.solve(req.body()))
            except RuntimeError as exc:
                req.error(exc, 409)

        @app.post("/api/ik/cancel")
        def cancel(req):
            req.json({"cancelled": RUNNER.cancel(req.body()["job"])})

    def describe(self) -> dict:
        return describe(self)


__all__ = ["IkMode", "JOBS_DIR", "PRODUCER", "applicable", "default_methods",
           "jobs_dir", "solved_methods", "strip"]
