"""Mode `executor`: how another 500 Hz executor drove the joints.

A REPLAY re-runs one rollout's recorded policy output — the same chunks at the
same instants — through a different servo arm: the linear line, the critically
damped tracker with look-ahead (`runtime/execution/servo_filter.py`), a torque feedforward,
higher gains. On the robot that is a session at the robot; without one it is
the same loops with a fitted plant as the arm. Either way the result is an
ordinary P9 record whose `deploy_config.labels` name the rollout it re-ran
(`source_name`, that record's directory name) and the `variant`.

**What this mode reads is that contract, nothing else.** Every record that
names the loaded rollout as its source is a replay of it, one switch each:

  * any record under `<record>/executor_replays/` (where a producer is asked
    to write), and
  * any record anywhere under the discovery root whose `labels.source_name`
    is the loaded record's name.

The chart shows one joint's measured angle against the command each executor
sent, with the playhead mapped onto the replays' common clock (`series.t_zero`).

**The producer is plugged in, never imported.** `--executor-producer PKG.MOD`
(or `[inspect] executor_producer` in the machine file) names a module the
dashboard runs as

    python -m PKG.MOD --list-variants                 -> JSON [{name, note}, ...]
    python -m PKG.MOD --source RECORD --variants a,b --out RECORD/executor_replays

With none configured the mode still lists and charts the replays that exist,
and says how to plug one in.

For the user: `docs/deploying_a_policy/rollout_inspection.md`, "executor (off
by default)"; the servo it compares: `docs/infra_logic/500hz_execution.md`,
"Checking a change on recordings".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from elrobot_deploy.dashboards.web.jobs import RUNNER
from elrobot_deploy.dashboards.inspect import discover
from elrobot_deploy.dashboards.inspect.modes import describe
from elrobot_deploy.dashboards.inspect.modes.executor import series as _series

#: Where a producer is asked to write a rollout's replays, relative to it.
REPLAYS_DIR = "executor_replays"

#: The sentence the panel shows when no producer is configured.
NO_PRODUCER = ("no executor producer configured: pass --executor-producer PKG.MOD "
               "or set [inspect] executor_producer in the machine file "
               "(docs/deploying_a_policy/rollout_inspection.md); existing replays are "
               "still listed")


def replays_of(record_dir: Path, root: Path | None) -> list[Path]:
    """Every replay record of the rollout at `record_dir`, by name."""
    record_dir = Path(record_dir)
    name = record_dir.name
    out: dict[str, Path] = {}
    own = record_dir / REPLAYS_DIR
    if own.is_dir():
        for rec in discover.scan(own):
            if rec.loadable:
                out[str(rec.path)] = rec.path
    if root is not None and Path(root).is_dir():
        for rec in discover.scan(root):
            if rec.loadable and rec.path != record_dir \
                    and rec.details.get("labels", {}).get("source_name") == name:
                out[str(rec.path)] = rec.path
    return sorted(out.values(), key=lambda p: p.name)


class ExecutorMode:
    name = "executor"
    title = "executors"
    blurb = ("Replays of this rollout through other 500 Hz executors — on the robot, "
             "or through a fitted plant — as joint traces against the command each sent.")

    def __init__(self, options=None):
        self.options = options
        self.producer = str(getattr(options, "executor_producer", "") or "")
        self._lock = threading.Lock()
        self._runs: dict[str, _series.Run] = {}
        self._offset: float | None = None
        self._variants: list[dict] | None = None
        self._state = None

    def _root(self) -> Path | None:
        return getattr(self._state, "root", None) if self._state else None

    def _python(self) -> str:
        return getattr(self.options, "python", None) or sys.executable

    # --- lifecycle ---------------------------------------------------------

    def _on_load(self, loaded, record_dir: Path) -> None:
        with self._lock:
            self._runs.clear()
            # Where the loaded rollout's own chunk 1 starts executing, relative
            # to the display window: the playhead's time on the replays' clock.
            try:
                rec = _series.load_record(record_dir)
                self._offset = _series.t_zero(rec) - float(loaded.timeline.t0)
            except Exception:                   # noqa: BLE001 — a rollout with no targets
                self._offset = None

    def _run(self, path: Path) -> _series.Run:
        key = str(path)
        with self._lock:
            run = self._runs.get(key)
        if run is None:
            run = _series.load(path)
            with self._lock:
                self._runs[key] = run
        return run

    def variants(self) -> list[dict]:
        if not self.producer:
            return []
        if self._variants is None:
            try:
                out = subprocess.run(
                    [self._python(), "-m", self.producer, "--list-variants"],
                    capture_output=True, text=True, timeout=60, check=True,
                    cwd=os.getcwd()).stdout
                self._variants = json.loads(out.strip().splitlines()[-1])
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                self._variants = []
        return self._variants

    # --- the answers -------------------------------------------------------

    def replays(self) -> dict:
        cur = self._state.current
        base = {"joints": _series.JOINT_NAMES, "producer": self.producer,
                "note": "" if self.producer else NO_PRODUCER}
        if cur is None:
            return {**base, "name": "", "replays": []}
        rows = []
        for path in replays_of(self._state.record_dir, self._root()):
            try:
                rows.append(self._run(path).describe())
            except Exception as exc:            # noqa: BLE001 — list it, broken
                rows.append({"name": path.name, "dir": str(path),
                             "error": f"{type(exc).__name__}: {exc}"})
        return {**base, "name": self._state.record_dir.name, "replays": rows,
                "variants": self.variants(), "offset_s": self._offset,
                "busy": RUNNER.busy()}

    def series(self, name: str, joint: int) -> dict:
        if self._state.current is None:
            raise RuntimeError("nothing loaded")
        for path in replays_of(self._state.record_dir, self._root()):
            if path.name == name:
                return self._run(path).series(joint)
        raise KeyError(name)

    def strip(self, name: str) -> dict:
        """The timeline strip for one run: servo tracking error per tick
        (`series.Run.tracking`), its times moved onto the loaded rollout's
        display axis (`t + offset_s`). `name` is a replay's directory name, or
        empty for the loaded rollout itself."""
        if self._state.current is None:
            raise RuntimeError("nothing loaded")
        if not name:
            path = Path(self._state.record_dir)
        else:
            path = next((p for p in replays_of(self._state.record_dir, self._root())
                         if p.name == name), None)
            if path is None:
                raise KeyError(name)
        try:
            out = self._run(path).tracking()
        except Exception as exc:                # noqa: BLE001 — say why, not 500
            out = {"t": [], "err_deg": [], "has_command": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        off = self._offset
        out["t"] = ([] if off is None else
                    [None if v is None else round(v + off, 4) for v in out["t"]])
        out["name"] = name
        out["unit"] = "deg"
        out["metric"] = "mean over the 14 arm joints of |measured - commanded|"
        return out

    def replay(self, req: dict) -> dict:
        if self._state.current is None:
            raise RuntimeError("load a rollout first")
        if not self.producer:
            raise RuntimeError(NO_PRODUCER)
        if RUNNER.busy():
            raise RuntimeError("a job is already running")
        variants = [str(v) for v in (req.get("variants") or []) if v]
        if not variants:
            raise ValueError("no variants named")
        record_dir = Path(self._state.record_dir)
        job = RUNNER.submit(
            kind=self.name, label=f"{' '.join(variants)}",
            cmd=[self._python(), "-m", self.producer, "--source", str(record_dir),
                 "--variants", ",".join(variants),
                 "--out", str(record_dir / REPLAYS_DIR)],
            cwd=os.getcwd(), jobs_dir=record_dir / REPLAYS_DIR / "_jobs",
            progress_arg=None,
            meta={"record": record_dir.name, "variants": variants})
        return {"job": job.as_json()}

    # --- mounting ----------------------------------------------------------

    def mount(self, app, state) -> None:
        self._state = state
        state.on_load(self._on_load)

        @app.get("/api/executor/replays")
        def replays(req):
            req.json(self.replays())

        @app.get("/api/executor/series")
        def series(req):
            req.json(self.series(req.arg("run", ""), int(req.arg("joint", "3"))))

        @app.get("/api/executor/strip")
        def strip(req):
            try:
                req.json(self.strip(req.arg("run", "")))
            except KeyError as exc:
                req.error(f"no replay {exc}", 404)
            except RuntimeError as exc:
                req.error(exc, 409)

        @app.get("/api/executor/jobs")
        def jobs(req):
            req.json({"jobs": [j.as_json() for j in RUNNER.jobs(kind=self.name)],
                      "busy": RUNNER.busy()})

        @app.post("/api/executor/replay")
        def replay(req):
            try:
                req.json(self.replay(req.body()))
            except RuntimeError as exc:
                req.error(exc, 409)

        @app.post("/api/executor/cancel")
        def cancel(req):
            req.json({"cancelled": RUNNER.cancel(req.body()["job"])})

    def describe(self) -> dict:
        return describe(self)


__all__ = ["ExecutorMode", "NO_PRODUCER", "REPLAYS_DIR", "replays_of"]
