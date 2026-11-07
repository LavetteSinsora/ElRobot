"""The ElRobot counterpart of g1d's inspector loader.

Reuse the recording's own dtypes and JPEG offsets. Browser charts are sampled; the playhead
reads full-resolution state and camera indices. No camera blobs are loaded into memory.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from threading import RLock

import numpy as np

from elrobot_deploy.dashboard.robot import describe
from elrobot_deploy.record import load

# These are elrobot-record/1's values, not imports from a potentially newer live schema.
SESSIONS = dict(enumerate(("INIT", "CHECK", "HOME", "WARMUP", "READY", "ACTIVE",
                          "EXITING", "STOPPED")))
BUSES = dict(enumerate(("STARTING", "RUNNING", "HOLDING", "DONE", "DRY", "RECOVERING")))


def clean(value):
    """JSON uses null for missing sensor values, never non-standard NaN/Infinity."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def previous(times, t):
    i = int(np.searchsorted(times, t, side="right")) - 1
    return i if i >= 0 else None


class Rollout:
    def __init__(self, path: Path, episode: Path | None = None):
        self.record = r = load(path)
        self.manifest = m = r.manifest
        if m.get("format") != "elrobot-record/1":
            raise ValueError("This inspector opens elrobot-record/1 recordings")
        starts = [float(r.state["t"][0])] if len(r.state) else []
        starts += [float(c.index["t_capture"][0]) for c in r.cameras.values() if len(c.index)]
        self.t0 = float(m.get("rollout", {}).get("t_start", min(starts, default=0)))
        ends = [self.t0]
        for rows, key in ((r.state, "t"), (r.targets, "deadline"), (r.session, "t_observed")):
            if len(rows):
                ends.append(float(rows[key][-1]))
        ends += [float(c.index["t_capture"][-1]) for c in r.cameras.values() if len(c.index)]
        self.duration = max(0., max(ends) - self.t0)
        self.episode = {}
        self.actions = []
        self.warnings = []
        if not m.get("complete"):
            self.warnings.append("Recording is incomplete or still being written. Reload to refresh.")
        self.robot, self.meshes = None, []
        urdf = Path(m.get("config", {}).get("urdf", ""))
        if urdf.is_file():
            try:
                self.robot, self.meshes = describe(urdf)
                self.warnings.append("3D uses the recorded URDF path on this machine; this record "
                                     "does not embed a model snapshot.")
            except (OSError, ValueError) as exc:
                self.warnings.append(f"3D model unavailable: {exc}")
        else:
            self.warnings.append("Recorded URDF is unavailable here; cameras and joint data still work.")
        if episode is not None:
            self._episode(episode)
        self.rotation_errors = self._rotation_errors(urdf)

    def _episode(self, path):
        self.episode = json.loads((path / "episode.json").read_text())
        times = {}
        db = path / "timeline.sqlite"
        if db.is_file():
            try:
                with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
                    times = dict(conn.execute("SELECT tick, t FROM timeline"))
            except sqlite3.Error:
                self.warnings.append("Episode timeline unavailable; action times use the 30 Hz clock.")
        # start() can create the episode before a camera ever delivers its opening frame.
        origin = float((self.episode.get("clock") or {}).get("t0_monotonic", self.t0)) - self.t0
        for f in sorted((path / "actions").glob("*/execution_outcome.json")):
            a = json.loads(f.read_text())
            a["start"] = origin + times.get(a["tick_start"], a["tick_start"] / 30.)
            a["end"] = origin + times.get(a["tick_end"], a["tick_end"] / 30.)
            request = f.with_name("action.json")
            a["request"] = json.loads(request.read_text()) if request.exists() else None
            self.actions.append(a)

    def _rotation_errors(self, urdf):
        """Compute orientation residual independently; old summaries picked max POSITION error."""
        out = np.full(len(self.record.targets), np.nan)
        if self.robot is None or not len(out):
            return out
        try:
            from elrobot_deploy.robot.elrobot_model import ElRobotModel, pose_to_mat
            model = ElRobotModel(urdf)
            for i, target in enumerate(self.record.targets):
                if not np.all(np.isfinite(target["pose"])):
                    continue
                reached = model.fk(target["q"][:7])
                goal = pose_to_mat(target["pose"])
                c = (np.trace(reached[:3, :3].T @ goal[:3, :3]) - 1) / 2
                out[i] = np.degrees(np.arccos(np.clip(c, -1, 1)))
        except (ImportError, OSError, ValueError) as exc:
            self.warnings.append(f"Orientation residual unavailable: {exc}")
        return out

    def data(self):
        r, m = self.record, self.manifest
        ix = np.unique(np.r_[np.arange(0, len(r.state), max(1, len(r.state) // 6000)),
                            len(r.state) - 1]) if len(r.state) else np.array([], dtype=int)
        st = r.state[ix]
        cameras = [{"name": n, "frames": len(c.index), "dropped": c.info.get("dropped", 0),
                    "lens": c.info.get("lens"), "sensor": c.info.get("sensor")}
                   for n, c in r.cameras.items()]
        events = [{"t": float(x["t_observed"]) - self.t0,
                   "session": SESSIONS.get(int(x["session"]), str(x["session"])),
                   "bus": BUSES.get(int(x["bus"]), str(x["bus"]))} for x in r.session]
        transitions = [e for i, e in enumerate(events) if i == 0 or
                       (e["session"], e["bus"]) != (events[i-1]["session"], events[i-1]["bus"])]
        return clean({"name": r.path.name, "path": str(r.path), "created": m.get("created"),
                      "ended": m.get("ended"), "complete": m.get("complete"),
                      "duration": self.duration, "code": m.get("code"),
                      "task": self.episode.get("task", "Recorded rollout"),
                      "episode_available": bool(self.episode),
                      "finish": self.episode.get("finish"), "cameras": cameras,
                      "state_count": len(r.state), "target_count": len(r.targets),
                      "warnings": self.warnings, "actions": self.actions, "events": transitions,
                      "model_available": self.robot is not None,
                      "state": {"t": np.round(st["t"] - self.t0, 6),
                                "q": np.round(np.degrees(st["q"]), 4),
                                "cmd": np.round(np.degrees(st["q_cmd"]), 4)},
                      "targets": {"t": np.round(r.targets["deadline"] - self.t0, 6),
                                  "q": np.round(np.degrees(r.targets["q"]), 4),
                                  "position_mm": np.round(r.targets["err"] * 1000, 4),
                                  "orientation_deg": np.round(self.rotation_errors, 4)}})

    def sample(self, seconds):
        r = self.record
        t = self.t0 + np.clip(seconds, 0, self.duration)
        si = previous(r.state["t"], t)
        ti = previous(r.targets["deadline"], t)
        st = None if si is None else r.state[si]
        tg = None if ti is None else r.targets[ti]
        ci = previous(r.session["t_observed"], t)
        control = None if ci is None else r.session[ci]
        cameras = {}
        for name, cam in r.cameras.items():
            if not len(cam.index):
                cameras[name] = None
                continue
            j = int(np.searchsorted(cam.index["t_capture"], t))
            j = min(j, len(cam.index) - 1)
            if j > 0 and abs(cam.index[j-1]["t_capture"] - t) < abs(cam.index[j]["t_capture"] - t):
                j -= 1
            cameras[name] = {"index": j, "t": float(cam.index[j]["t_capture"] - self.t0),
                             "delta_ms": float((cam.index[j]["t_capture"] - t) * 1000)}
        return clean({"t": float(t - self.t0), "cameras": cameras,
                      "state_t": None if st is None else float(st["t"] - self.t0),
                      "measured": None if st is None else st["q"],
                      "command": None if st is None else st["q_cmd"],
                      "load": None if st is None else st["load"],
                      "temp": None if st is None else st["temp"],
                      "target": None if tg is None else tg["q"],
                      "pose": None if tg is None else tg["pose"],
                      "position_mm": None if tg is None else float(tg["err"] * 1000),
                      "orientation_deg": None if ti is None else self.rotation_errors[ti],
                      "session": None if control is None else SESSIONS.get(int(control["session"])),
                      "bus": None if control is None else BUSES.get(int(control["bus"]))})


class Catalog:
    def __init__(self, root, episodes=None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Recording directory does not exist: {self.root}")
        self.episodes = Path(episodes).resolve() if episodes else None
        self.paths = {}
        self.cache = {}
        self.lock = RLock()
        self.records()

    def records(self):
        files = ([self.root / "manifest.json"] if (self.root / "manifest.json").exists()
                 else sorted(self.root.rglob("manifest.json")))
        entries, paths = [], {}
        for file in files:
            path = file.parent.resolve()
            if not path.is_relative_to(self.root):
                continue
            if any(x in ("policy_replays", "executor_replays") for x in file.relative_to(self.root).parts):
                continue
            try:
                m = json.loads(file.read_text())
                if m.get("format") != "elrobot-record/1":
                    continue
                key = path.relative_to(self.root).as_posix()
                paths[key] = path
                entries.append({"id": key, "name": path.name, "created": m.get("created"),
                                "complete": bool(m.get("complete"))})
            except (OSError, ValueError):
                continue
        with self.lock:
            self.paths = paths
        return sorted(entries, key=lambda x: x.get("created") or x["name"], reverse=True)

    def get(self, key, reload=False):
        with self.lock:
            if key not in self.paths:
                raise KeyError("Recording not found")
            if reload or key not in self.cache:
                path = self.paths[key]
                ep = None
                if self.episodes and self.episodes.is_dir():
                    for f in self.episodes.glob("*/episode.json"):
                        if f.parent.is_symlink():
                            continue
                        try:
                            if Path(json.loads(f.read_text()).get("record", "")).resolve() == path:
                                ep = f.parent
                                break
                        except (OSError, ValueError):
                            continue
                self.cache[key] = Rollout(path, ep)
                while len(self.cache) > 2:
                    oldest = next(k for k in self.cache if k != key)
                    del self.cache[oldest]
            return self.cache[key]
