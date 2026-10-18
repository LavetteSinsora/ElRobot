"""The Astra harness: the workspace's episodes, and the four tools behind the MCP server.

Started by P0 at INIT of an `astra` launch; runs until STOPPED, serving any number of episodes
(one per `start`). What runs where:

    Codex / Astra    (opened on the workspace, `ElRobot/astra/`, which holds AGENTS.md)
      | stdio (MCP, JSON-RPC)
    MCP server        astra/mcp_server.py, a child of Codex; stateless
      | Unix socket (/tmp/elrobot-astra/harness.sock), one JSON line each way per tool call
    the harness       this process: the tools, the episode folders, images, timelines
      | Unix socket (player.sock)                         ^ tails P9's record
    P12, the Astra source -> intervention ring -> P6 (IK) -> P7 (bus) -> the arm

**An episode** is one rollout: `start(task)` asks P0 for START (P0 is waiting at READY, the
arm at home: the operator pressed Home once, at launch), opens `episodes/<stamp>_<task>/`,
waits for the first ticks and returns the opening observation. `finish` asks P0 for RESET:
the arm goes home, P0 waits at READY again, and the next `start` begins the next episode. If
the operator presses Start themselves, the next `start` adopts that rollout.

**The record is the truth.** Everything an episode writes (frames/, timeline.sqlite, the
measured values in every reply) is read back from P9's record as P9 writes it (`record.Tail`),
never from the rings. `Ingest` turns it into ticks: one per frame of the anchor camera (30 Hz),
every other camera's nearest frame, the bus read nearest that instant, the target the IK was
solving then.

**One action** (`send_action`): checked here (nothing moves on a bad one), planned and run by
P12, which answers twice: `planned` (the composed waypoints, each with its deadline) and
`done` (when the arm was released). The reply is built once the record covers the release:
`actions/N/execution_outcome.json`, and step N+1's observation (strip A now, sheet H2 of the
motion, the text), which is also what the tool returns.
"""

from __future__ import annotations

import bisect
import json
import os
import socket
import socketserver
import threading
import time
from pathlib import Path

import numpy as np

from elrobot_deploy.astra import render
from elrobot_deploy.astra.frames import GripperMap, TaskFrame, rotation_angle, rpy_from_mat
from elrobot_deploy.astra.replies import error_text, now_section, result_section
from elrobot_deploy.astra.timeline import Timeline
from elrobot_deploy.astra.trajectory import Limits, parse_action
from elrobot_deploy.astra.workspace import (
    AstraLaunch,
    anchor_camera,
    configured_cameras,
    new_episode,
    update_json,
)
from elrobot_deploy.robot.transforms import mat_from_quat
from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all, schema

#: How long a tick waits for the other cameras' frames before it is written without them.
OTHER_CAMERA_WAIT_S = 0.5
#: Another camera's frame further than this from the anchor's is not that tick's.
MATCH_S = 0.1
#: Ticks before the opening observation is taken (the arm at rest at the episode's start).
OPEN_TICKS = 6
COMMIT_S = 0.25
#: How much state history is kept in memory (the record keeps all of it).
STATE_KEEP_S = 10.0
#: How long a reply waits for the record to cover the arm's release.
COVER_WAIT_S = 3.0
#: `start`: how long it waits for P0 to be ready (homing after the last episode), for the
#: rollout to begin, for P9's record of it, and for the first ticks.
READY_WAIT_S = 30.0
ACTIVE_WAIT_S = 10.0
RECORD_WAIT_S = 10.0
OPEN_WAIT_S = 20.0

JOINTS = tuple(f"rev_motor_0{i}" for i in range(1, 8)) + ("gripper (rev_motor_08)",)


def _cat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b if len(a) == 0 else (a if len(b) == 0 else np.concatenate([a, b]))


class Ingest:
    """P9's record -> ticks: frames/ files, timeline.sqlite rows, and the same rows in memory
    for the replies. Only the harness's main thread calls `step`; readers wait on `cond`."""

    def __init__(self, cfg, folder: Path, record, *, fk, log=print):
        from elrobot_deploy.record import Tail
        self.cfg = cfg
        self.folder = Path(folder)
        self.tail = Tail(record)
        self.log = log
        self.fk = fk                                   # q7 -> 4x4, base_link
        self.frame = TaskFrame.from_config(cfg)
        self.gmap = GripperMap.from_config(cfg)
        self.cams = [c for c in cfg.astra.cameras if c in self.tail.cameras]
        if not self.cams:
            self.cams = list(self.tail.cameras)
        self.anchor = anchor_camera(cfg, self.cams)
        self.kinds = [cfg.camera(c).lens for c in self.cams]
        self.aspects = [cfg.camera(c).width / cfg.camera(c).height for c in self.cams]
        self.timeline = Timeline(self.folder / "timeline.sqlite", self.cams, JOINTS)
        m = self.tail.manifest["streams"]
        from elrobot_deploy.record import _dtype
        self.state = np.zeros(0, _dtype(m["state"]["descr"]))
        self.targets = np.zeros(0, _dtype(m["targets"]["descr"]))
        self.cam_rows: dict[str, np.ndarray] = {}
        self.pending: list = []                        # anchor frames not yet ticks
        self.first_fid: int | None = None
        self.t0: float | None = None
        self.ticks: dict[int, dict] = {}
        self.order: list[int] = []                     # ticks in order
        self.times: list[float] = []                   # their monotonic instants
        self.actions: list[dict] = []                  # {id, t0, t_end}
        self.cond = threading.Condition()
        self._t_commit = time.monotonic()

    # --- what happened when ---------------------------------------------------------------

    def begin_action(self, aid: int, t0: float) -> None:
        with self.cond:
            self.actions.append({"id": aid, "t0": t0, "t_end": None})

    def end_action(self, aid: int, t_end: float) -> None:
        with self.cond:
            for a in self.actions:
                if a["id"] == aid:
                    a["t_end"] = t_end
            late = [k for k, tm in zip(self.order, self.times)
                    if tm > t_end and self.ticks[k]["action_id"] == aid]
        if late:
            for k in late:
                self.ticks[k]["action_id"], self.ticks[k]["phase"] = None, "holding"
            self.timeline.relabel(range(late[0], late[-1] + 1), None, "holding")

    def _driver(self, t: float):
        for a in reversed(self.actions):
            if a["t0"] - 1e-6 <= t and (a["t_end"] is None or t <= a["t_end"] + 1e-6):
                return a["id"]
        return None

    # --- the ingest ------------------------------------------------------------------------

    def step(self) -> int:
        new = self.tail.poll()
        if len(new["state"]):
            self.state = _cat(self.state, new["state"])
        if len(new["targets"]):
            self.targets = _cat(self.targets, new["targets"])
        for c in self.cams:
            rows = new.get(c)
            if rows is not None and len(rows):
                self.cam_rows[c] = _cat(self.cam_rows.get(c, rows[:0]), rows)
                if c == self.anchor:
                    self.pending.extend(rows)
        n = 0
        while self.pending and self._tick(self.pending[0]):
            self.pending.pop(0)
            n += 1
        if n:
            self._trim()
        if time.monotonic() - self._t_commit >= COMMIT_S:
            self.timeline.commit()
            self._t_commit = time.monotonic()
        return n

    def _trim(self) -> None:
        t = self.times[-1] - STATE_KEEP_S
        if len(self.state) and self.state["t"][0] < t - 1.0:
            self.state = self.state[np.searchsorted(self.state["t"], t):]
        for c, rows in self.cam_rows.items():
            if len(rows) > 600:
                self.cam_rows[c] = rows[-300:]

    def _frame_path(self, cam: str, tick: int) -> str:
        return f"frames/{cam}/t{tick // 1000:03d}/tick{tick:06d}.jpg"

    def _save(self, cam: str, row, tick: int) -> str:
        rel = self._frame_path(cam, tick)
        path = self.folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.tail.jpeg(cam, row))
        return rel

    def _tick(self, a) -> bool:
        """Turn one anchor frame into a tick, or say it must wait (False)."""
        tau = float(a["t_capture"])
        if not len(self.state) or self.state["t"][-1] < tau:
            return False                                       # the bus read after it, first
        others = {}
        for c in self.cams:
            if c == self.anchor:
                continue
            rows = self.cam_rows.get(c)
            behind = rows is None or not len(rows) or rows["t_capture"][-1] < tau
            if behind and time.monotonic() - tau < OTHER_CAMERA_WAIT_S:
                return False
            if rows is not None and len(rows):
                i = int(np.argmin(np.abs(rows["t_capture"] - tau)))
                if abs(float(rows["t_capture"][i]) - tau) <= MATCH_S:
                    others[c] = rows[i]
        fid = int(a["frame_id"])
        if self.first_fid is None:
            self.first_fid, self.t0 = fid, tau
        tick = fid - self.first_fid
        row = {"tick": tick, "t": round(tau - self.t0, 4), "t_mono": tau}
        aid = self._driver(tau)
        row["action_id"], row["phase"] = aid, ("executing" if aid is not None else "holding")
        for c in self.cams:
            src = a if c == self.anchor else others.get(c)
            row[f"{c}_path"] = None if src is None else self._save(c, src, tick)
        self._measure(row, tau)
        with self.cond:
            self.ticks[tick] = row
            self.order.append(tick)
            self.times.append(tau)
            self.cond.notify_all()
        self.timeline.add(row)
        return True

    def _task(self, T) -> tuple[np.ndarray, np.ndarray]:
        M = self.frame.M
        return M @ T[:3, 3], rpy_from_mat(M @ T[:3, :3])

    def _measure(self, row: dict, tau: float) -> None:
        st = self.state
        i = int(np.clip(np.searchsorted(st["t"], tau), 1, len(st) - 1)) if len(st) > 1 else 0
        if len(st) > 1 and abs(st["t"][i - 1] - tau) < abs(st["t"][i] - tau):
            i -= 1
        s = st[i]
        q = np.asarray(s["q"], np.float64)
        xyz, rpy = self._task(self.fk(q[:7]))
        row.update(meas_xyz=xyz, meas_rpy=rpy,
                   meas_gripper_mm=round(float(self.gmap.mm(q[7])), 2))
        slow = np.flatnonzero(np.isfinite(st["t_slow"][: i + 1]))
        if slow.size:
            sl = st[int(slow[-1])]
            row["meas_dq_rad_s"], row["meas_load_frac"] = sl["speed"].tolist(), sl["load"].tolist()
        row["meas_q_rad"] = q.tolist()
        q_cmd = np.asarray(s["q_cmd"], np.float64)
        row["cmd_q_rad"] = q_cmd.tolist()
        tg = self.targets
        j = None
        if len(tg):
            ok = np.flatnonzero((tg["deadline"] <= tau + self.cfg.dt / 2)
                                & np.isfinite(tg["pose"][:, 0]))
            if ok.size:
                j = int(ok[np.argmax(tg["deadline"][ok])])
        if j is None:
            cxyz, crpy = self._task(self.fk(q_cmd[:7]))
            row.update(cmd_xyz=cxyz, cmd_rpy=crpy,
                       cmd_gripper_mm=round(float(self.gmap.mm(q_cmd[7])), 2),
                       ik_residual_mm=None, ik_residual_deg=None)
        else:
            t = tg[j]
            pose = np.asarray(t["pose"], np.float64)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = mat_from_quat(pose[3:]), pose[:3]
            cxyz, crpy = self._task(T)
            reached = self.fk(np.asarray(t["q"], np.float64)[:7])
            deg = np.degrees(rotation_angle(reached[:3, :3].T @ T[:3, :3]))
            row.update(cmd_xyz=cxyz, cmd_rpy=crpy,
                       cmd_gripper_mm=round(float(self.gmap.mm(float(t["q"][7]))), 2),
                       ik_residual_mm=(round(float(t["err"]) * 1e3, 3) if np.isfinite(t["err"])
                                       else None),
                       ik_residual_deg=round(float(deg), 3))
        for prefix in ("meas", "cmd"):
            for k, name in enumerate(("x_m", "y_m", "z_m")):
                row[f"{prefix}_{name}"] = round(float(row[f"{prefix}_xyz"][k]), 5)
            for k, name in enumerate(("roll_deg", "pitch_deg", "yaw_deg")):
                row[f"{prefix}_{name}"] = round(float(row[f"{prefix}_rpy"][k]), 3)

    # --- readers (any thread) ----------------------------------------------------------------

    def latest(self) -> dict | None:
        with self.cond:
            return self.ticks[self.order[-1]] if self.order else None

    def tick_at(self, t_mono: float) -> int | None:
        with self.cond:
            if not self.order:
                return None
            i = bisect.bisect_left(self.times, t_mono)
            if i >= len(self.times):
                return self.order[-1]
            if i > 0 and t_mono - self.times[i - 1] < self.times[i] - t_mono:
                i -= 1
            return self.order[i]

    def row(self, tick: int) -> dict | None:
        with self.cond:
            return self.ticks.get(tick)

    def wait_for(self, t_mono: float, timeout: float) -> bool:
        """Until a tick at or after `t_mono` exists (False: the record stopped short)."""
        end = time.monotonic() + timeout
        with self.cond:
            while not (self.times and self.times[-1] >= t_mono):
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(min(left, 0.1))
            return True

    def frames(self, tick: int, cams) -> list:
        row = self.row(tick) or {}
        out = []
        for c in cams:
            rel = row.get(f"{c}_path")
            out.append(None if rel is None else render.decode((self.folder / rel).read_bytes()))
        return out

    def close(self) -> None:
        self.timeline.close()


class PlayerLink:
    """The harness's side of P12's socket: one request at a time, answers in order."""

    def __init__(self, path: str):
        self.path = path
        self.sock = None
        self.buf = b""

    def _connect(self, timeout: float = 30.0) -> None:
        end = time.monotonic() + timeout
        while True:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(self.path)
                self.sock, self.buf = s, b""
                return
            except OSError:
                s.close()
                if time.monotonic() > end:
                    raise ConnectionError(f"the Astra source (P12) is not listening on "
                                          f"{self.path}") from None
                time.sleep(0.1)

    def send(self, msg: dict) -> None:
        if self.sock is None:
            self._connect()
        self.sock.sendall(json.dumps(msg).encode() + b"\n")

    def recv(self) -> dict:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                self.sock = None
                raise ConnectionError("the Astra source (P12) closed its socket")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)


class Episode:
    """One task: its folder, its ticks (`Ingest`), its decision steps. Built by `start`; its
    ingest is stepped by the harness's main thread; the tools run in the server's threads."""

    def __init__(self, harness: Harness, folder: Path, task: str, ingest: Ingest):
        self.h = harness
        self.cfg = harness.cfg
        self.folder = Path(folder)
        self.name = self.folder.name
        self.task = task
        self.ingest = ingest
        self.step = 1
        self.n_actions = 0
        self.finished = False
        self.stopped_why: str | None = None
        self.closed = False
        self.opened = threading.Event()
        self.opening: dict | None = None
        #: When its rollout began (the harness's `t_active`): how a rollout is told apart.
        self.t_active: float | None = None
        self._n_inspect = 0

    @property
    def running(self) -> bool:
        return not (self.finished or self.stopped_why)

    # --- folders and images ----------------------------------------------------------------

    def step_dir(self, n: int | None = None) -> Path:
        d = self.folder / "actions" / f"{self.step if n is None else n:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, path: Path) -> str:
        """A path as Astra uses it: relative to the workspace (Codex's working directory)."""
        return str(Path(path).relative_to(self.h.workspace))

    def _save(self, img, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render.encode(img, self.cfg.astra.jpeg_quality))
        return path

    def strip_image(self, tick: int, cams=None, prefix: str = "now  "):
        ing = self.ingest
        cams = list(cams or ing.cams)
        idx = [ing.cams.index(c) for c in cams]
        return render.strip(ing.frames(tick, cams), [ing.kinds[i] for i in idx],
                            render.label(tick, ing.row(tick)["t"], prefix),
                            [ing.aspects[i] for i in idx])

    def sheet_image(self, ticks, cams=None):
        ing = self.ingest
        cams = list(cams or ing.cams)
        idx = [ing.cams.index(c) for c in cams]
        cols = [(render.label(k, ing.row(k)["t"]), ing.frames(k, cams)) for k in ticks]
        return render.sheet(cols, [ing.kinds[i] for i in idx], [ing.aspects[i] for i in idx])

    def _now(self, d: Path) -> tuple[str, Path]:
        """Strip A at the newest tick into step folder `d`, and its "Current observation"."""
        row = self.ingest.latest()
        img = self._save(self.strip_image(row["tick"]), d / "observation_now.jpg")
        return now_section(row, self.ingest.cams, self.rel(img)), img

    # --- the start ---------------------------------------------------------------------------

    def open(self) -> None:
        """The opening observation: actions/0001, and `start`'s reply. (Main thread.)"""
        ing = self.ingest
        d = self.step_dir(1)
        text, img = self._now(d)
        (d / "observation.txt").write_text(text + "\n")
        update_json(self.folder / "episode.json", cameras=list(ing.cams), clock={
            "t0_monotonic": ing.t0,
            "t0_wall": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
                time.time() - (time.monotonic() - ing.t0))),
            "anchor_camera": ing.anchor,
            "tick_hz": float(self.cfg.camera(ing.anchor).fps) if ing.anchor else 30.0})
        head = [(f"Episode {self.name} started. Its folder: episodes/{self.name}/ "
                 f"(episodes/current points at it; AGENTS.md describes what is inside)."),
                f"Task: {self.task}",
                f"Cameras, left to right in every image: {' | '.join(ing.cams)}.", ""]
        if ing.cams != configured_cameras(self.cfg):
            missing = [c for c in configured_cameras(self.cfg) if c not in ing.cams]
            head.insert(3, f"Not running this episode: {', '.join(missing)}.")
        self.opening = {"text": "\n".join(head) + text, "images": [str(img)], "is_error": False}
        self.opened.set()
        self.h.log(f"episode {self.name}: open (task: {self.task})")

    # --- the tools ---------------------------------------------------------------------------

    def _refuse(self, problems) -> dict:
        d = self.step_dir()
        with open(d / "rejected.jsonl", "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "problems": problems}) + "\n")
        return _error(error_text(problems))

    def send_action(self, args) -> dict:
        _, speed, problems = parse_action(args, self.h.limits)
        if problems:
            return self._refuse(problems)
        n = self.step
        d = self.step_dir(n)
        link = self.h.link
        link.send({"op": "action", "id": n, "waypoints": args["waypoints"], "speed": speed})
        planned = None
        while True:
            msg = link.recv()
            if msg.get("op") == "planned":
                planned = msg
                self.ingest.begin_action(n, msg["t0"])
                (d / "action.json").write_text(json.dumps({"waypoints": args["waypoints"],
                                                           "speed": speed}, indent=2) + "\n")
            elif msg.get("op") == "error":
                if planned is None:
                    return self._refuse([msg.get("error", "refused by the Astra source")])
                done = {"t_release": time.monotonic(), "interrupted": msg.get("error")}
                break
            elif msg.get("op") == "done":
                done = msg
                break
        self.ingest.end_action(n, done["t_release"])
        self.ingest.wait_for(done["t_release"] + 0.05, COVER_WAIT_S)
        out = self._outcome(n, planned, done, speed)
        (d / "execution_outcome.json").write_text(json.dumps(out, indent=2) + "\n")
        self.n_actions += 1
        with open(self.folder / "actions.jsonl", "a") as fh:
            fh.write(json.dumps({"action": n, "dir": self.rel(d), "tick_start": out["tick_start"],
                                 "tick_end": out["tick_end"], "speed": speed,
                                 "waypoints": args["waypoints"],
                                 "interrupted": out["interrupted"]}) + "\n")
        # Step N+1: what Astra sees when deciding the next action.
        self.step = n + 1
        nd = self.step_dir()
        now_text, img_now = self._now(nd)
        img_hist = self._save(self.sheet_image(out["history_ticks"]),
                              nd / "observation_history.jpg")
        text = now_text + "\n\n" + result_section(
            out, self.ingest.cams, self.rel(d), image=self.rel(img_hist),
            timeline=self.rel(self.folder / "timeline.sqlite"))
        (nd / "observation.txt").write_text(text + "\n")
        self.h.log(f"action {n}: ticks {out['tick_start']} -> {out['tick_end']}"
                   + (f", INTERRUPTED: {out['interrupted']}" if out["interrupted"] else ""))
        return {"text": text, "images": [str(img_now), str(img_hist)], "is_error": False}

    def _pose(self, pose7, g) -> dict:
        xyz, rpy = self.ingest.frame.describe(pose7)
        return {"xyz": np.round(xyz, 4).tolist(), "rpy": np.round(rpy, 2).tolist(),
                "gripper_mm": round(float(g), 1)}

    def _outcome(self, n: int, planned: dict, done: dict, speed: str) -> dict:
        ing = self.ingest
        k0 = ing.tick_at(planned["t0"])
        k1 = ing.tick_at(min(done["t_release"], ing.times[-1] if ing.times else 0.0))
        last = ing.latest()
        wps = []
        for w in planned["waypoints"]:
            k = ing.tick_at(w["deadline"]) if ing.times and ing.times[-1] >= w["deadline"] - 0.02 \
                else None
            r = ing.row(k) if k is not None else None
            wps.append({
                "wp": w["index"], "deadline_tick": k,
                "commanded": self._pose(w["pose"], w["gripper_mm"]),
                "measured": None if r is None else {
                    "xyz": np.round(r["meas_xyz"], 4).tolist(),
                    "rpy": np.round(r["meas_rpy"], 2).tolist(),
                    "gripper_mm": round(r["meas_gripper_mm"], 1)}})
        worst = None
        for k in range(k0, k1 + 1):
            r = ing.row(k)
            if r and r.get("ik_residual_mm") is not None and (
                    worst is None or r["ik_residual_mm"] > worst["mm"]):
                worst = {"mm": round(r["ik_residual_mm"], 2),
                         "deg": round(r["ik_residual_deg"] or 0.0, 2), "tick": k}
        span = max(k1 - k0, 0)
        hist = [k0 + round(i * span / 3) for i in range(3)]
        hist = [k if ing.row(k) else last["tick"] for k in hist]
        r0, r1 = ing.row(k0), ing.row(k1)
        return {
            "action": n, "speed": speed, "tick_start": k0, "tick_end": k1,
            "duration_s": (round(r1["t"] - r0["t"], 2) if r0 and r1
                           else round(done["t_release"] - planned["t0"], 2)),
            "planned_duration_s": round(planned["duration_s"], 3),
            "interrupted": done.get("interrupted"),
            "reference": self._pose(planned["ref"]["pose"], planned["ref"]["gripper_mm"]),
            "waypoints": wps, "max_ik_residual": worst, "history_ticks": hist,
        }

    def inspect(self, args) -> dict:
        if not isinstance(args, dict) or not isinstance(args.get("ticks"), list) or \
                not args["ticks"]:
            return _error("inspect_observation(ticks, cams?): `ticks` is a non-empty list of "
                          "tick numbers")
        last = self.ingest.latest()["tick"]
        ticks, problems = [], []
        for k in args["ticks"]:
            if isinstance(k, bool) or not isinstance(k, int):
                problems.append(f"tick {k!r} is not an integer")
            elif self.ingest.row(k) is None:
                problems.append(f"tick {k} is not recorded (ticks run 0 ... {last})")
            else:
                ticks.append(k)
        cams = args.get("cams") or None
        if cams is not None and (not isinstance(cams, list)
                                 or any(c not in self.ingest.cams for c in cams)):
            problems.append(f"cams: a subset of {self.ingest.cams}, got {cams!r}")
        if problems:
            return _error("inspect_observation refused:\n" + "\n".join(f"  - {p}"
                                                                          for p in problems))
        cams = cams or list(self.ingest.cams)
        d = self.step_dir() / "inspections"
        images, lines = [], []
        groups = [ticks] if len(ticks) == 1 else [ticks[i:i + 4] for i in range(0, len(ticks), 4)]
        for g in groups:
            self._n_inspect += 1
            path = d / f"{self._n_inspect:03d}.jpg"
            if len(g) == 1:
                self._save(self.strip_image(g[0], cams, prefix=""), path)
                what = f"tick {g[0]}, cameras {' | '.join(cams)}"
            else:
                self._save(self.sheet_image(g, cams), path)
                what = f"columns tick {', '.join(map(str, g))}; rows {', '.join(cams)}"
            images.append(str(path))
            lines.append(f"[image {self.rel(path)}: {what}]")
        return {"text": "\n".join(lines), "images": images, "is_error": False}

    def finish(self, args) -> dict:
        row = self.ingest.latest() or {"tick": None, "t": None}
        update_json(self.folder / "episode.json", finish={
            "success": args["success"], "summary": args["summary"],
            "lessons": args.get("lessons"), "tick": row["tick"], "t": row["t"],
            "actions": self.n_actions, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "human_outcome": None})
        self.finished = True
        self.h.spent = self.h.t_active             # this rollout is over, whatever P0 says yet
        control = self.h.control
        going_home = control.session == schema.ACTIVE
        if going_home:
            control.request = schema.REQ_RESET
        self.h.log(f"episode {self.name} finished: success={args['success']}: "
                   f"{args['summary']}")
        return {"text": f"Recorded in episodes/{self.name}/episode.json. The episode is over"
                        + ("; the robot is going home. `start` begins the next one."
                           if going_home else "."),
                "images": [], "is_error": False}


class Harness:
    """The tools. `call(tool, args)` -> {"text", "images": [paths], "is_error"}."""

    def __init__(self, cfg, rings, launch: AstraLaunch, link: PlayerLink, *, fk, log=print):
        self.cfg = cfg
        self.control = rings["control"]
        self.launch = launch
        self.workspace = Path(launch.workspace)
        self.link = link
        self.fk = fk
        self.log = log
        self.limits = Limits.from_config(cfg)
        self.episode: Episode | None = None
        #: When the main loop last saw the session become ACTIVE (a Start nobody asked for
        #: here is adopted by the next `start`).
        self.t_active: float | None = None
        #: The `t_active` of the rollout the last episode used: never adopted again (after
        #: `finish`, P0 may not have taken the RESET yet when the next `start` arrives).
        self.spent: float | None = None
        self.busy = threading.Lock()                 # one action (or start / finish) at a time

    def call(self, tool: str, args) -> dict:
        try:
            if tool == "start":
                return self.start(args)
            if tool not in ("send_action", "inspect_observation", "finish"):
                return _error(f"unknown tool {tool!r}")
            ep = self.episode
            if ep is None or not ep.opened.is_set():
                why = "no episode is running: call `start` with the task first"
                return _error(error_text([why]) if tool == "send_action" else why)
            if tool == "inspect_observation":
                return ep.inspect(args)
            if not ep.running:
                why = ("the episode is finished: `start` begins the next one" if ep.finished
                       else f"{ep.stopped_why}; `start` begins a new episode")
                return _error(error_text([why]) if tool == "send_action" else why)
            if tool == "finish" and not (isinstance(args, dict)
                                         and isinstance(args.get("success"), bool)
                                         and isinstance(args.get("summary"), str)):
                return _error("finish(success, summary, lessons?): `success` true/false, "
                              "`summary` text, `lessons` optional text")
            if not self.busy.acquire(blocking=False):
                return _error(error_text([("another action is still running (one at a time; "
                                           "this one was not queued)")])
                              if tool == "send_action" else
                              "an action is still running: finish once it has replied")
            try:
                return ep.send_action(args) if tool == "send_action" else ep.finish(args)
            finally:
                self.busy.release()
        except Exception as exc:                                   # noqa: BLE001
            self.log(f"{tool} failed: {type(exc).__name__}: {exc}")
            return _error(f"{tool} failed inside the harness: {type(exc).__name__}: {exc}")

    # --- start ---------------------------------------------------------------------------------

    def _wait(self, predicate, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > end or self.control.session >= schema.STOPPED:
                return False
            time.sleep(0.02)
        return True

    def _find_record(self, t_from: float) -> Path | None:
        """P9's record of the rollout that began after `t_from` (its manifest says when)."""
        best = None
        for m in Path(self.launch.record_base).glob("*/manifest.json"):
            try:
                t = json.loads(m.read_text())["rollout"]["t_start"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if t >= t_from - 1.0 and (best is None or t > best[0]):
                best = (t, m.parent)
        return None if best is None else best[1]

    def start(self, args) -> dict:
        task = args.get("task") if isinstance(args, dict) else None
        if not isinstance(task, str) or not task.strip():
            return _error("start(task): `task` is the task in words, as the person gave it")
        if not self.busy.acquire(blocking=False):
            return _error("busy: a start or an action is in progress")
        try:
            return self._start(task.strip())
        finally:
            self.busy.release()

    def _start(self, task: str) -> dict:
        ep = self.episode
        if ep is not None and ep.running:
            return _error(f"episode {ep.name} is still running: call `finish` first")
        control = self.control
        not_ready = ("the robot is not ready (session {}): the person must start "
                     "`elrobot_deploy astra` and press Home")
        spent = control.session == schema.ACTIVE and self.t_active == self.spent
        if spent or control.session in (schema.INIT, schema.CHECK, schema.HOME) or (
                control.session == schema.READY and control.wait != schema.WAIT_START):
            # Homing after the last episode (or the launch still checking): wait for READY.
            self._wait(lambda: control.session == schema.READY
                       and control.wait == schema.WAIT_START, READY_WAIT_S)
        if control.session == schema.READY and control.wait == schema.WAIT_START:
            t_from = time.monotonic()
            control.request = schema.REQ_START
            if not self._wait(lambda: control.session == schema.ACTIVE, ACTIVE_WAIT_S):
                return _error(not_ready.format(schema.SESSION_NAMES.get(control.session)))
        elif (control.session == schema.ACTIVE and self.t_active is not None
              and self.t_active != self.spent):
            t_from = self.t_active                     # Start pressed by the person: adopt it
        else:
            return _error(not_ready.format(schema.SESSION_NAMES.get(control.session)))
        self._wait(lambda: self._find_record(t_from) is not None, RECORD_WAIT_S)
        record = self._find_record(t_from)
        if record is None:
            return _error("the rollout began but no recording of it appeared (is P9 running?)")
        folder = new_episode(self.cfg, self.launch, task, configured_cameras(self.cfg))
        (folder / "record").symlink_to(record)
        update_json(folder / "episode.json", record=str(record))
        ep = Episode(self, folder, task, Ingest(self.cfg, folder, record, fk=self.fk,
                                                log=self.log))
        if ep.ingest.anchor is None:
            return _error("no camera is recorded: an episode needs at least one")
        ep.t_active = self.t_active
        self.spent = self.t_active                     # this rollout is now this episode's
        self.episode = ep                              # the main loop steps it from here
        if not ep.opened.wait(OPEN_WAIT_S):
            ep.stopped_why = "no camera frames arrived for the opening observation"
            return _error(f"{ep.stopped_why} (is the {ep.ingest.anchor} camera running?)")
        return ep.opening

    # --- the main loop's part ------------------------------------------------------------------

    def tick(self) -> None:
        """Called by the main loop: ingest, open, notice the rollout ending, close."""
        control = self.control
        active = control.session == schema.ACTIVE
        if active and self.t_active is None:
            self.t_active = time.monotonic()
        elif not active:
            self.t_active = None
        ep = self.episode
        if ep is None or ep.closed:
            return
        new = ep.ingest.step()
        if not ep.opened.is_set() and len(ep.ingest.order) >= OPEN_TICKS:
            ep.open()
        if not active and ep.running:
            self.spent = ep.t_active
            ep.stopped_why = (f"the rollout ended (session "
                              f"{schema.SESSION_NAMES.get(control.session)}): the person pressed "
                              f"Reset or Exit, or a guard stopped the arm")
            self.log(f"episode {ep.name}: {ep.stopped_why}")
        if not active and not new and ep.ingest.tail.complete() and not self.busy.locked():
            ep.ingest.step()                           # what landed with the manifest
            ep.ingest.close()
            ep.closed = True


def _error(text: str) -> dict:
    return {"text": text, "images": [], "is_error": True}


def serve(path: str, harness: Harness):
    """The MCP server's socket: one JSON line in ({"tool", "args"}), one out, per call; calls
    in parallel each get their own thread (`send_action` refuses a second at once)."""

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline()
            try:
                req = json.loads(line)
                out = harness.call(req.get("tool"), req.get("args") or {})
            except ValueError as exc:
                out = _error(f"bad request: {exc}")
            self.wfile.write(json.dumps(out).encode() + b"\n")

    if os.path.exists(path):
        os.unlink(path)
    server = socketserver.ThreadingUnixStreamServer(path, Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="astra-tools", daemon=True).start()
    return server


def run(shm_name: str, cfg, *, launch: AstraLaunch) -> None:
    """The harness as a process: serves the tools from INIT to STOPPED."""
    def log(text: str) -> None:
        print(f"[astra] {text}", flush=True)

    arena = Arena.attach(shm_name, layout_for(cfg))
    server = harness = None
    try:
        rings = open_all(arena)
        control = rings["control"]
        from elrobot_deploy.robot.elrobot_model import ElRobotModel
        model = ElRobotModel(cfg.urdf)
        harness = Harness(cfg, rings, launch, PlayerLink(launch.player_socket), fk=model.fk,
                          log=log)
        server = serve(launch.harness_socket, harness)
        log(f"workspace {launch.workspace}: open it in Codex; Astra calls `start` with the task")
        while control.session < schema.STOPPED:
            harness.tick()
            time.sleep(0.01)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if harness is not None and harness.episode is not None and not harness.episode.closed:
            harness.episode.ingest.close()
        for p in (launch.harness_socket, launch.player_socket):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(launch.socket_dir)        # empty once both sockets are gone
        except OSError:
            pass
        arena.close()


__all__ = ["Episode", "Harness", "Ingest", "PlayerLink", "run", "serve"]
