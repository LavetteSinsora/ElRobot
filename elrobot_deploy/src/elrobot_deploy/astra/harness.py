"""The Astra harness: the session folder, and the three tools behind the MCP server.

Started by P0 at INIT of an `astra` session; runs until STOPPED. What runs where:

    Codex / Astra
      | stdio (MCP, JSON-RPC)
    MCP server        astra/mcp_server.py, a child of Codex; stateless
      | Unix socket (AstraLaunch.harness_socket), one JSON line each way per tool call
    the harness       this process: the tools, the folder, the images, the timeline
      | Unix socket (AstraLaunch.player_socket)          ^ tails P9's record
    P12, the Astra source -> intervention ring -> P6 (IK) -> P7 (bus) -> the arm

**The record is the truth.** Everything the harness writes (frames/, timeline.sqlite, the
measured values in every reply) is read back from P9's record as P9 writes it (`record.Tail`),
never from the rings: what Astra is told is what was recorded. `Ingest` turns it into ticks:
one per frame of the anchor camera (30 Hz), every other camera's nearest frame, the bus read
nearest that instant, the target the IK was solving then.

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
from elrobot_deploy.astra.session import (
    AstraLaunch,
    anchor_camera,
    codex_script,
    opening_prompt,
    update_json,
    write_readme,
)
from elrobot_deploy.astra.timeline import Timeline
from elrobot_deploy.astra.trajectory import Limits, parse_action
from elrobot_deploy.robot.transforms import mat_from_quat
from elrobot_deploy.runtime.shared_memory import Arena, layout_for, open_all, schema

#: How long a tick waits for the other cameras' frames before it is written without them.
OTHER_CAMERA_WAIT_S = 0.5
#: Another camera's frame further than this from the anchor's is not that tick's.
MATCH_S = 0.1
#: Ticks before the opening observation is taken (the arm at rest at the session start).
OPEN_TICKS = 6
COMMIT_S = 0.25
#: How much state history is kept in memory (the record keeps all of it).
STATE_KEEP_S = 10.0
#: How long a reply waits for the record to cover the arm's release.
COVER_WAIT_S = 3.0

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


class Harness:
    """The tools. `call(tool, args)` -> {"text", "images": [paths], "is_error"}."""

    def __init__(self, cfg, rings, launch: AstraLaunch, ingest: Ingest, link: PlayerLink, *,
                 log=print):
        self.cfg = cfg
        self.control = rings["control"]
        self.launch = launch
        self.folder = Path(launch.session_dir)
        self.ingest = ingest
        self.link = link
        self.log = log
        self.frame = ingest.frame
        self.limits = Limits.from_config(cfg)
        self.quality = cfg.astra.jpeg_quality
        self.step = 1
        self.n_actions = 0
        self.ready = False
        self.finished = False
        self.stopped_why: str | None = None
        self.busy = threading.Lock()
        self._n_inspect = 0

    # --- folders and images ----------------------------------------------------------------

    def step_dir(self, n: int | None = None) -> Path:
        d = self.folder / "actions" / f"{self.step if n is None else n:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, path: Path) -> str:
        return str(Path(path).relative_to(self.folder))

    def _save(self, img, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render.encode(img, self.quality))
        return path

    def strip_image(self, tick: int, cams=None, prefix: str = "now  "):
        cams = list(cams or self.ingest.cams)
        idx = [self.ingest.cams.index(c) for c in cams]
        row = self.ingest.row(tick)
        return render.strip(self.ingest.frames(tick, cams), [self.ingest.kinds[i] for i in idx],
                            render.label(tick, row["t"], prefix),
                            [self.ingest.aspects[i] for i in idx])

    def sheet_image(self, ticks, cams=None):
        cams = list(cams or self.ingest.cams)
        idx = [self.ingest.cams.index(c) for c in cams]
        cols = [(render.label(k, self.ingest.row(k)["t"]), self.ingest.frames(k, cams))
                for k in ticks]
        return render.sheet(cols, [self.ingest.kinds[i] for i in idx],
                            [self.ingest.aspects[i] for i in idx])

    # --- the session start -------------------------------------------------------------------

    def open(self) -> None:
        """The opening observation (tool-less start): actions/0001, the opening prompt, and
        the script that starts Codex on it."""
        row = self.ingest.latest()
        cams = list(self.ingest.cams)
        if cams != [c for c in self.cfg.astra.cameras if c in {k.name for k in self.cfg.cameras}]:
            # A camera of the profile did not start (the iPhone not around, say): the README
            # describes the ones Astra will actually see.
            write_readme(self.cfg, self.launch, cams)
            update_json(self.folder / "session.json", cameras=cams)
            self.log(f"cameras for this session: {', '.join(cams)}")
        d = self.step_dir(1)
        self._save(self.strip_image(row["tick"]), d / "observation_now.jpg")
        text = now_section(row, self.ingest.cams)
        (d / "observation.txt").write_text(text + "\n")
        (self.folder / "opening_prompt.md").write_text(opening_prompt(self.launch.task, text))
        script = self.folder / "start_codex.sh"
        script.write_text(codex_script(self.launch))
        os.chmod(script, 0o755)
        update_json(self.folder / "session.json", clock={
            "t0_monotonic": self.ingest.t0,
            "t0_wall": time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(time.time() - (time.monotonic()
                                                                   - self.ingest.t0))),
            "anchor_camera": self.ingest.anchor, "tick_hz": self.ingest_fps()})
        self.ready = True
        self.log(f"READY. Start Astra in another terminal:\n    {script}")

    def ingest_fps(self) -> float:
        return float(self.cfg.camera(self.ingest.anchor).fps) if self.ingest.anchor else 30.0

    # --- the tools ---------------------------------------------------------------------------

    def call(self, tool: str, args) -> dict:
        try:
            if tool == "send_action":
                return self.send_action(args)
            if tool == "inspect_observation":
                return self.inspect(args)
            if tool == "finish":
                return self.finish(args)
            return _error(f"unknown tool {tool!r}")
        except Exception as exc:                                   # noqa: BLE001
            self.log(f"{tool} failed: {type(exc).__name__}: {exc}")
            return _error(f"{tool} failed inside the harness: {type(exc).__name__}: {exc}")

    def _not_now(self) -> str | None:
        if not self.ready:
            return "the session is still starting (no opening observation yet)"
        if self.finished:
            return "the episode is finished: no robot tool works after `finish`"
        if self.stopped_why:
            return self.stopped_why
        return None

    def _refuse(self, problems) -> dict:
        d = self.step_dir()
        with open(d / "rejected.jsonl", "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), "problems": problems}) + "\n")
        return _error(error_text(problems))

    def send_action(self, args) -> dict:
        why = self._not_now()
        if why:
            return _error(error_text([why]))
        if not self.busy.acquire(blocking=False):
            return _error(error_text([("another action is still running (one at a time; "
                                       "this one was not queued)")]))
        try:
            return self._send_action(args)
        finally:
            self.busy.release()

    def _send_action(self, args) -> dict:
        _, speed, problems = parse_action(args, self.limits)
        if problems:
            return self._refuse(problems)
        n = self.step
        d = self.step_dir(n)
        self.link.send({"op": "action", "id": n, "waypoints": args["waypoints"],
                        "speed": speed})
        planned = None
        while True:
            msg = self.link.recv()
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
        now = self.ingest.latest()
        img_now = self._save(self.strip_image(now["tick"]), nd / "observation_now.jpg")
        img_hist = self._save(self.sheet_image(out["history_ticks"]),
                              nd / "observation_history.jpg")
        text = now_section(now, self.ingest.cams) + "\n\n" + result_section(
            out, self.ingest.cams, self.rel(d))
        (nd / "observation.txt").write_text(text + "\n")
        self.log(f"action {n}: ticks {out['tick_start']} -> {out['tick_end']}"
                 + (f", INTERRUPTED: {out['interrupted']}" if out["interrupted"] else ""))
        return {"text": text, "images": [str(img_now), str(img_hist)], "is_error": False}

    def _pose(self, pose7, g) -> dict:
        xyz, rpy = self.frame.describe(pose7)
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
        return {
            "action": n, "speed": speed, "tick_start": k0, "tick_end": k1,
            "duration_s": round(ing.row(k1)["t"] - ing.row(k0)["t"], 2) if ing.row(k0) and
            ing.row(k1) else round(done["t_release"] - planned["t0"], 2),
            "planned_duration_s": round(planned["duration_s"], 3),
            "interrupted": done.get("interrupted"),
            "reference": self._pose(planned["ref"]["pose"], planned["ref"]["gripper_mm"]),
            "waypoints": wps, "max_ik_residual": worst, "history_ticks": hist,
        }

    def inspect(self, args) -> dict:
        if not self.ready:
            return _error("the session is still starting (no opening observation yet)")
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
        if self.finished:
            return _error("finish was already called")
        if not isinstance(args, dict) or not isinstance(args.get("success"), bool) or \
                not isinstance(args.get("summary"), str):
            return _error("finish(success, summary, lessons?): `success` true/false, "
                          "`summary` text, `lessons` optional text")
        if not self.busy.acquire(blocking=False):
            return _error("an action is still running: finish once it has replied")
        try:
            row = self.ingest.latest() or {"tick": None, "t": None}
            update_json(self.folder / "session.json", finish={
                "success": args["success"], "summary": args["summary"],
                "lessons": args.get("lessons"), "tick": row["tick"], "t": row["t"],
                "actions": self.n_actions, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "human_outcome": None})
            self.finished = True
            going_home = self.control.session == schema.ACTIVE
            if going_home:
                self.control.request = schema.REQ_RESET
            self.log(f"finish: success={args['success']}: {args['summary']}")
            return {"text": "Recorded in session.json. The episode is over"
                            + ("; the robot is going home." if going_home else "."),
                    "images": [], "is_error": False}
        finally:
            self.busy.release()


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
    """The harness as a process. Waits for the rollout, then serves until STOPPED."""
    def log(text: str) -> None:
        print(f"[astra] {text}", flush=True)

    arena = Arena.attach(shm_name, layout_for(cfg))
    server = ingest = None
    try:
        rings = open_all(arena)
        control = rings["control"]
        folder = Path(launch.session_dir)
        log(f"session folder {folder}")
        while control.session != schema.ACTIVE:
            if control.session >= schema.STOPPED:
                return
            time.sleep(0.05)
        record = Path(launch.record_dir)
        while not (record / "manifest.json").exists():
            if control.session >= schema.STOPPED:
                return
            time.sleep(0.05)
        link_path = folder / "record"
        if not link_path.exists():
            link_path.symlink_to(record)
        from elrobot_deploy.robot.elrobot_model import ElRobotModel
        model = ElRobotModel(cfg.urdf)
        ingest = Ingest(cfg, folder, record, fk=model.fk, log=log)
        if ingest.anchor is None:
            log("no camera is recorded: an Astra session needs at least one")
        harness = Harness(cfg, rings, launch, ingest, PlayerLink(launch.player_socket), log=log)
        server = serve(launch.harness_socket, harness)
        t_warn = time.monotonic() + 10.0
        while control.session < schema.STOPPED:
            ingest.step()
            if not harness.ready and len(ingest.order) >= OPEN_TICKS:
                harness.open()
            elif not harness.ready and time.monotonic() > t_warn:
                log(f"still no frames from the anchor camera ({ingest.anchor}): is it running?")
                t_warn = time.monotonic() + 10.0
            if control.session != schema.ACTIVE and harness.stopped_why is None:
                harness.stopped_why = (
                    f"the rollout ended (session {schema.SESSION_NAMES.get(control.session)}"
                    f"): the operator pressed Reset or Exit, or a guard stopped the arm")
                log(harness.stopped_why)
            time.sleep(0.01)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if ingest is not None:
            ingest.close()
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


__all__ = ["Harness", "Ingest", "PlayerLink", "run", "serve"]
