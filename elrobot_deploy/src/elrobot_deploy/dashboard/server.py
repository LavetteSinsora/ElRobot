"""P10, the dashboard: a small HTTP server and one page that draws the arm from our URDF.

    elrobot_deploy run                 P0 starts it (http://127.0.0.1:8080 by default)
    elrobot_deploy view                no robot, no session: the URDF at any pose (sliders),
                                       with the home and exit poses to compare against

What the page shows, live (polling `/api/state` ~20 times a second):

  * the arm as MEASURED (solid), as COMMANDED by the bus process (a blue ghost: what it is
    writing to the servos, or would write in a dry run) and, before homing, the HOME pose (a
    green ghost: where the arm will go when you press Home); the IK's target point;
  * the session, what P0 waits for, the bus process's state, the hardware check, every joint
    (measured, commanded, error, temperature, load), the phone (streaming, engaged, latency);
  * the buttons Home / Start / Reset / Exit. A button only STORES a request in the arena
    (`control.request`); P0 decides. The page never commands the arm.

Endpoints: `GET /` (the page), `GET /static/...`, `GET /api/robot` (the URDF as JSON),
`GET /mesh/<i>` (only the URDF's own meshes), `GET /api/state`, `POST /api/request`
(`{"request": "HOME" | "START" | "RESET" | "EXIT"}`).

Standard library only (`http.server`); three.js is vendored under `static/vendor/three`, so the
page works without the internet. The server binds `[dashboard] host` (localhost by default):
it has no password, so open it to the LAN only on a network you trust.
"""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from elrobot_deploy.dashboard.robot import describe
from elrobot_deploy.runtime.shared_memory import schema

STATIC = Path(__file__).resolve().parent / "static"

REQUESTS = {"HOME": schema.REQ_HOME, "START": schema.REQ_START, "RESET": schema.REQ_RESET,
            "EXIT": schema.REQ_EXIT}


def _floats(a) -> list[float] | None:
    if a is None:
        return None
    return [None if not math.isfinite(float(x)) else round(float(x), 6)
            for x in np.asarray(a, dtype=float).ravel()]


class Source:
    """What `/api/state` answers from: the live arena of a session, or nothing (`view`)."""

    def __init__(self, cfg, rings=None, *, dry_run: bool = False):
        self.cfg = cfg
        self.rings = rings
        self.dry_run = dry_run
        self._report_seq = 0
        self._report = None

    def state(self) -> dict:
        cfg = self.cfg
        out = {"live": self.rings is not None, "dry_run": self.dry_run,
               "driver": cfg.bus.driver, "home": _floats(cfg.poses.home),
               "exit": _floats(cfg.poses.exit),
               "lower": _floats(cfg.limits.lower), "upper": _floats(cfg.limits.upper),
               "workspace": [list(cfg.teleop.workspace_min), list(cfg.teleop.workspace_max)]}
        if self.rings is None:
            return out
        c = self.rings["control"]
        seq, fault = c.fault()
        rseq, report = c.report()
        if rseq != self._report_seq:
            self._report_seq = rseq
            try:
                self._report = json.loads(report)
            except ValueError:
                self._report = None
        now = time.monotonic()
        out.update({
            "session": schema.SESSION_NAMES.get(c.session, str(c.session)),
            "wait": schema.WAIT_NAMES.get(c.wait, str(c.wait)),
            "bus": schema.BUS_NAMES.get(c.bus, str(c.bus)),
            "fault": fault if seq else "",
            "refusal": c.refusal()[1],
            "intervene": c.intervene, "k_exec": c.k_exec,
            "report": self._report,
        })
        st = self.rings["state"].latest()
        if st is not None:
            out.update({
                "measured": _floats(st["q"]), "command": _floats(st["q_cmd"]),
                "age_s": round(now - float(st["t"]), 3),
                "temp": _floats(st["temp"]), "load": _floats(st["load"]),
                "alarm": [int(x) for x in st["alarm"]],
            })
        tg = self.rings["targets"].newest_full()
        if tg is not None:
            out.update({"target": _floats(tg["q"]), "target_pose": _floats(tg["pose"]),
                        "target_err_mm": None if not math.isfinite(float(tg["err"]))
                        else round(float(tg["err"]) * 1e3, 2),
                        "target_age_s": round(now - float(tg["deadline"]), 3)})
        m = self.rings["intervene"].meta()
        t_last = float(m["t_last_sample"])
        out["phone"] = {
            "samples": int(m["samples"]), "fresh": bool(m["fresh"]),
            "engaged": bool(m["engaged"]), "track": int(m["track"]),
            "world_epoch": int(m["world_epoch"]), "released": int(m["released"]),
            "latency_ms": round(float(m["latency_s"]) * 1e3, 1)
            if math.isfinite(float(m["latency_s"])) and m["samples"] else None,
            "age_s": round(now - t_last, 3) if t_last > 0 else None,
        }
        return out

    def request(self, name: str) -> bool:
        if self.rings is None or name not in REQUESTS:
            return False
        self.rings["control"].request = REQUESTS[name]
        return True


def make_handler(source: Source, robot: dict, meshes: list[Path]):
    robot_json = json.dumps(robot).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "elrobot-dashboard"

        def log_message(self, fmt, *args):          # quiet: the terminal belongs to P0
            pass

        def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK, cache=False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if path.startswith("/static/"):
                target = (STATIC / path[len("/static/"):]).resolve()
                if STATIC not in target.parents:
                    return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return self._file(target, cache="/vendor/" in path)
            if path == "/api/robot":
                return self._send(robot_json, "application/json")
            if path == "/api/state":
                return self._json(source.state())
            if path.startswith("/mesh/"):
                try:
                    f = meshes[int(path[len("/mesh/"):])]
                except (ValueError, IndexError):
                    return self._json({"error": "no such mesh"}, HTTPStatus.NOT_FOUND)
                return self._send(f.read_bytes(), "model/stl", cache=True)
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/request":
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                name = str(json.loads(self.rfile.read(n) or b"{}").get("request", "")).upper()
            except (ValueError, AttributeError):
                return self._json({"ok": False, "error": "bad request"}, HTTPStatus.BAD_REQUEST)
            ok = source.request(name)
            return self._json({"ok": ok, "request": name},
                              HTTPStatus.OK if ok else HTTPStatus.CONFLICT)

        def _file(self, f: Path, cache=False):
            if not f.is_file():
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            ctype = {".js": "text/javascript", ".css": "text/css",
                     ".html": "text/html; charset=utf-8"}.get(
                f.suffix, mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            self._send(f.read_bytes(), ctype, cache=cache)

    return Handler


def serve(source: Source, host: str, port: int) -> ThreadingHTTPServer:
    """Start serving in a background thread; returns the server (`.shutdown()` to stop)."""
    robot, meshes = describe(source.cfg.urdf)
    httpd = ThreadingHTTPServer((host, port), make_handler(source, robot, meshes))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard").start()
    return httpd


def run_session(shm_name: str, cfg, *, dry_run: bool = False) -> None:
    """P10 as P0 starts it: serve the live arena until the session is STOPPED (and a few
    seconds after, so the page shows the end)."""
    from elrobot_deploy.runtime.shared_memory import Arena, build, open_all
    arena = Arena.attach(shm_name, build())
    httpd = None
    try:
        rings = open_all(arena)
        try:
            httpd = serve(Source(cfg, rings, dry_run=dry_run), cfg.dashboard.host,
                          cfg.dashboard.port)
        except OSError as exc:
            print(f"[P10] the dashboard could not start on {cfg.dashboard.host}:"
                  f"{cfg.dashboard.port}: {exc}. Is another session (or `view`) running? "
                  f"Change it with --set dashboard.port=...", flush=True)
            return
        control = rings["control"]
        while control.session < schema.STOPPED:
            time.sleep(0.1)
        time.sleep(2.0)
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        arena.close()


def run_view(cfg) -> None:
    """`elrobot_deploy view`: the URDF and the profile's poses, no robot, until Ctrl-C."""
    httpd = serve(Source(cfg), cfg.dashboard.host, cfg.dashboard.port)
    print(f"viewer: http://{cfg.dashboard.host}:{cfg.dashboard.port}  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()


__all__ = ["Source", "make_handler", "run_session", "run_view", "serve"]
