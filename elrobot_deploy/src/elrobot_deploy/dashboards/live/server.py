"""P10 — the live dashboard's server: the routes and what they read and write.
`docs/deploying_a_policy/running_and_live_dashboard.md`, "The live dashboard".

**Reads** the whole arena. **Writes** three things, all operator requests:
`control.request` (Start/Reset/Exit), `requested_config` (the Deploy settings
panel) and the prompt request. None of them is in force until P0 accepts a
START: P0 is the only process that reads `requested_config`, and it activates
it — with the prompt — only after validating both. Settings are validated
here too, before storing, so the panel gets its answer at once; the prompt
can only be judged by the policy, which lives in P5, so its verdict comes back
through the arena a dummy inference later and the page polls for it.

`http.server` from the standard library, as rollout inspection uses: this is
one operator on one browser looking at one robot, and a framework would buy
concurrency nobody needs against a dependency list that is deliberately short
(`pyproject.toml`). `ThreadingHTTPServer` matters more here than there —
the camera stream and the state poll are both long-lived and must not queue
behind each other.

**Three channels, because they want opposite things.**

* **Commands** go up as an ordinary POST. Rare, tiny, and the only thing on this
  server that is allowed to change anything. It writes `request` and returns;
  P0 sees the word within its next poll, ~1 ms later.
* **Cameras** come down as `multipart/x-mixed-replace` — MJPEG, which a browser
  renders natively in an `<img>` with no JavaScript and no decode on either
  side. P2 already keeps the compressed bytes the robot sent beside the pixels
  it decoded, so this is a passthrough.
* **State** is POLLED by the browser, not pushed. That is a deliberate choice
  about backpressure: a push would keep writing into a socket the tunnel cannot
  drain, and the lag would grow without bound while the operator watched an
  older and older robot. A poll cannot get behind — a slow link lowers the frame
  rate and nothing else — and it keeps the command POST from queueing behind a
  stream that is saturating an SSH connection they share.

**One view.** The page is the 3-D scene, the cameras, the controls, the
prompt, the Deploy settings and a status line. Deeper questions (IK residuals,
servo tracking, timing) are answered offline over the recording
(`elrobot_deploy inspect`), not on the box during a run.

**Always the newest, never a queue.** Every read here takes the latest value out
of the arena at the instant it is asked. Nothing is buffered for the browser and
nothing is retried: a dropped dashboard frame is a frame the operator does not
need, because a newer one is 33 ms behind it.

**Bind localhost.** The operator's browser is on a laptop and the runtime is on
the deploy box, and the way across is the SSH session that is already open:
`ssh -L 8080:localhost:8080 <deploy box>`, then `localhost:8080` in the browser.
Binding `0.0.0.0` and browsing to the LAN address instead exposes the one
server that can start the robot to the whole network, and depends on the
laptop's route to the robot subnet. Loopback over a tunnel has neither problem.
"""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.dashboards.web.http import COMMON_STATIC as SHARED_STATIC
from elrobot_deploy.dashboards.web.pack import ALIGN as _ALIGN
from elrobot_deploy.dashboards.web.pack import pack as _pack
from elrobot_deploy.dashboards.web.pack import spans as _spans
from elrobot_deploy.config_loading.rollout_settings import (LAUNCH_BLEND, LAUNCH_IK,
                                                        RolloutSettings, SettingsError)
from elrobot_deploy.config_loading.rollout_settings import check as check_settings

STATIC = Path(__file__).resolve().parent / "static"

#: Shared with the inspection dashboards, and shared on purpose: `gl.js` and
#: `marks.js` are the renderer this dashboard reuses wholesale, so they are
#: served from where they live (`dashboards/web/static`) rather than copied
#: into a second static directory that would drift.

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}

#: The MJPEG boundary. Any token works; it only has to not appear in a JPEG.
_BOUNDARY = "g1dframe"

#: How long the stream loop sleeps when the ring holds nothing it has not
#: already sent. A camera publishes at 30 Hz, so this wakes a few times per
#: frame — enough that the stream is never the reason a frame is late, and
#: coarse enough not to spin a core against a camera that has stopped.
_IDLE_S = 0.005

#: The requests the browser may send, by name. A dict rather than
#: `getattr(schema, ...)` so a typo in a POST body cannot reach for an arbitrary
#: attribute of the schema module.
_REQUESTS = {
    "start": schema.REQ_START,
    "reset": schema.REQ_RESET,
    "exit": schema.REQ_EXIT,
}


def _listed(v):
    """A tuple of numbers as a JSON list, or None."""
    return None if v is None else [float(x) for x in v]


class Busy(RuntimeError):
    """A settings or prompt request while a rollout is ACTIVE (HTTP 409)."""


class Live:
    """Everything the browser can ask about, read from the arena on demand.

    Holds no history and no cache. The one piece of state is `scene`, which is
    the compiled model, and that is not about the robot at all.

    `intervene` is whether this launch runs `--intervene`, which narrows the
    settings a rollout may switch to (`rollout_settings.check`).
    """

    def __init__(self, cfg, hw, rings, scene, *, deploy_py=None, step=None,
                 intervene: bool = False):
        self.cfg, self.hw, self.rings, self.scene = cfg, hw, rings, scene
        self.deploy_py, self.step = deploy_py, step
        self.intervene = bool(intervene)
        self._lock = threading.Lock()
        #: `scene` poses into MjData it owns, and `mj_forward` releases the GIL,
        #: so two requests posing at once corrupt MjData's stack and segfault
        #: the process — which is how P10 died at READY with a second tab open.
        self._scene_lock = threading.Lock()

    # --- the cameras --------------------------------------------------------

    def cameras(self) -> list[dict]:
        """Which cameras exist, what each one is FOR, and the order to show them.

        Fixed for the life of the process — the camera set is `cfg.cameras`,
        which the arena's layout is built from — so the page fetches this once
        and builds its panels from it.

        **Names alone are not enough.** `cfg.cameras` is what EXISTS
        and `cfg.observed` is what the policy actually reads; a camera may be
        recorded on every run and read by nothing. A pane captioned only
        `head` sits beside the two views the arm is being driven from and looks
        exactly like them, which is the one thing an operator watching a live
        run must not be misled about. So each camera says whether it was
        observed, whether it is the primary — the camera whose frame becomes
        `t_anchor` — and whether one frame carries two eyes.

        `slot` is always None live: the infra has no notion of which input of
        a policy a camera feeds, so the pane says "used". (Rollout inspection
        fills it from a record that carries one.)

        Observed cameras first, then record-only, alphabetical within each —
        the same order rollout inspection shows, so the live view and the
        replay of the same run do not disagree about which pane is which.
        """
        cfg = self.cfg
        # One pane per ROBOT-FILE camera, streaming its wire JPEG: the split
        # head is shown whole, both eyes, and counts as used / primary when
        # either of its views is (the views are what the policy
        # names; the pane is what the robot sends).
        views = cfg.views
        observed = {views[v].source if v in views else v for v in cfg.observed}
        primary = views[cfg.primary_camera].source \
            if cfg.primary_camera in views else cfg.primary_camera
        names = sorted(self.rings["camera_sources"],
                       key=lambda n: (n not in observed, n))
        return [{"name": n,
                 "slot": None,
                 "used": n in observed,
                 "primary": n == primary,
                 "binocular": bool(cfg.cameras[n].binocular),
                 # So the page can reserve the right box and letterbox into it
                 # rather than cropping a frame to fill a pane.
                 "height": cfg.cameras[n].height,
                 "width": cfg.cameras[n].width,
                 # A fisheye's lens square and circle as fitted at launch,
                 # outlined over the pane (the stream is the whole frame).
                 "crop": _listed(getattr(cfg.cameras[n], "crop", None)),
                 "lens_circle": _listed(getattr(cfg.cameras[n], "lens_circle", None))}
                for n in names]

    # --- the banner ---------------------------------------------------------

    def banner(self) -> dict:
        """What the operator is looking at, read from the arena every time.

        Never from what the dashboard believes it sent. That is the whole reason
        a dropped `request` is tolerable (see `Control.request`): the phase and
        the pending request are both READ BACK, so a command that lost a race
        shows up as a banner that did not change, and the operator presses it
        again. A UI that rendered its own optimistic state would hide exactly the
        failure it caused.
        """
        control = self.rings["control"]
        session = control.session
        request = control.request
        return {
            "session": int(session),
            "session_name": schema.SESSION_NAMES.get(int(session), "?"),
            "request": int(request),
            "request_name": schema.REQUEST_NAMES.get(int(request), "?"),
            "k_exec": int(control.k_exec),
            "warm": int(control.warm),
            "newest_chunk_id": int(control.newest_chunk_id),
            # Which policy is being served (and which checkpoint, if any). Free
            # — it is already in `cfg` — and it closes the failure of finishing
            # a session unsure what produced the rollouts just recorded.
            "deployment": Path(self.deploy_py).parent.name if self.deploy_py
                          else "",
            "step": self.step,
            "checkpoint": self.cfg.checkpoint,
            # The prompt IN USE — the active block's echo, not the
            # launch config's, which is only the first rollout's default.
            "prompt": control.active()[1],
            "generation": int(control.rollout_generation),
            "rollout": self.rollout(),
        }

    def rollout(self) -> dict:
        """The status line's settings: what the current (or last) rollout runs
        with — the control region's ACTIVE block, never the request."""
        from elrobot_deploy.runtime.ik import SOLVERS
        active_rec, _ = self.rings["control"].active()
        s = RolloutSettings.from_record(active_rec).as_dict()
        if s["ik"] == LAUNCH_IK:
            s["ik"] = self.cfg.ik
        if s["blend"] == LAUNCH_BLEND:
            s["blend"] = self.cfg.blend
        s["ik_mode"] = SOLVERS[s["ik"]][1] if s["ik"] in SOLVERS else "custom"
        if self.cfg.policy_action_mode == "joint":
            # Joint actions are the solution: P6 runs no IK in joint mode.
            s["ik"], s["ik_mode"] = "not used", "joint actions"
        return s

    # --- settings and prompt (between rollouts) -----------------------------

    def settings(self) -> dict:
        """The Deploy settings panel and the prompt field, read back.

        `requested` is what the operator asked for, `active` what the current
        (or last) rollout ran with; they differ exactly until the next START
        is accepted. `prompt.verdict.of` names the request P5 judged, so a
        page that submitted request `seq` knows the verdict is its own only
        when the two match. `refusal` is why P0 last refused a START.
        """
        control = self.rings["control"]
        active_rec, active_prompt = control.active()
        seq, text = control.prompt_request()
        of, verdict, reason = control.verdict()
        refusal_seq, refusal = control.refusal()
        applied = control.settings_applied
        def named(rec) -> dict:
            # The record carries a launch-named `module:factory` solver as
            # "the launch's own"; the page shows its name.
            d = RolloutSettings.from_record(rec).as_dict()
            if d["ik"] == LAUNCH_IK:
                d["ik"] = self.cfg.ik
            if d["blend"] == LAUNCH_BLEND:
                d["blend"] = self.cfg.blend
            return d

        return {
            "requested": named(control.requested_config()),
            "active": named(active_rec),
            "launch": RolloutSettings.of(self.cfg).as_dict(),
            "generation": int(control.rollout_generation),
            "applied": int(applied & ~schema.APPLIED_FAILED),
            "applied_failed": bool(applied & schema.APPLIED_FAILED),
            "editable": int(control.session) != schema.ACTIVE,
            # What the panel offers. `ik` is every solver the schema knows,
            # plus this launch's own `module:factory` if it named one; the
            # ones this launch cannot run (a waist solver in async, anything
            # but arm_qp under --intervene) are refused on submit, by name.
            "options": {"ik": list(schema.IK_OPTIONS) + (
                [self.cfg.ik] if self.cfg.ik not in schema.IK_OPTIONS else []),
                        # The async blend: the shipped ones, plus this
                        # launch's own `module:function` if it named one.
                        "blend": list(schema.BLEND_OPTIONS) + (
                [self.cfg.blend] if self.cfg.blend not in schema.BLEND_OPTIONS else [])},
            "limits": {"replan_period": [2, int(self.cfg.H)],
                       "horizon": int(self.cfg.H),
                       "prompt_bytes": schema.PROMPT_BYTES},
            "intervene": self.intervene,
            # Joint actions: P6 runs no IK, so the panel shows "not used".
            "joint_actions": self.cfg.policy_action_mode == "joint",
            "prompt": {
                "active": active_prompt,
                "request": {"seq": int(seq), "text": text},
                "verdict": {"of": int(of),
                            "name": schema.PROMPT_VERDICT_NAMES.get(verdict, "?"),
                            "reason": reason},
            },
            "refusal": {"seq": int(refusal_seq), "text": refusal},
        }

    def request_settings(self, changes: dict) -> dict:
        """Store a settings request. Validated first, and refused while ACTIVE.

        `changes` is a partial `RolloutSettings.as_dict`, applied over the
        CURRENT request, so the panel may send only what it changed. It is
        checked with the same `rollout_settings.check` P0 runs at START; P0 checks
        again, because it is the authority and this is only the fast answer.
        """
        with self._lock:
            control = self.rings["control"]
            if int(control.session) == schema.ACTIVE:
                raise Busy("a rollout is running: settings can change only "
                           "between rollouts (Reset first)")
            base = RolloutSettings.from_record(control.requested_config())
            new = base.updated(changes)
            check_settings(new, self.cfg, self.hw, intervene=self.intervene)
            control.request_config(new.to_record())
        return self.settings()

    def request_prompt(self, text) -> dict:
        """Store a prompt request. Returns its `seq`, for the page to match
        against P5's verdict (`settings()["prompt"]["verdict"]["of"]`).

        Only the arena's size and emptiness are checked here — the token limit
        is the policy's (`Policy.check_prompt_length`), and it is judged in P5.
        """
        if not isinstance(text, str) or not text.strip():
            raise SettingsError("the prompt is empty")
        with self._lock:
            control = self.rings["control"]
            if int(control.session) == schema.ACTIVE:
                raise Busy("a rollout is running: the prompt can change only "
                           "between rollouts (Reset first)")
            try:
                seq = control.request_prompt(text)
            except ValueError as exc:
                raise SettingsError(f"prompt too long: {exc}") from None
        return {"seq": seq, "text": text}

    # --- the robot ----------------------------------------------------------

    def _target_at(self, k: int):
        """The body target for the tick being executed, or None.

        `k_exec` names the tick P7 is driving toward, so this is the
        command in force right now — which is what the operator means by "what
        is it being told to do".
        """
        return self.rings["body"].full(k)

    def frame(self) -> tuple[dict, bytes]:
        """One dashboard frame: a JSON header and the float32 payload it maps.

        Serialised on `_scene_lock` because every request gets its own thread
        and posing is not reentrant — see `__init__`. One tab never overlaps
        itself (`app.js` awaits each poll), so it takes a second client to race:
        a tab left open from the previous run, or one over the tunnel.
        """
        with self._scene_lock:
            return self._frame_locked()

    def _frame_locked(self) -> tuple[dict, bytes]:
        """`frame`'s body; the caller holds `_scene_lock`.

        The overlays are absent rather than zeroed when they do not apply. A
        zeroed transform is a robot at the origin, which draws; an absent one
        draws nothing, and "nothing is being commanded" is the true statement
        during HOME, READY and every P0 ramp.
        """
        control = self.rings["control"]
        session = int(control.session)
        k_exec = int(control.k_exec)

        posed = self.scene.measured()
        if posed is None:
            return {"ready": False, "banner": self.banner()}, b""
        transforms, tcp, q, grip = posed

        parts: list[tuple[str, np.ndarray]] = [
            ("transforms", transforms), ("tcp", tcp),
        ]
        meta: dict = {
            "ready": True,
            "banner": self.banner(),
            "measured": {
                # The numbers themselves, for the panel that reads them out.
                # Cheap next to the geometry and worth having when a joint is
                # the thing in question.
                "q_arm": [float(v) for v in q[list(self.hw.joints_arm)]],
                "q_waist": [float(v) for v in q[list(self.hw.joints_waist)]],
                "grip": [float(v) for v in grip],
            },
            "solved": None,
            "ghost": None,
        }

        # Only inside a rollout. During HOME and READY the ring may still hold
        # P0's ramp — real targets, honestly written — but the operator asked for
        # measured only outside a run, and a commanded overlay during homing
        # would show the arm chasing a ramp nobody chose.
        target = self._target_at(k_exec) if session == schema.ACTIVE else None
        if target is not None:
            s_transforms, s_tcp, jaws = self.scene.solved(target)
            parts += [("solved", s_transforms), ("solved_tcp", s_tcp)]
            meta["solved"] = {
                "k": int(target["k"]),
                "err": (float(target["err"])
                        if np.isfinite(target["err"]) else None),
                "q": [float(v) for v in np.asarray(target["q"], dtype=float)],
            }
            ghosted = self.scene.ghost(target, jaws)
            if ghosted is None:
                # A joint-mode chunk has no Cartesian target: its
                # `pose` is NaN by contract, so there is no commanded gripper
                # to draw — the joint targets above ARE the command. Said,
                # rather than drawn as nothing, so an empty overlay is not
                # read as a broken one.
                meta["ghost_absent"] = ("no Cartesian target for this tick "
                                      "(joint-mode actions, or a P0 ramp)")
            else:
                g_transforms, g_tcp = ghosted
                parts += [("ghost", g_transforms), ("ghost_tcp", g_tcp)]
                meta["ghost"] = {
                    "pose": np.asarray(target["pose"],
                                       dtype=float).ravel().tolist(),
                }

        meta["layout"] = _spans(parts)
        return meta, _pack(parts)

    # --- the one thing this process writes ----------------------------------

    def request(self, name: str) -> dict:
        """Store an operator request. The only write P10 makes to the arena.

        Serialised on a lock because `ThreadingHTTPServer` gives every request
        its own thread and a double-click is two of them: without it, two stores
        could interleave and the second could be lost while its POST reported
        success. The lock makes this process a single writer of the word even
        though it is a threaded server.
        """
        verb = _REQUESTS.get(name)
        if verb is None:
            raise KeyError(name)
        with self._lock:
            self.rings["control"].request = verb
        return {"stored": name, "request": verb}


class Handler(BaseHTTPRequestHandler):
    server_version = "elr-dashboard"
    #: HTTP/1.1, so the MJPEG stream and the poll can share a keep-alive
    #: connection instead of tearing one down per frame.
    protocol_version = "HTTP/1.1"
    #: TCP_NODELAY. A reply is two writes (headers, then body) on a keep-alive
    #: connection; with Nagle on, a body whose last segment is short waits for
    #: the browser's delayed ACK — a steady ~40 ms per head-camera frame here,
    #: against ~1 ms for the wrist frames, whose sizes happened to dodge it.
    disable_nagle_algorithm = True

    @property
    def live(self) -> Live:
        return self.server.live

    def log_message(self, fmt, *args):
        """Silence. A 15 Hz poll would otherwise bury P0's log — the one place
        the operator reads what the session is doing — under access lines."""

    # --- plumbing -----------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The dashboard is live by definition; a cached frame is a wrong one.
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _static(self, rel: str) -> None:
        # This dashboard's files, then the ones shared with inspection.
        for root in (STATIC, SHARED_STATIC):
            path = (root / rel).resolve()
            if root not in path.parents or not path.is_file():
                continue
            self._send(200, path.read_bytes(),
                       _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
            return
        self._json({"error": f"no such file: {rel}"}, 404)

    # --- routes -------------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/banner":
                return self._json(self.live.banner())
            if path == "/api/settings":
                return self._json(self.live.settings())
            if path == "/api/meta":
                # The model plus the cameras, each saying what it is for. Both
                # are fixed for the life of the process, so the page fetches
                # this once and builds its panels from it rather than
                # hardcoding names a robot file is free to change.
                return self._json({
                    **self.live.scene.meta(),
                    "cameras": self.live.cameras(),
                })
            if path == "/api/scene.bin":
                return self._scene_bin()
            if path == "/api/frame":
                return self._frame()
            if path.startswith("/api/stream/"):
                return self._stream(path[len("/api/stream/"):])
            return self._json({"error": f"no such route: {path}"}, 404)
        except BrokenPipeError:
            # The operator closed the tab, or the tunnel dropped, mid-write.
            # Routine for a stream, and nothing to report.
            pass
        except Exception:                                 # noqa: BLE001
            traceback.print_exc()
            try:
                self._json({"error": traceback.format_exc()}, 500)
            except Exception:                             # noqa: BLE001
                pass

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path not in ("/api/command", "/api/settings", "/api/prompt"):
                return self._json({"error": f"no such route: {path}"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "the body is not JSON"}, 400)
            if not isinstance(body, dict):
                return self._json({"error": "the body must be an object"}, 400)
            if path == "/api/settings":
                return self._guarded(self.live.request_settings, body)
            if path == "/api/prompt":
                return self._guarded(self.live.request_prompt,
                                     body.get("prompt"))
            try:
                return self._json(self.live.request(str(body.get("request"))))
            except KeyError:
                return self._json(
                    {"error": f"unknown request {body.get('request')!r}; "
                              f"expected one of {sorted(_REQUESTS)}"}, 400)
        except BrokenPipeError:
            pass
        except Exception:                                 # noqa: BLE001
            traceback.print_exc()
            try:
                self._json({"error": traceback.format_exc()}, 500)
            except Exception:                             # noqa: BLE001
                pass

    def _guarded(self, fn, arg) -> None:
        """A settings/prompt request: 409 while ACTIVE, 400 with the reason
        when it is refused, else the new state."""
        try:
            return self._json(fn(arg))
        except Busy as exc:
            return self._json({"error": str(exc)}, 409)
        except (ConfigError, ValueError) as exc:
            return self._json({"error": str(exc)}, 400)

    def _scene_bin(self) -> None:
        s = self.live.scene
        body = _pack([("vertices", s.vertices), ("normals", s.normals),
                      ("indices", s.indices)])
        layout = _spans([("vertices", s.vertices), ("normals", s.normals),
                         ("indices", s.indices)])
        self._send(200, body, "application/octet-stream",
                   {"X-Layout": json.dumps(layout)})

    def _frame(self) -> None:
        """A JSON header and its payload in one response.

        One request rather than two so the header and the bytes it describes
        cannot come from different instants — the robot moves between them, and
        a `layout` describing an older frame's arrays is a picture of a robot
        that never existed.
        """
        meta, payload = self.live.frame()
        head = json.dumps(meta).encode()
        # PAD the header to a 4-byte boundary. `layout`'s offsets are relative to
        # the payload, and the browser reads them with
        # `new Float32Array(buffer, byteOffset, length)` — which THROWS on a
        # byteOffset that is not a multiple of four. The JSON's length is
        # arbitrary, so without this the payload starts wherever the header
        # happened to end and three frames in four fail to parse. `_pack` already
        # aligns each section within the payload; this keeps the payload itself
        # aligned, so the two compose.
        pad = -(4 + len(head)) % _ALIGN
        body = (len(head).to_bytes(4, "little") + head + b"\0" * pad + payload)
        self._send(200, body, "application/octet-stream")

    def _stream(self, camera: str) -> None:
        """MJPEG for one camera, newest frame always.

        The loop reads the ring only when the socket is ready for another frame,
        so a slow link makes the operator's view LOWER RATE rather than OLDER:
        whatever was published while the last write drained is skipped, and what
        goes out next is what the robot sent most recently. That is the whole
        reason this reads inside the loop rather than subscribing to a queue.
        """
        regions = self.live.rings["camera_sources"]
        region = regions.get(camera)
        if region is None:
            return self._json(
                {"error": f"no such camera {camera!r}; have {sorted(regions)}"},
                404)

        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        # This response has no length and never ends, so under HTTP/1.1 its
        # framing has to be stated: without `Connection: close` a strict client
        # is entitled to expect `Content-Length` or chunked encoding, and what it
        # does instead of erroring is wait. Browsers happen to read
        # `multipart/x-mixed-replace` to end-of-connection regardless; anything
        # else talking to this route would hang on the first frame.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        import time
        last_id = -1
        while not self.server.stopping.is_set():
            got = region.read_jpeg()
            if got is None:
                time.sleep(_IDLE_S)
                continue
            _, frame_id, payload = got
            if frame_id == last_id:
                # Nothing new yet. Sleeping beats spinning at 500 Hz on a
                # camera that publishes at 30.
                time.sleep(_IDLE_S)
                continue
            last_id = frame_id
            head = (f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n").encode()
            self.wfile.write(head + payload + b"\r\n")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, live: Live):
        super().__init__(addr, Handler)
        self.live = live
        #: Set when P10 is shutting down, so a camera stream blocked in its own
        #: loop leaves instead of holding the process open.
        self.stopping = threading.Event()


def serve(live: Live, *, host: str = "127.0.0.1", port: int = 8080) -> Server:
    """Start the server on a thread and return it.

    `127.0.0.1`, not `0.0.0.0`, and that is the default rather than a suggestion
    — see the module docstring. Reaching this from another machine is a port forward on
    the SSH session that is already open.
    """
    server = Server((host, port), live)
    threading.Thread(target=server.serve_forever, name="p10-http",
                     daemon=True).start()
    return server
