"""Rollout inspection: the local web app over `discover`, `loader`, `timeline`
and `scene`, plus a MODE that owns the third panel.

    elrobot_deploy inspect <root-or-record> [--port 8770]

**One rollout is held at a time, in memory, and nothing is cached to disk.**
Pressing Load drops the previous rollout — closing its blob handles — and builds
the next. A rollout costs about half a second of MuJoCo plus the read, which is
cheaper than owning a cache that can go stale against a record re-recorded under
the same name.

**Binary where it matters.** Geometry and transforms go out as raw little-endian
float32/uint32 with their layout described in the JSON that precedes them, not
as JSON numbers: the transform buffer alone is several megabytes. JPEG frames
are served as-is from the blob — P9 stored exactly what the robot sent, so
there is no decode on either side.

**Modes.** The page's first two sections — the cameras, and the robot with its
measured / commanded / TCP switches — are the same whatever the question. The
third section is the question (`modes/__init__.py`): each mode mounts its own
routes on the same `App`, is told when a rollout loads, and ships its own
`mode_<name>.js` for the panel. With no mode chosen there is no third section.

For the user: `docs/deploying_a_policy/rollout_inspection.md` ("The view",
"Writing a mode").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from elrobot_deploy.dashboards.web import http
from elrobot_deploy.dashboards.web.pack import pack as _pack
from elrobot_deploy.dashboards.web.pack import spans as _spans
from elrobot_deploy.dashboards.inspect import discover, loader
from elrobot_deploy.dashboards.inspect import events as events_mod
from elrobot_deploy.dashboards.inspect import modes as modes_mod
from elrobot_deploy.dashboards.inspect import scene as scene_mod
from elrobot_deploy.dashboards.inspect import timeline as timeline_mod

STATIC = Path(__file__).resolve().parent / "static"


@dataclass
class Loaded:
    """Everything the browser can ask about the current rollout."""

    rollout: loader.Rollout
    timeline: timeline_mod.Timeline
    scene: scene_mod.Scene
    #: The record's path relative to the discovery root (`discover.Recording.key`).
    key: str
    #: The replan marks. Built once at load beside the timeline, because they
    #: are the same size and shape of work — a pass over one stream — and
    #: `/api/state` would otherwise recompute them on every poll.
    marks: events_mod.Marks

    def _scene_parts(self) -> list[tuple[str, np.ndarray]]:
        s = self.scene
        return [("vertices", s.vertices), ("normals", s.normals),
                ("indices", s.indices)]

    def _transform_parts(self) -> list[tuple[str, np.ndarray]]:
        """The per-frame arrays, in the order they are packed.

        One list, read by both the packer and the layout map, because the two
        going out of step is a class of bug that produces a robot rather than an
        error: every geom drawn with some other geom's transform still renders.
        """
        s = self.scene
        return [
            ("transforms", s.transforms), ("tcp", s.tcp),
            ("ghost", s.ghost), ("ghost_tcp", s.ghost_tcp),
            ("ghost_valid", s.ghost_valid),
            ("solved", s.solved), ("solved_tcp", s.solved_tcp),
            ("solved_valid", s.solved_valid),
            # The solved overlay posed on the IK's own trunk instead of the
            # commanded one. Zero-length on a record that predates `waist_ik`,
            # which costs a span of `[offset, 0]` and tells the browser exactly
            # what it needs to know — no separate flag to keep in step, and no
            # section it has to guess the presence of.
            ("solved_pinned", s.solved_pinned),
            ("solved_pinned_tcp", s.solved_pinned_tcp),
            # The re-solved overlays, one set of three sections per method the
            # IK bench solved for. Named by tier rather than numbered so
            # that a recording solved for a subset — or for a tier this build
            # does not know about — still lines its sections up by name instead
            # of by position.
            #
            # Absent entirely when there is no sidecar, which the browser reads
            # off `meta.wbik_tiers` being empty. A missing section and an empty
            # one are the same thing to it; there is no third state to handle.
            *[part
              for tier in s.wbik
              for part in ((f"wbik_{tier}", s.wbik[tier]),
                           (f"wbik_{tier}_tcp", s.wbik_tcp[tier]),
                           (f"wbik_{tier}_err", s.wbik_err[tier]))],
            *([("wbik_valid", s.wbik_valid),
               ("wbik_off_err", s.wbik_off_err)] if s.wbik else []),
        ]

    def scene_bin(self) -> bytes:
        return _pack(self._scene_parts())

    def transforms_bin(self) -> bytes:
        return _pack(self._transform_parts())

    def describe(self) -> dict:
        s = self.scene
        return {
            "key": self.key,
            "name": self.rollout.path.name,
            "dir": str(self.rollout.path),
            "summary": self.rollout.summary(),
            "timeline": timeline_mod.describe(self.timeline, self.rollout),
            "events": self.marks.as_json(self.timeline),
            "scene": s.meta(),
            # Byte layout of the two binary endpoints. Sent rather than implied
            # so the browser never hardcodes an offset this file could change.
            "layout": {
                "scene_bin": _spans(self._scene_parts()),
                "transforms_bin": _spans(self._transform_parts()),
            },
            # Names AND roles. The browser draws one pane per camera and has to
            # caption it, and "which slot did this fill, did the policy see it,
            # is it the anchor" is a property of the record rather than of the
            # viewer — a camera can be recorded and read by nothing, and a pane
            # that did not say so would read as an input.
            "cameras": [
                {"name": c.name, "slot": c.slot, "used": c.used,
                 "primary": c.primary, "binocular": c.binocular,
                 "height": c.height, "width": c.width,
                 "crop": c.crop, "lens_circle": c.lens_circle}
                for c in self.rollout.cameras.values()
            ],
        }

    def close(self) -> None:
        self.rollout.close()


class State:
    """The single loaded rollout, behind a lock. `record_dir` is the directory
    it came from, which is where a mode looks for that rollout's sidecars."""

    def __init__(self, root: Path):
        self._lock = threading.Lock()
        self._loaded: Loaded | None = None
        #: The discovery root (`Options.root`): a directory, or one record.
        self.root = Path(root)
        self.record_dir: Path | None = None
        self._listeners: list = []

    @property
    def current(self) -> Loaded | None:
        return self._loaded

    def on_load(self, fn) -> None:
        """`fn(loaded, record_dir)` after every successful load."""
        self._listeners.append(fn)

    def load(self, key: str, fps: float, clock: str = "device") -> Loaded:
        path = discover.resolve(key, self.root)
        with self._lock:
            # Built before the old one is dropped so a failure leaves the
            # previous rollout intact rather than leaving the UI with nothing.
            rollout = loader.load(path)
            try:
                tl = timeline_mod.build(rollout, fps, clock)
                sc = scene_mod.build(rollout, tl)
                mk = events_mod.build(rollout, tl)
            except Exception:
                rollout.close()
                raise
            if self._loaded is not None:
                self._loaded.close()
            self._loaded = Loaded(rollout, tl, sc, key, mk)
            self.record_dir = path
            for fn in self._listeners:
                fn(self._loaded, path)
            return self._loaded

    def close(self) -> None:
        with self._lock:
            if self._loaded is not None:
                self._loaded.close()
                self._loaded = None
            self.record_dir = None


def build_app(options, modes: list[str] | None = None) -> http.App:
    """The dashboard: the viewer's routes, then every mode's, on one `App`.

    `options` is an `options.Options` (or a root path, which is
    wrapped in default options). `modes` limits which modes are mounted,
    as names of built-ins or `pkg.module:Class` specs (default: the
    built-ins, then `options.modes`); a mode that fails to import is reported
    and left out rather than taking the viewer down with it.
    """
    from elrobot_deploy.dashboards.inspect.options import Options

    if not isinstance(options, Options):
        options = Options(root=Path(options).expanduser().resolve())
    state = State(options.root)
    app = http.App("rollout inspection", static=[STATIC, http.COMMON_STATIC])
    app.state = state
    app.options = options
    app.on_close(state.close)

    @app.get("/api/tree")
    def tree(req):
        req.json(discover.as_json(discover.scan(options.root), options.root))

    @app.get("/api/state")
    def api_state(req):
        cur = state.current
        req.json(cur.describe() if cur else {"loaded": False})

    @app.get("/api/scene.bin")
    def scene_bin(req):
        cur = state.current
        if cur is None:
            return req.json({"error": "nothing loaded"}, 409)
        req.bytes(cur.scene_bin(), "application/octet-stream")

    @app.get("/api/transforms.bin")
    def transforms_bin(req):
        cur = state.current
        if cur is None:
            return req.json({"error": "nothing loaded"}, 409)
        req.bytes(cur.transforms_bin(), "application/octet-stream")

    @app.get("/api/img/")
    def image(req):
        """`/api/img/<camera>/<display frame>` -> the recorded JPEG.

        Indexed by DISPLAY frame, not by camera frame: the timeline owns the
        mapping, so the browser asks for "frame 604 of the run" and never has
        to know that the two cameras are on different frame numbers there.
        """
        cur = state.current
        if cur is None:
            return req.json({"error": "nothing loaded"}, 409)
        rest = req.path[len("/api/img/"):]
        cam_name, frame_s = rest.split("/", 1)
        frame = int(frame_s)
        cam = cur.rollout.cameras.get(cam_name)
        if cam is None:
            return req.json({"error": f"no camera {cam_name!r}"}, 404)
        frame = max(0, min(cur.timeline.n - 1, frame))
        i = cur.timeline.at(cam_name, frame)
        if i < 0:
            return req.json({"error": "no frame yet"}, 404)
        req.bytes(cam.jpeg(i), "image/jpeg", extra={
            "Cache-Control": "no-store",
            # Which camera frame this display frame resolved to. The UI shows
            # it, because the two wrists landing on different numbers is a
            # real property of the record and worth being able to see.
            "X-Frame-Index": str(int(cam.index["frame_id"][i])),
        })

    @app.post("/api/load")
    def load(req):
        body = req.body()
        cur = state.load(str(body["key"]),
                         float(body.get("fps", timeline_mod.DEFAULT_FPS)),
                         str(body.get("clock", "device")))
        req.json(cur.describe())

    mounted = modes_mod.mount(app, state, modes, options)

    @app.get("/api/modes")
    def api_modes(req):
        req.json([m.describe() for m in mounted])

    return app


def serve(options) -> None:
    """Run the dashboard until Ctrl-C (`options.Options`)."""
    http.serve(build_app(options), options.host, options.port)
