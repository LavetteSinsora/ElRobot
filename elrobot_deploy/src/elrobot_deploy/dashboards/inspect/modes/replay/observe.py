"""A policy's inputs, rebuilt from a record: the `Observation` and the `Reader`.

The live path is `policy_contract.observation.observe(anchor, prompt)` over the arena and
`policy_contract.Reader` over its rings. Offline there is no arena, so this builds the
same two objects from what P9 kept:

* **frames** — the recorded JPEG of each camera, decoded exactly as P2 decodes
  it (`cv2.imdecode(..., IMREAD_COLOR)`, BGR), cut to the policy's VIEW by
  columns (a side_by_side head is recorded whole) or, for a fisheye, to the
  square P2 cut (`fisheye_circle_fit.cut`), and converted to RGB by the
  infra's own `to_rgb`, so the colour path is the live one;
* **state** — the `state.bin` row the anchor rule picked, through the infra's
  own `State` conversion;
* **history** — `RecordReader` answers `states_between`, `frame_ids_between`
  and `frames_by_id` from the record, under the live reader's rules (declared
  streams only, closed windows, ascending) plus one more: nothing after the
  instant P5 finished acquiring (`t_obs_received`) is visible, because the live
  policy could not have read it.

`frame_id` is the recorded camera's own counter (`*.idx.bin`), which is what
the live ring numbers frames by; both eyes of a split camera share it.

Used by the replay producer (`run.py`), in the policy's interpreter. The live
objects it mirrors: `docs/deploying_a_policy/policy_contract.md`, "The
observation (`obs`)" and "History: `reader`".
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from elrobot_deploy.policy_contract.observation import (FrameTimes, Frame, Observation,
                                                    Reader, StateWindow, _state,
                                                    _state_window, to_rgb)
from elrobot_deploy.dashboards.inspect.modes.replay.anchors import AnchorSpec, View

#: Decoded source frames kept at once. A replay walks forward, so a handful
#: is enough; a history-reading policy re-reads the recent past.
_CACHE = 64


class RecordFrames:
    """Decoded frames of a record, per recorded camera and row, cut to views."""

    def __init__(self, record, views: dict[str, View]):
        self.record = record
        self.views = dict(views)
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def _decode(self, source: str, row: int) -> np.ndarray:
        key = (source, int(row))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        import cv2

        cam = self.record.cameras[source]
        img = cv2.imdecode(np.frombuffer(cam.jpeg(int(row)), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{source} frame row {row} did not decode")
        self._cache[key] = img
        while len(self._cache) > _CACHE:
            self._cache.popitem(last=False)
        return img

    def pixels(self, view: str, row: int) -> np.ndarray:
        """The view's pixels at the source camera's `row`: RGB, a fresh copy."""
        v = self.views[view]
        img = self._decode(v.source, row)
        if v.square is not None:
            from elrobot_deploy.runtime.drivers.fisheye_circle_fit import cut
            img = cut(img, *v.square)
        elif v.width is not None:
            img = img[:, v.x0:v.x0 + v.width]
        return to_rgb(img)

    def frame(self, view: str, row: int) -> Frame:
        cam = self.record.cameras[self.views[view].source]
        return Frame(pixels=self.pixels(view, row),
                     t_recv=float(cam.index["t_recv"][row]),
                     frame_id=int(cam.index["frame_id"][row]))


def observation(record, frames: RecordFrames, spec: AnchorSpec, prompt: str,
                views) -> Observation:
    """The observation P5 built at this replan, rebuilt."""
    out = {name: frames.frame(name, spec.frames[frames.views[name].source])
           for name in views}
    return Observation(t_anchor=float(spec.t_anchor), frames=out,
                       state=_state(record.state[spec.state_row]), prompt=prompt,
                       t_trigger=float(spec.t_trigger))


class RecordReader(Reader):
    """The history reader, served from the record.

    `now` is the causality bound: set it to the replan's `t_obs_received`
    before each `infer`, and no window reaches past it.
    """

    def __init__(self, record, frames: RecordFrames, history: dict):
        super().__init__(None, dict(history or {}), {})
        self.record = record
        self.frames = frames
        self.now = float("inf")
        self._state_t = np.asarray(record.state["t_recv"], np.float64)

    def _clip(self, t1: float) -> float:
        return min(float(t1), self.now)

    def states_between(self, t0: float, t1: float) -> StateWindow:
        self._declared("state", "states")
        if t1 < t0:
            raise ValueError(f"empty window: t1 {t1} < t0 {t0}")
        t1 = self._clip(t1)
        lo = int(np.searchsorted(self._state_t, t0, side="left"))
        hi = int(np.searchsorted(self._state_t, t1, side="right"))
        return _state_window(self.record.state[lo:max(lo, hi)])

    def _ring(self, cam: str):
        self._declared(cam, "frame_ids_between/frames_by_id")
        if cam not in self.frames.views:
            raise LookupError(f"no camera {cam!r} in this record; its views are "
                              f"{sorted(self.frames.views)}")
        return self.record.cameras[self.frames.views[cam].source]

    def frame_ids_between(self, cam: str, t0: float, t1: float) -> FrameTimes:
        src = self._ring(cam)
        if t1 < t0:
            raise ValueError(f"empty window: t1 {t1} < t0 {t0}")
        t1 = self._clip(t1)
        t = np.asarray(src.index["t_recv"], np.float64)
        keep = (t >= t0) & (t <= t1)
        return FrameTimes(np.asarray(src.index["frame_id"][keep], np.int64),
                          t[keep])

    def frames_by_id(self, cam: str, ids) -> list[Frame]:
        src = self._ring(cam)
        fid = np.asarray(src.index["frame_id"], np.int64)
        t = np.asarray(src.index["t_recv"], np.float64)
        out = []
        for want in np.asarray(ids, dtype=np.int64).reshape(-1):
            rows = np.flatnonzero(fid == want)
            if not len(rows) or t[rows[0]] > self.now:
                continue                    # never captured, or not yet
            out.append(self.frames.frame(cam, int(rows[0])))
        return out


__all__ = ["RecordFrames", "RecordReader", "observation"]
