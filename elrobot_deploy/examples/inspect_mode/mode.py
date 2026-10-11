"""An inspection mode of your own: inference latency per replan, on the strip.

    g1d_deploy inspect --mode examples/inspect_mode/mode.py:LatencyMode

A mode is a class (`docs/deploying_a_policy/rollout_inspection.md`, "Writing a
mode"): it registers routes on the server and ships a `mode_<name>.js` that
draws. This one reads the recording's own chunks — each carries the six
instants of its replan (`docs/infra_logic/arena_layout_and_reads.md`, "The
records, field by field") — and plots, per replan:

* trigger -> chunk written: what the replan period has to fit (the number
  async's START check is about);
* trigger -> anchor: how long the anchor waited for a fresher frame (negative:
  it took a frame that already existed).

Nothing here needs a GPU or a subprocess; a mode that does must run it as one
(`dashboards/web/jobs.py`'s RUNNER), never inside the server.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from g1d_deploy.dashboards.inspect.modes import describe


def latency_strip(loaded) -> dict:
    """`{t, total_ms, anchor_ms}`, one entry per replan; `t` in seconds from
    the display window's start (`timeline.t0`), the main timeline's axis."""
    c = loaded.rollout.chunks
    c = c[c["chunk_id"] > 0]
    t = np.asarray(c["t_trigger"], np.float64) - float(loaded.timeline.t0)
    total = (c["t_chunk_written"] - c["t_trigger"]) * 1e3
    anchor = (c["t_anchor"] - c["t_trigger"]) * 1e3
    return {"t": np.round(t, 4).tolist(),
            "total_ms": np.round(total, 2).tolist(),
            "anchor_ms": np.round(anchor, 2).tolist()}


class LatencyMode:
    name = "latency"                          # ?mode=latency, and mode_latency.js
    title = "inference latency"
    blurb = "Per replan: trigger to chunk written, and the anchor's wait"
    static_dir = Path(__file__).parent / "static"

    def __init__(self, options=None):
        self.options = options
        self._state = None

    def mount(self, app, state) -> None:
        self._state = state

        @app.get("/api/latency/strip")
        def strip(req):
            cur = state.current
            if cur is None:
                req.error(RuntimeError("nothing loaded"), 409)
                return
            req.json(latency_strip(cur))

    def describe(self) -> dict:
        return describe(self)
