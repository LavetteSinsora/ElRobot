"""The live dashboard P10 serves
(`docs/deploying_a_policy/running_and_live_dashboard.md`, "The live dashboard").

`dashboards/inspect/` looks at a rollout that has finished; this looks at the
one happening now, and can drive it. The split of responsibility is the same in
both: Python poses the model and packs the numbers, the browser draws them, and
the renderer (`dashboards/web/static/gl.js`) is literally the same file so
the two views cannot disagree about what the robot looks like.

What is particular here is the write path. This is the only part of the system
outside P0 that can change what the session does, and it does it through
operator requests — `control.request` and the requested settings and prompt —
which P0 consumes. See `shared_memory/rings.py::Control` for why `request` is
the one place two writers are tolerated, and `runtime/launcher/state_machine.py`
for what happens to each request.
"""
