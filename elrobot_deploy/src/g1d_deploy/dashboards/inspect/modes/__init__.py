"""The rollout dashboard's modes: what the third section asks of a rollout.
For the user, and how to write one:
`docs/deploying_a_policy/rollout_inspection.md`, "Modes" and "Writing a mode";
a worked plug-in is `examples/inspect_mode/`.

A `Mode` is one comparison a loaded rollout can be put through. Two are on by
default, and a third ships switched off:

    ik          which IK solver would have reached this rollout's poses — the
                infra's own solvers (`g1d_deploy.runtime.ik`), benched against
                the record
    replay      what a policy (this checkpoint, another step, another run,
                other sample steps) would have commanded on the rollout's own
                observations — re-run offline, drawn as ghosts
    executor    (off by default) how another 500 Hz executor drove the
                joints — replay records of this rollout, as traces; the
                producer is plugged in. Mounted by `--mode executor`, or by
                naming a producer (`--executor-producer`)

**Extra modes plug in by module path**, `"pkg.module:Class"`, from the command
line (`--mode`, repeatable) or the machine file (`[inspect] modes = [...]`).
A mode is a class with:

    name: str               unique; the dropdown's value and the URL's ?mode=
    title: str              the dropdown's label
    blurb: str              its tooltip
    static_dir: Path | None optional: served under /static/, searched before
                            the built-in files, where `mode_<name>.js` lives
    __init__(self[, options])   `options` is `dashboards/inspect/options.py::Options`
    mount(self, app, state)     register routes (`app.get` / `app.post`),
                                subscribe to `state.on_load(fn(loaded, dir))`
    describe(self) -> dict      `modes.describe(self)` is enough

and a `mode_<name>.js` exporting `activate(ctx)` / `deactivate()` (see
`static/app.js`'s `hooks`). The server imports no producer: what a mode
*reads* is a contract on disk, and what *writes* it is a subprocess
(`dashboards/web/jobs.py`), because producers take the GPU or pin cores.

`mount` builds every requested mode; one that fails to import is reported and
left out, so a broken plug-in never takes the viewer down.
"""

from __future__ import annotations

import inspect
import traceback
from pathlib import Path
from typing import Protocol

#: The built-in modes, in the order the dropdown offers them.
BUILTIN = {
    "ik": "g1d_deploy.dashboards.inspect.modes.ik:IkMode",
    "replay": "g1d_deploy.dashboards.inspect.modes.replay:ReplayMode",
}
#: Shipped but off unless asked for by name (`--mode executor`).
OPTIONAL = {
    "executor": "g1d_deploy.dashboards.inspect.modes.executor:ExecutorMode",
}


class Mode(Protocol):
    name: str
    title: str
    blurb: str

    def mount(self, app, state) -> None:
        """Register routes; subscribe to `state.on_load`."""

    def describe(self) -> dict:
        ...


def describe(mode: Mode) -> dict:
    return {"name": mode.name, "title": mode.title, "blurb": mode.blurb,
            "script": f"/static/mode_{mode.name}.js"}


def load_class(spec: str):
    """`"pkg.module:Class"` or `"path/file.py:Class"` (relative to the working
    directory) -> the class. Raises `ValueError` on a bad spelling."""
    from g1d_deploy.policy_contract.loading import load_attr
    mod, sep, attr = str(spec).rpartition(":")
    if not sep or not mod or not attr.isidentifier():
        raise ValueError(f"mode {spec!r} must be \"pkg.module:Class\" or "
                         f"\"path/file.py:Class\"")
    return load_attr(spec)


def build(spec: str, options=None):
    """Instantiate a mode class, handing it `options` if it takes an argument."""
    cls = load_class(spec)
    try:
        takes = len(inspect.signature(cls).parameters) >= 1
    except (TypeError, ValueError):
        takes = False
    return cls(options) if takes else cls()


def mount(app, state, specs: list[str] | None = None, options=None) -> list[Mode]:
    """Build and mount `specs` (default: the built-ins, then `options.modes`)."""
    if specs is None:
        specs = list(BUILTIN.values())
        if getattr(options, "executor_producer", ""):
            specs.append(OPTIONAL["executor"])
        specs += list(getattr(options, "modes", None) or ())
    out: list[Mode] = []
    seen: set[str] = set()
    for spec in specs:
        spec = BUILTIN.get(spec) or OPTIONAL.get(spec) or spec
        try:
            mode = build(spec, options)
            if mode.name in seen:
                raise ValueError(f"a mode named {mode.name!r} is already mounted")
            static = getattr(mode, "static_dir", None)
            if static:
                # Before the built-in directories, so a plug-in's own files win.
                app.static_dirs.insert(0, Path(static).resolve())
            mode.mount(app, state)
        except Exception:                   # noqa: BLE001 — reported, not fatal
            print(f"[inspect] mode {spec!r} unavailable:")
            traceback.print_exc()
            continue
        seen.add(mode.name)
        out.append(mode)
    return out


__all__ = ["BUILTIN", "OPTIONAL", "Mode", "build", "describe", "load_class", "mount"]
