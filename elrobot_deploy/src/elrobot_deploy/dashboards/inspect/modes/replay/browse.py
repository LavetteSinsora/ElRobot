"""The replay panel's run picker: server-side directory browsing.

Checkpoints (for a policy that has them) live wherever their owner keeps them, so the dashboard does not
guess a layout. It browses the filesystem of the box it runs on, starting from
places that are likely to be right:

* the recorded checkpoint's run directory (its parent) — "another step of the
  same run" is the commonest question;
* the machine file's `[inspect] runs` and `--runs` directories;
* the directories picked recently (remembered in `Options.state_file`);
* a path typed or pasted into the field.

A directory is a RUN when it has numeric step subdirectories — the one layout
elrobot_deploy knows (`policy_contract/checkpoints.py::numeric_steps`: numeric names) — and its
steps are listed. A
policy with its own layout says so with `resolve_checkpoint(run, step)`, which
the picker learns by asking the class in a subprocess (`replay-policy
--describe-entry`); it then also accepts a free-text step and hands `--run` /
`--step` to the replay, which lets the class resolve them. A pasted step
directory is taken as-is (`--checkpoint`).

Read-only: this lists directories and never opens a file in them.

Used by the replay mode's routes (`modes/replay/__init__.py`). For a user:
`docs/deploying_a_policy/rollout_inspection.md`, "policy checkpoint (`replay`)".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: How many subdirectories one listing returns. A directory of thousands of
#: step folders is listed by its steps, not by its children.
MAX_DIRS = 400
#: How many recently used directories are remembered.
RECENT = 12


def listing(path: str | Path) -> dict:
    """One directory: its subdirectories and, if it is a run, its steps."""
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        raise ValueError(f"{path!r}: give an absolute path")
    p = p.resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not a directory")
    dirs = []
    try:
        children = sorted(os.scandir(p), key=lambda e: e.name)
    except OSError as exc:
        raise FileNotFoundError(f"{p}: {exc}") from exc
    steps = []
    files = []
    for e in children:
        if e.name.startswith("."):
            continue
        try:
            if not e.is_dir():
                if e.name.endswith(".toml"):
                    files.append({"name": e.name, "path": os.path.join(p, e.name)})
                continue
        except OSError:
            continue
        if e.name.isdigit():
            steps.append({"step": int(e.name), "path": os.path.join(p, e.name)})
            continue
        if len(dirs) < MAX_DIRS:
            dirs.append({"name": e.name, "path": os.path.join(p, e.name)})
    steps.sort(key=lambda s: s["step"])
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs, "steps": steps, "files": files,
            "truncated": len(dirs) >= MAX_DIRS}


def deploy_of(path: str | Path) -> dict:
    """What a deploy file names, for the replay form: the policy entry (made
    absolute against the file), the interpreter it runs under ("" = the
    dashboard's), and the checkpoint directory it points at ("" when it names
    none, or one this function cannot place: an alias only the class resolves).
    Reads the TOML only; nothing is imported."""
    from elrobot_deploy.config_loading.deploy_file import policy_interpreter
    from elrobot_deploy.config_loading.resolved import tomllib
    from elrobot_deploy.policy_contract import loading

    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        raise ValueError(f"{path!r}: give an absolute path")
    if not p.is_file():
        raise FileNotFoundError(f"{p} is not a file")
    try:
        tree = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{p}: {exc}") from exc
    pol = tree.get("policy") or {}
    if not pol.get("entry"):
        raise ValueError(f"{p} names no [policy] entry")
    base = p.resolve().parent
    ck = tree.get("checkpoint") or {}
    checkpoint = ""
    if ck.get("path"):
        checkpoint = str((base / Path(str(ck["path"])).expanduser()).resolve())
    elif ck.get("run"):
        run = (base / Path(str(ck["run"])).expanduser()).resolve()
        step = str(ck.get("step", "latest"))
        if step == "latest" and run.is_dir():
            nums = sorted(int(d.name) for d in run.iterdir()
                          if d.is_dir() and d.name.isdigit())
            step = str(nums[-1]) if nums else ""
        if step and (run / step).is_dir():
            checkpoint = str(run / step)
    return {"path": str(p.resolve()),
            "entry": loading.normalize(str(pol["entry"]), base_dir=base),
            "python": policy_interpreter(pol["python"], base) if pol.get("python") else "",
            "checkpoint": checkpoint}


#: How long a native dialog may stay open before the request gives up.
NATIVE_TIMEOUT_S = 900.0


def native_pick(what: str, start: str = "") -> dict:
    """Open the desktop's own file dialog ON THIS BOX and return the choice.

    The dashboard is normally viewed on the machine it runs on, where the
    desktop's dialog is what a user expects; a web page cannot get a real path
    out of the browser's own file dialog, so the server opens one instead.
    `what` is "deploy" (a file, the list filtered to `*.toml`) or "ckpt" (a
    folder); `start` is the path in the field, where the dialog opens.

    The dialog is `native_dialog.py` run under the system Python (GTK), else
    `zenity`, else `kdialog`. Returns `{"path": ...}`, `{"cancelled": True}`,
    or `{"unavailable": why}` when there is no display or no dialog program —
    the page then falls back to its own folder list (`listing`), which is what
    a viewer coming in over an SSH tunnel gets.
    """
    import shutil
    import subprocess

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return {"unavailable": "no display on the box the dashboard runs on"}
    folder = what != "deploy"
    title = "Pick a checkpoint folder" if folder else "Pick a deploy.toml"
    start = str(Path(str(start or Path.home())).expanduser())
    here = Path(start)
    while not here.is_dir() and here != here.parent:
        here = here.parent

    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=NATIVE_TIMEOUT_S)
        except (subprocess.TimeoutExpired, OSError):
            return None

    candidates = []
    system_python = shutil.which("python3", path="/usr/bin:/bin")
    if system_python:
        script = Path(__file__).with_name("native_dialog.py")
        candidates.append(([system_python, str(script), "deploy" if not folder else "ckpt",
                            start], 2))
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", f"--title={title}", f"--filename={here}/"]
        cmd += ["--directory"] if folder else ["--file-filter=deploy files (*.toml) | *.toml"]
        candidates.append((cmd, None))
    if shutil.which("kdialog"):
        cmd = ["kdialog", "--title", title]
        cmd += ["--getexistingdirectory", str(here)] if folder \
            else ["--getopenfilename", str(here), "*.toml"]
        candidates.append((cmd, None))

    why = "no file dialog program on this box (GTK for python3, zenity or kdialog)"
    for cmd, unavailable_code in candidates:
        proc = run(cmd)
        if proc is None:
            continue
        lines = proc.stdout.strip().splitlines()
        if proc.returncode == 0 and lines:
            return {"path": lines[-1]}
        if proc.returncode == 1:
            return {"cancelled": True}
        # This program could not show a dialog: try the next one.
        why = (proc.stderr.strip().splitlines() or ["the dialog did not open"])[-1]
    return {"unavailable": why}


def recent(state_file: Path) -> list[str]:
    try:
        data = json.loads(Path(state_file).read_text())
    except (OSError, ValueError):
        return []
    return [str(d) for d in data.get("recent_runs", []) if isinstance(d, str)]


def remember(state_file: Path, path: str) -> None:
    """Put `path` first in the recently used list. Best effort: a box whose
    home is read-only simply has no memory."""
    rows = [path] + [d for d in recent(state_file) if d != path]
    try:
        state_file = Path(state_file)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_name(state_file.name + ".tmp")
        tmp.write_text(json.dumps({"recent_runs": rows[:RECENT]}, indent=1))
        os.replace(tmp, state_file)
    except OSError:
        pass


def starts(options, checkpoint: str) -> list[dict]:
    """The picker's starting points, most likely first, each existing."""
    out: list[dict] = []

    def add(path, why):
        if not path:
            return
        p = Path(str(path)).expanduser()
        if p.is_dir() and not any(o["path"] == str(p) for o in out):
            out.append({"path": str(p), "why": why})

    if checkpoint:
        c = Path(checkpoint)
        add(c.parent, "the recorded checkpoint's run")
        add(c.parent.parent, "above it")
    for r in getattr(options, "runs", None) or ():
        add(r, "[inspect] runs")
    for r in recent(getattr(options, "state_file", Path("/nonexistent"))):
        add(r, "used recently")
    add(Path.home(), "home")
    return out


__all__ = ["MAX_DIRS", "deploy_of", "listing", "native_pick", "recent", "remember",
           "starts"]
