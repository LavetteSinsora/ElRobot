"""What `elrobot_deploy inspect` was told, from the flags and the machine file.

    elrobot_deploy inspect <root-or-record> [--port N] [--mode pkg.mod:Mode]
                 [--executor-producer pkg.mod] [--runs DIR ...]

Every value can also live in the machine file's `[inspect]` table, so a box
that always inspects the same way says it once:

    [inspect]
    modes = ["my_pkg.my_mode:MyMode"]        # extra modes, beside the built-ins
    executor_producer = "my_pkg.simrig"     # what the executor mode spawns
    runs = ["/data/runs", "~/checkpoints"]  # where the replay picker starts
    port = 8770

Flags win over the file; list-valued flags ADD to the file's list rather than
replacing it, so a one-off `--mode` does not drop the box's own.

Nothing here reads an environment variable of its own. The machine file is
found the way every elrobot_deploy command finds it (`machine_file.machine_path`:
`--machine`, `$ELROBOT_MACHINE`, `<project root>/machine.toml`). Inspection is the
one command that does not need one: without it there is simply no `[inspect]`
table.

Built by `cli/inspect.py`, read by the server and every mode. For a user:
`docs/deploying_a_policy/rollout_inspection.md` (the flags), `docs/setup/machine_toml.md`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

#: The port rollout inspection has always used.
DEFAULT_PORT = 8770

#: The file the replay picker remembers directories in. Per user, outside any
#: checkout: it is a convenience, and nothing reads it but the picker.
DEFAULT_STATE = Path("~/.local/state/elrobot_deploy/inspect.json")


@dataclass
class Options:
    """Everything the server and its modes are configured with."""

    #: A directory searched for records, or one record directory.
    root: Path
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    #: Extra modes, `"pkg.module:Class"`, mounted after the built-ins.
    modes: list[str] = field(default_factory=list)
    #: The executor mode's producer, a module run as `python -m <it>`; "" = none.
    executor_producer: str = ""
    #: Where the replay picker's directory browser starts, beside the recorded
    #: checkpoint's parent and the recently used directories.
    runs: list[Path] = field(default_factory=list)
    #: The interpreter producers are spawned with: this one, so they run in the
    #: same venv as the server.
    python: str = field(default_factory=lambda: sys.executable)
    #: Where recently used run directories are remembered.
    state_file: Path = field(default_factory=lambda: DEFAULT_STATE.expanduser())
    #: Where these came from, for the startup banner.
    machine_file: str | None = None


def _machine_table(machine: str | None) -> tuple[dict, str | None, str]:
    """The machine file's `[inspect]` table and its `[paths] records`
    ({} and "" when there is no file)."""
    from elrobot_deploy.config_loading.resolved import tomllib
    from elrobot_deploy.config_loading.errors import ConfigError
    from elrobot_deploy.config_loading.machine_file import machine_path

    path = machine_path(machine)
    if path is None:
        return {}, None, ""
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"machine file {path}: {exc}") from exc
    table = raw.get("inspect") or {}
    if not isinstance(table, dict):
        raise ConfigError(f"machine file {path}: [inspect] must be a table")
    from elrobot_deploy.config_loading.machine_file import records_root
    return table, str(path), records_root(raw, path)


def resolve(root, *, machine: str | None = None, host: str | None = None,
            port: int | None = None, modes=(), executor_producer: str | None = None,
            runs=()) -> Options:
    """Flags over the machine file's `[inspect]` over the defaults. With no
    `root`, the machine file's `[paths] records`."""
    table, path, records = _machine_table(machine)
    if root is None:
        if not records:
            from elrobot_deploy.config_loading.errors import ConfigError
            raise ConfigError(
                "no records to open: name a directory (elrobot_deploy inspect "
                "DIR), or set [paths] records in the machine file")
        root = records
    file_modes = [str(m) for m in table.get("modes") or ()]
    file_runs = [str(r) for r in table.get("runs") or ()]
    return Options(
        root=Path(root).expanduser().resolve(),
        host=host or str(table.get("host") or "127.0.0.1"),
        port=int(port if port is not None else table.get("port") or DEFAULT_PORT),
        modes=file_modes + [m for m in modes if m not in file_modes],
        executor_producer=str(executor_producer
                              or table.get("executor_producer") or ""),
        runs=[Path(r).expanduser() for r in file_runs + list(runs)],
        machine_file=path)


__all__ = ["DEFAULT_PORT", "DEFAULT_STATE", "Options", "resolve"]
