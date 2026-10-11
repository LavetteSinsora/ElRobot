"""The machine file: this deploy box's overrides of the defaults. OPTIONAL.

Read at config resolution by every command. Design: `docs/setup/machine_toml.md`.

    <project root>/machine.toml       (template: machine.example.toml, beside it)

Everything in it has a default or is detected: the NIC DDS binds to (found on
the robot's subnet), the CPU pin table (derived from the topology,
`runtime/cpu_pinning.py::derive`), the dashboard port, environment the policy
process should start with, and where this box keeps its recordings
(`[paths] records`). With no file, every value is its default. None of it is
about the robot (its address and login are the robot file's `[board]`) or the
task. It lives in the checkout and is never committed (`.gitignore`).

Resolution, first found wins: `--machine PATH`, then `$ELROBOT_MACHINE`, then
`<project root>/machine.toml`, then no file at all. A NAMED file that does not
exist is an error; nothing is ever created for you.

Each scalar can be overridden for one shell with a `ELROBOT_*` variable (named in
machine.example.toml).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from elrobot_deploy.config_loading.resolved import tomllib
from elrobot_deploy.config_loading.errors import ConfigError

MACHINE_ENV = "ELROBOT_MACHINE"
#: The checkout: the directory holding pyproject.toml, two above this package
#: (`src/elrobot_deploy/`). The project is installed editable (`uv sync`), so this
#: is the clone the user is working in.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = PROJECT_ROOT / "machine.toml"
TEMPLATE = PROJECT_ROOT / "machine.example.toml"

#: The subnet the robot's DDS lives on — the G1-D's factory network.
DEFAULT_SUBNET = "192.168.123."


@dataclass(frozen=True)
class Machine:
    """A machine file, resolved. `sources` says where each value came from —
    `file`, `env`, `auto` or `default` — for `elrobot_deploy check` and the manifest."""

    #: None when there is no machine file (every value is a default), and for
    #: a Machine built in memory (offline inspection).
    path: str | None
    text: str | None
    iface: str
    subnet: str
    #: `[dashboard] port`: where P10 listens, on loopback.
    dashboard_port: int = 8080
    #: `[cpu]` as written (`reserved`, `rt_priority`, optional `pin`), handed
    #: to `resolved._cpu`, which derives the table when `pin` is absent.
    cpu: dict = field(default_factory=dict)
    #: `[env]`: exported into P5 before it imports anything.
    env: dict[str, str] = field(default_factory=dict)
    #: `[paths] records`: this box's record root, absolute; "" = unset. A run
    #: whose deploy file names no `[record] out` records to
    #: `<records>/<task>/`, and `elrobot_deploy inspect` with no root opens it.
    records: str = ""
    sources: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        return (f"{self.path or '(no machine file: defaults)'}  "
                f"iface={self.iface or '<none>'} ({self.sources.get('iface', '?')})")


#: Keys that were the machine file's and are the robot file's now: refused by
#: name, so a value written here is never silently ignored.
_MOVED = ("robot_ip", "board")


def machine_path(explicit: str | Path | None = None) -> Path | None:
    """The machine file to read: `explicit`, then `$ELROBOT_MACHINE`, then
    `<project root>/machine.toml`; None when there is none (defaults apply).

    A NAMED path (flag or variable) that does not exist is an error.
    """
    for candidate, how in ((explicit, "--machine"),
                           (os.environ.get(MACHINE_ENV), MACHINE_ENV)):
        if not candidate:
            continue
        p = Path(candidate).expanduser()
        if p.is_file():
            return p.resolve()
        raise ConfigError(f"machine file {p} does not exist ({how})")
    return DEFAULT_PATH if DEFAULT_PATH.is_file() else None


def detect_iface(subnet: str = DEFAULT_SUBNET) -> str | None:
    """The first interface holding an IPv4 address on `subnet`, or None.

    `ip -4 -o addr`, first match on `subnet`.
    """
    try:
        out = subprocess.run(("ip", "-4", "-o", "addr"), capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pattern = re.compile(r"^\d+:\s+(\S+)\s+inet\s+" + re.escape(subnet))
    for line in out.splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1).split("@")[0]
    return None


def _port(value) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"dashboard port {value!r} is not a number") from None
    if not 0 <= port <= 65535:
        raise ConfigError(f"dashboard port {port} is outside 0..65535")
    return port


def records_root(raw: dict, path: Path | None) -> str:
    """`[paths] records` (or `$ELROBOT_RECORDS`) as an absolute path, "" if unset.
    Relative to the machine file's directory (the project root without one);
    `~` expanded."""
    path = path or DEFAULT_PATH
    value = os.environ.get("ELROBOT_RECORDS", "").strip() or \
        str((raw.get("paths") or {}).get("records") or "").strip()
    if not value:
        return ""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (path.parent / p).resolve())


def load_machine(path: str | Path | None = None, *, resolve_iface: bool = True) -> Machine:
    """Read the machine file if there is one (see `machine_path`), apply
    `ELROBOT_*` overrides, and resolve `iface = "auto"`.

    `resolve_iface=False` leaves an unresolvable NIC as "" instead of raising —
    for readers (`elrobot_deploy check`) that report the problem rather than refuse.
    """
    p = machine_path(path)
    text = p.read_text() if p is not None else None
    try:
        raw = tomllib.loads(text) if text is not None else {}
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"machine file {p}: {exc}") from exc
    moved = [k for k in _MOVED if k in raw]
    if moved:
        raise ConfigError(
            f"machine file {p}: {moved} belong to the robot file now, not the "
            f"machine file (the address and login are facts about one robot). "
            f"Delete them here; the factory values are the default, and another "
            f"robot's go in its robot file's `[board]` (ip, ssh_user, "
            f"ssh_password) — docs/setup/robot_toml.md.")
    net = raw.get("network", {})
    sources: dict[str, str] = {}

    def pick(key: str, env: str, value, default):
        if os.environ.get(env, "").strip():
            sources[key] = "env"
            return os.environ[env].strip()
        if value is not None:
            sources[key] = "file"
            return value
        sources[key] = "default"
        return default

    subnet = str(net.get("subnet", DEFAULT_SUBNET))
    iface = pick("iface", "ELROBOT_IFACE", net.get("iface"), "auto")
    if iface == "auto":
        found = detect_iface(subnet)
        sources["iface"] = "auto"
        if found is None:
            if resolve_iface:
                raise ConfigError(
                    f"no network interface holds a {subnet}x address, so there is "
                    f"no NIC to bind DDS to. Plug the robot cable in and give this "
                    f"box its static address on that subnet, or name the NIC in "
                    f"the machine file ([network] iface) or with ELROBOT_IFACE.")
            found = ""
        iface = found

    return Machine(
        path=str(p) if p is not None else None, text=text,
        iface=str(iface), subnet=subnet,
        dashboard_port=_port(pick("dashboard_port", "ELROBOT_DASHBOARD_PORT",
                                  raw.get("dashboard", {}).get("port"), 8080)),
        cpu=dict(raw.get("cpu", {})),
        env={str(k): str(v) for k, v in raw.get("env", {}).items()},
        records=records_root(raw, p),
        sources=sources)


__all__ = [
    "DEFAULT_PATH",
    "MACHINE_ENV",
    "PROJECT_ROOT",
    "TEMPLATE",
    "Machine",
    "detect_iface",
    "load_machine",
    "machine_path",
    "records_root",
]
