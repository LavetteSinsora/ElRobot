"""Is a live deployment holding the GPU right now?

A policy replay loads a policy, and the deploy box's policy process (P5)
may be holding most of the card (e.g. a framework that preallocates).
Starting a replay beside a live rollout would at best OOM the replay and at
worst the robot's own inference, mid-rollout. So a replay is refused while a
deployment runs, and the refusal says which one.

**The arena is the tell.** P0 creates the shared-memory arena as
`/dev/shm/g1d-<its pid>` before anything else starts and unlinks it when the
session ends (`runtime/launcher/state_machine.py`). A segment whose pid is alive is a
running deployment; one whose pid is gone is a leftover of a crash and holds
nothing. Off Linux there is no `/dev/shm` to look in, and nothing is detected.

Used by the inspection replay (`dashboards/inspect/modes/replay/`) and by
`elrobot_deploy check-policy` level 2.
"""

from __future__ import annotations

import os
from pathlib import Path

SHM = Path("/dev/shm")
#: `runtime.launcher` `Launcher`'s arena name prefix.
PREFIX = "elr-"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                 # exists, owned by someone else
    return True


def live_deployments(shm: Path | None = None) -> list[dict]:
    """`[{"arena": path, "pid": int}]` for every running deployment."""
    shm = SHM if shm is None else Path(shm)
    out = []
    try:
        entries = sorted(shm.glob(PREFIX + "*"))
    except OSError:
        return out
    for p in entries:
        tail = p.name[len(PREFIX):]
        if tail.isdigit() and _alive(int(tail)):
            out.append({"arena": str(p), "pid": int(tail)})
    return out


def refusal(shm: Path | None = None) -> str:
    """Why a GPU job must not start now, or "" when it may."""
    live = live_deployments(shm)
    if not live:
        return ""
    who = ", ".join(f"{d['arena']} (P0 pid {d['pid']})" for d in live)
    return (f"a live deployment is running ({who}) and its policy holds the "
            f"GPU; a replay would compete with the robot's own inference. "
            f"Exit the deployment first, or replay on another box.")


__all__ = ["PREFIX", "live_deployments", "refusal"]
