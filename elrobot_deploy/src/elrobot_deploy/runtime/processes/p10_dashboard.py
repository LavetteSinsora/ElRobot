"""P10 — the operator's dashboard (this process; the server itself is
`dashboards/live/`). Described in
`docs/deploying_a_policy/running_and_live_dashboard.md`, "The live dashboard".

**Reads** the whole arena. **Writes** only operator requests: `control.request`,
`requested_config` and the prompt request — never anything in force.
Only P0 reads what it produces, and a run behaves identically whether or not it
exists — except that START arrives through it, which is why P0 keeps it
alive across every reset.

**Not merged into P9, deliberately.** Both processes scan the arena, and that is
the whole of what they have in common:

* P9 sweeps at 500 Hz because completeness is geometry rather than diligence
  — a slot lives seconds and it must miss nothing. P10 samples the newest
  value ten or twenty times a second and is lossy by design. Two loops, not one
  loop with a flag.
* P9's value is the record, and it writes `manifest.json` from its own `finally`.
  An HTTP server raises `BrokenPipeError` as a matter of routine — every closed
  tab is one. Putting a network-facing server in the process that owns the
  rollout record would let a closed browser tab cost the record.
* Their placement wants differ: P9 is pinned to an E core at normal priority,
  away from the camera cluster, and buffered so a disk stall drops records
  instead of blocking a read.
* P9 is optional (`--no-record`) and scoped to the rollout. P10 must be alive
  during WARMUP and READY, because that is exactly when the operator is looking
  at it to decide whether to press Start.

A second reader costs nothing: the arena's primitives were chosen so no reader
can delay a writer, and a seqlock reader contends with nobody.
"""

from __future__ import annotations

from elrobot_deploy.runtime import cpu_pinning as affinity
from elrobot_deploy.runtime.shared_memory import schema

#: Where the server listens. Loopback on purpose — the operator's browser
#: reaches it through the SSH port forward that is already open, which is also
#: what keeps it off the LAN paths that are unreliable from that Mac.
HOST = "127.0.0.1"
#: The port when neither the caller nor the machine file names one. The machine
#: file's `[dashboard] port` (`ELROBOT_DASHBOARD_PORT`) arrives as `hw.dashboard_port`.
DEFAULT_PORT = 8080

#: How often the process checks whether the session has ended. Nothing is
#: scheduled against it: the server does its own work on its own threads, and
#: this loop exists only so the process leaves when the session does.
_POLL_S = 0.05


def run(shm_name: str, cfg, hw, *, deploy_py=None, port: int | None = None,
        step: str | None = None, intervene: bool = False, ready=None) -> None:
    """P10's process entry point.

    MuJoCo is imported HERE, in the child, like every other heavy dependency in
    this system: the processes are spawned and never forked, and P0 must not
    pull a model compiler into its own address space to run a web server.

    The dashboard is *gated* like P7 and P8 — it sets `ready` once it is
    listening — because START arrives through it. A P0 that reached READY before
    the socket was up would be waiting for a press the operator has no way to
    make: the terminal offers no second path.
    """
    affinity.pin(hw, "p10_dashboard")
    import time

    from elrobot_deploy.runtime.shared_memory import Arena, build, open_all
    from elrobot_deploy.config_loading.derive import derive
    from elrobot_deploy.dashboards.live.scene import LiveScene
    from elrobot_deploy.dashboards.live.server import Live, serve

    arena = Arena.attach(shm_name, build(cfg, derive(cfg, hw)))
    server = None
    try:
        rings = open_all(arena, cfg)
        scene = LiveScene(cfg, hw, rings)
        # `step` is P0's (the resolved deploy file's `[checkpoint] step`), and
        # the port the machine file's: nothing here reads the environment.
        live = Live(cfg, hw, rings, scene, deploy_py=deploy_py, step=step,
                    intervene=intervene)
        if port is None:
            port = getattr(hw, "dashboard_port", None)
        server = serve(live, host=HOST,
                       port=int(DEFAULT_PORT if port is None else port))
        print(f"[P10] dashboard on http://{HOST}:{server.server_address[1]} — "
              f"from another machine: ssh -L {server.server_address[1]}:localhost:"
              f"{server.server_address[1]} <this box>", flush=True)
        if ready is not None:
            ready.set()

        control = rings["control"]
        while control.session < schema.STOPPED:
            time.sleep(_POLL_S)
    finally:
        if server is not None:
            # The streams block in their own loops; the flag is how they learn
            # to stop, and `shutdown` will not return until they have.
            server.stopping.set()
            server.shutdown()
            server.server_close()
        arena.close()


__all__ = ["run", "HOST", "DEFAULT_PORT"]
