"""Starting and stopping P0's children, each process by module path, spawned (never forked)."""

from __future__ import annotations

import multiprocessing as mp
import signal

from elrobot_deploy.runtime import timing as _clock


def _child(dotted: str, args: tuple, kwargs: dict) -> None:
    """The spawn entry point for every child process.

    The process module is imported *here*, in the child, so P0 never pulls mujoco or the
    serial port into its own address space. Spawned, not forked: a forked process inherits
    whatever the parent had open.

    SIGINT is ignored: Ctrl-C reaches the whole process group, and a bus process that died on
    the first one could not execute the exit ramp the first one is supposed to start. The
    operator's interrupt is P0's alone; a child ends when P0 stops it.
    """
    import importlib

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    module, _, func = dotted.rpartition(".")
    getattr(importlib.import_module(module), func)(*args, **kwargs)


def _spawn_process(name: str, dotted: str, args: tuple, kwargs: dict):
    """Spawn a child running `dotted(*args, **kwargs)`. (g1d also ran P5 under a policy's own
    interpreter here; that path is in `legacy_g1d/`.)"""
    proc = mp.get_context("spawn").Process(
        target=_child, args=(dotted, args, kwargs), name=name, daemon=False)
    proc.start()
    return proc


def _spawn_event():
    """The `ready` latch a gated child sets once it is on the wire.

    NOT in the arena, deliberately: this is a one-shot lifecycle fact read exactly once, by P0,
    before the robot has moved, and an `Event` already is one.

    An `Event` rather than a pipe or a queue because the fact is a LATCH, not a
    stream: `set()` is idempotent, `wait()` is re-entrant, and no reader can
    consume it out from under another. It must come from the spawn context and
    be passed at `Process` construction — the only way a synchronisation
    primitive survives `spawn` — which is what `_Children.start` does with it.
    """
    return mp.get_context("spawn").Event()


class _Children:
    """The spawned processes, by name. No supervision, by decision.

    Nothing here notices that a child has died: a dead process is invisible,
    and the alternative is a restart policy the system does not
    have.

    P0 asks `alive()` where a wait would otherwise never end (a bus process that died before
    it was ready, a P6 that died mid-rollout). That is a fact, not supervision: nothing is
    restarted.
    """

    def __init__(self, spawn, event=None):
        self._spawn = spawn
        self._event = event or _spawn_event
        self.handles: dict[str, object] = {}
        #: The `ready` latch of every child started with `gated=True`, by name.
        self.ready: dict[str, object] = {}

    def start(self, name: str, dotted: str, args: tuple, *,
              gated: bool = False, **kwargs) -> None:
        """Spawn a child. `gated` also hands it a `ready` latch to set.

        The latch rides in `kwargs`, which `_child` already splats into the
        child's `run` — so a gated process needs no plumbing beyond a `ready`
        keyword and one `set()`.
        """
        if gated:
            self.ready[name] = kwargs["ready"] = self._event()
        self.handles[name] = self._spawn(name, dotted, args, kwargs)

    def alive(self, name: str) -> bool:
        """Is this child still running?

        The one exception to "nothing here notices that a child has died", and
        it is not supervision: nothing is restarted and no policy is applied.
        P0's waits have no timeout by decision, so the only thing that can distinguish "still
        starting" from "died" is whether the process is there.
        """
        handle = self.handles.get(name)
        if handle is None:
            return False
        try:
            return bool(handle.is_alive())
        except Exception:
            # A spawn backend that cannot answer is treated as alive: the wait
            # then simply runs to its own timeout.
            return True

    def stop(self, *names: str) -> None:
        for name in names:
            handle = self.handles.pop(name, None)
            if handle is None:
                continue
            try:
                handle.terminate()
                handle.join()
            except Exception:
                pass

    def wait(self, *names: str, timeout: float) -> list[str]:
        """Wait for children to exit **on their own**. Returns those that did not.

        Not `stop`: `terminate` is SIGTERM, whose default disposition kills the
        interpreter outright, so no `finally` in the child runs. A process that
        has work to finish (the bus process turns torque off and writes its log from its own
        `finally`) must be given the chance to notice the session word and close itself.

        A child that left cleanly is dropped from `handles`, so the `stop_all`
        that follows signals only the ones that did not. Sending SIGTERM to a
        process that has already exited is harmless, but "was this child killed
        or did it finish?" should be answerable from what P0 did, not from
        whether the signal happened to matter.
        """
        # One budget for the whole shutdown, not one per child: `timeout` is how
        # long the operator waits, and four children each granted the full
        # window would quadruple that on the one path where something is
        # already wrong.
        deadline = _clock.now() + timeout
        stragglers = []
        for name in names:
            handle = self.handles.get(name)
            if handle is None:
                continue
            try:
                handle.join(max(0.0, deadline - _clock.now()))
                left = not getattr(handle, "is_alive", lambda: False)()
            except Exception:
                left = False
            if left:
                self.handles.pop(name, None)
            else:
                stragglers.append(name)
        return stragglers

    def stop_all(self) -> None:
        self.stop(*list(self.handles))
