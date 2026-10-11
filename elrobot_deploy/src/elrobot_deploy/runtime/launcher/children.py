"""Starting and stopping P0's children: each process by module path, under
this interpreter or (P5) the policy's own (`[policy] python`, through
`runtime/processes/policy_host.py` and `config_loading/json_transport.py`)."""

from __future__ import annotations

import multiprocessing as mp
import signal

from elrobot_deploy.runtime import timing as _clock


def _child(dotted: str, args: tuple, kwargs: dict) -> None:
    """The spawn entry point for every child process.

    The process module is imported *here*, in the child, so P0 never pulls DDS,
    CUDA or mujoco into its own address space. Processes are spawned and never
    forked, because a forked CUDA or DDS context is unusable.

    SIGINT is ignored: Ctrl-C reaches the whole process group, and a P7 that
    died on the first one could not execute the exit ramp that the first one is
    supposed to start. The operator's interrupt is P0's alone; a child ends when
    P0 stops it.
    """
    import importlib

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    module, _, func = dotted.rpartition(".")
    getattr(importlib.import_module(module), func)(*args, **kwargs)


class _InterpreterChild:
    """A child running under another interpreter (`processes.interp`), with
    the three methods `_Children` asks of a `multiprocessing.Process`."""

    def __init__(self, proc):
        self._proc = proc

    @property
    def pid(self) -> int:
        return self._proc.pid

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()

    def join(self, timeout: float | None = None) -> None:
        import subprocess
        try:
            self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            pass


def _spawn_interpreter(python: str, name: str, dotted: str, args: tuple,
                       kwargs: dict) -> _InterpreterChild:
    """`dotted(*args, **kwargs)` in a fresh `python`, the call as `codec` JSON
    on its stdin — never a pickle, because that interpreter's numpy may be
    another major version (`processes.interp`). Every argument must be
    `codec`-encodable: the arena name, `DeployConfig`, `Hardware`."""
    import subprocess

    from elrobot_deploy.config_loading import json_transport as codec
    payload = codec.dumps({"dotted": dotted, "args": tuple(args),
                           "kwargs": dict(kwargs)})
    proc = subprocess.Popen([python, "-m", "elrobot_deploy.runtime.processes.policy_host"],
                            stdin=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(payload)
    finally:
        proc.stdin.close()
    return _InterpreterChild(proc)


def _spawn_process(name: str, dotted: str, args: tuple, kwargs: dict):
    """Spawn a child. `interpreter` in `kwargs` (popped) runs it under THAT
    Python instead (`_spawn_interpreter`) — P5 under `[policy] python`."""
    kwargs = dict(kwargs)
    python = kwargs.pop("interpreter", None)
    if python:
        return _spawn_interpreter(python, name, dotted, args, kwargs)
    proc = mp.get_context("spawn").Process(
        target=_child, args=(dotted, args, kwargs), name=name, daemon=False)
    proc.start()
    return proc


def _spawn_event():
    """The `ready` latch a gated child sets once it is on the wire.

    NOT in the arena, deliberately. The arena describes the robot — `session`,
    `k_exec`, `warm` — and is read at 500 Hz by processes that must never branch
    on a startup detail; this is a one-shot lifecycle fact read exactly once, by
    P0, before the robot has moved. Giving it a slot would cost a `schema.py`
    entry and a layout-hash bump to express something an `Event` already is.

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

    `reset` stops P6, P7 and P8 and starts them again, and that is not a
    retraction of the paragraph above. Supervision is a process restarted
    BECAUSE it failed, which requires noticing that it did and deciding what a
    failure means; reset restarts three named processes because the operator
    asked for a new rollout, on a path where all three were healthy. Nothing
    here watches anything, and a child that dies on its own is still invisible.
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
        It exists because P0 now *waits* on P5, and `_wait` has no
        timeout by decision — so the only thing that can distinguish "still
        loading a checkpoint" from "died on a bad one" is whether the process
        is there. Asking is a fact; a timeout would be a guess about how long a
        checkpoint takes to load.
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
        has buffered work to write — P9 holds up to `FLUSH_BYTES` per stream and
        writes `manifest.json` from its own `finally` — must be given the
        chance to notice the session word and close itself.

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
