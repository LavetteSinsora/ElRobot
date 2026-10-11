"""Running a producer from a dashboard: one subprocess at a time.

A subprocess and not a thread, and this is not a style choice. A policy replay
imports the adapter's framework, which may preallocate most of the GPU (JAX
with `XLA_PYTHON_CLIENT_PREALLOCATE=true` does) — the moment that happens
in-process the dashboard owns the GPU for as long as it runs, and every later
replay OOMs. A subprocess takes the GPU, finishes, and gives it back. It also
lets a producer run under another interpreter (the adapter's).

**One at a time**, for the same reason: two producers would each try to
preallocate most of the card. The queue is therefore a queue and not a
pool.

Progress crosses the boundary as a JSON file the producer rewrites atomically
(`os.replace` onto itself), polled here. Stdout goes to a log beside it, so a
failed run can be read after the fact instead of being a red box with no
detail. A mode submits a `Job` with the command already built — what the
arguments mean is the mode's business, and the producer's.

Used by the inspection modes (`docs/deploying_a_policy/rollout_inspection.md`,
"Writing a mode").
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    id: str
    kind: str                               # the mode's name for the log
    label: str                              # what the panel shows
    cmd: list[str]
    cwd: str
    status: str = "queued"                  # queued|running|done|failed|cancelled
    returncode: int | None = None
    error: str = ""
    pid: int | None = None
    t_submit: float = field(default_factory=time.time)
    t_start: float = 0.0
    t_end: float = 0.0
    log: str = ""
    progress_path: str = ""
    meta: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "label": self.label,
            "status": self.status, "returncode": self.returncode,
            "error": self.error, "log": self.log,
            "elapsed_s": round((self.t_end or time.time())
                               - (self.t_start or self.t_submit), 1),
            "progress": self.progress(),
            **self.meta,
        }

    def progress(self) -> dict:
        if not self.progress_path:
            return {}
        try:
            return json.loads(Path(self.progress_path).read_text())
        except (OSError, ValueError):
            return {}


class Runner:
    """A one-deep worker thread that spawns producers."""

    def __init__(self):
        self._q: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._current: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # --- api ---------------------------------------------------------------

    def submit(self, *, kind: str, label: str, cmd: list[str], cwd: Path | str,
               jobs_dir: Path, progress_arg: str | None = "--progress-file",
               meta: dict | None = None) -> Job:
        """Queue `cmd`. `jobs_dir` gets `<id>.log` and `<id>.progress.json`;
        the latter is appended to `cmd` as `<progress_arg> <path>` when the
        producer takes one."""
        job = Job(id=uuid.uuid4().hex[:8], kind=kind, label=label, cmd=list(cmd),
                  cwd=str(cwd), meta=dict(meta or {}))
        jobs_dir = Path(jobs_dir)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job.log = str(jobs_dir / f"{job.id}.log")
        job.progress_path = str(jobs_dir / f"{job.id}.progress.json")
        if progress_arg:
            job.cmd += [progress_arg, job.progress_path]
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._q.put(job)
        return job

    def jobs(self, limit: int = 20, kind: str | None = None) -> list[Job]:
        with self._lock:
            rows = [self._jobs[i] for i in self._order]
        if kind is not None:
            rows = [j for j in rows if j.kind == kind]
        return rows[-limit:]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def busy(self) -> bool:
        with self._lock:
            return any(j.status in ("queued", "running")
                       for j in self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            return True
        if job.status == "running" and self._current is not None:
            # The process group, not the process: a launcher is a shell that
            # execs python, and a bare SIGTERM to the shell can orphan it.
            try:
                os.killpg(os.getpgid(self._current.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                return False
            job.status = "cancelled"
            return True
        return False

    # --- internals ---------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job.status == "cancelled":
                continue
            job.status, job.t_start = "running", time.time()
            try:
                with open(job.log, "wb") as log:
                    proc = subprocess.Popen(
                        job.cmd, cwd=job.cwd, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"})
                    self._current, job.pid = proc, proc.pid
                    job.returncode = proc.wait()
            except Exception as exc:        # noqa: BLE001
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            else:
                if job.status != "cancelled":
                    job.status = "done" if job.returncode == 0 else "failed"
                if job.status == "failed" and not job.error:
                    job.error = tail(job.log)
            finally:
                self._current = None
                job.t_end = time.time()


def tail(path: str, n: int = 12) -> str:
    try:
        lines = Path(path).read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


#: One runner per process: the GPU and the cores are per box, not per mode.
RUNNER = Runner()

__all__ = ["RUNNER", "Job", "Runner", "tail"]
