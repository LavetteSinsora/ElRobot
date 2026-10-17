"""P7, the bus process. Interpolates the target ring and streams goal positions.

It replaces g1d's P1 (state), P7 (body) and P8 (hand), because all 8 servos share one
half-duplex wire that only one process may own (PORTING_PLAN.md workstream A). P0 starts it at
CHECK and it runs, holding the port, until the session reaches STOPPED; it is never restarted
inside a session (that would mean releasing torque), so `k_exec` never rewinds. Every tick, at
`settings.hz` (250 Hz agreed 2026-10-17):

  1. `BusServo.step`: walk the target line to now, shape it, clamp it        (pure, ~50 us)
  2. sync_write the 8 goal positions                                          (~0.3 ms on the wire)
  3. sync_read the 8 present positions; every `slow_every`th tick the same read also covers
     speed, load and temperature                                              (~1.4 / 1.75 ms)
  4. one row into the state ring (what P0, P6 and the phone source read), and one into the
     in-memory log, written to CSV when the process ends.

Write before read: the command goes out on the tick boundary, and the squeeze bound uses a
measurement one tick (4 ms) old, which is fresh for a jaw.

**The guards** (`execution/guard.py`, thresholds in the profile's `[safety]`): a joint
running away (far off its command and moving further away: its torque goes off at once), a
joint blocked (far behind for a long time), the arm unreadable (reads failing). On a runaway or
a blocked joint, with `recover` (P0's sessions), the loop keeps following the ring and says
`BUS_RECOVERING`: P0 then ramps the arm home and to rest, and torque goes off there. On
unreadable reads, or a second fault during a recovery, or without `recover`, it stops following
the ring and HOLDS (`BUS_HOLDING`): the session ends with torque on.

The parent learns of it through the arena (`control.bus`, `control.fault`), the `ready` latch P0
waits on, and optionally `status` (a multiprocessing queue, for `cli/executor_test.py`):
`("ready", q0)`, `("fault", message)`, `("done", log_path)`. Torque goes off only when the
session reaches STOPPED, i.e. when the person running it says so (and only if `release`): never
on a fault, which would drop the arm.
"""

from __future__ import annotations

import csv
import json
import signal
from pathlib import Path

import numpy as np

from elrobot_deploy.robot.elrobot_joints import N_JOINTS, JointLimits
from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.drivers.st3215_driver import BUS_ERRORS, DriverError, make_driver
from elrobot_deploy.runtime.execution.bus_servo import BusServo, ExecutorSettings
from elrobot_deploy.runtime.execution.guard import Guard, GuardSettings
from elrobot_deploy.runtime.shared_memory import Arena, build, layout_for, open_all, schema

PHASES = {"hold": 0, "inside": 1, "fault": 2}

LOG = np.dtype([
    ("t", "<f8"), ("dt_ms", "<f4"), ("write_ms", "<f4"), ("read_ms", "<f4"),
    ("phase", "u1"), ("k_exec", "<i8"),
    ("line", "<f4", (N_JOINTS,)), ("cmd", "<f4", (N_JOINTS,)), ("meas", "<f4", (N_JOINTS,)),
    ("slow", "u1"), ("speed", "<f4", (N_JOINTS,)), ("load", "<f4", (N_JOINTS,)),
])


class TickLog:
    """Every tick, kept in memory (writing a file at 250 Hz would put disk latency in the loop).

    `max_s`: keep only the newest this many seconds (whole one-minute chunks are dropped), so a
    long teleop session neither grows without bound nor takes minutes to save. None = keep all.
    """

    CHUNK = 250 * 60

    def __init__(self, max_s: float | None = None, hz: float = 250.0):
        self.chunks = [np.zeros(self.CHUNK, LOG)]
        self.n = 0                       # rows in `chunks`
        self.max_chunks = None if max_s is None else max(2, int(np.ceil(max_s * hz / self.CHUNK)) + 1)

    def row(self) -> np.ndarray:
        if self.n == self.CHUNK * len(self.chunks):
            if self.max_chunks is not None and len(self.chunks) >= self.max_chunks:
                self.chunks.pop(0)
                self.n -= self.CHUNK
            self.chunks.append(np.zeros(self.CHUNK, LOG))
        r = self.chunks[-1][self.n % self.CHUNK]
        self.n += 1
        return r

    def array(self) -> np.ndarray:
        return np.concatenate(self.chunks)[: self.n]

    def save(self, path: Path) -> Path:
        save_log(self.array(), path)
        return path


def save_log(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ["t", "dt_ms", "write_ms", "read_ms", "phase", "k_exec", "slow"]
    cols = [f"{k}{j + 1}" for k in ("line", "cmd", "meas", "speed", "load")
            for j in range(N_JOINTS)]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head + cols)
        for r in a:
            w.writerow([f"{r['t']:.6f}", f"{r['dt_ms']:.3f}", f"{r['write_ms']:.3f}",
                        f"{r['read_ms']:.3f}", int(r["phase"]), int(r["k_exec"]), int(r["slow"])]
                       + [f"{x:.6f}" for k in ("line", "cmd", "meas", "speed", "load")
                          for x in r[k]])


def load_log(path: Path) -> np.ndarray:
    """A saved log back as the `LOG` array (for the test's analysis, or a notebook)."""
    with Path(path).open() as f:
        rows = list(csv.reader(f))
    head, body = rows[0], np.array(rows[1:], dtype=np.float64)
    col = {h: i for i, h in enumerate(head)}
    out = np.zeros(len(body), LOG)
    for k in ("t", "dt_ms", "write_ms", "read_ms", "phase", "k_exec", "slow"):
        out[k] = body[:, col[k]]
    for k in ("line", "cmd", "meas", "speed", "load"):
        out[k] = body[:, [col[f"{k}{j + 1}"] for j in range(N_JOINTS)]]
    return out


def run_config(shm_name: str, cfg, *, ready=None, log_path: str | None = None,
               dry_run: bool = False) -> None:
    """The process as P0 starts it: everything from the run's `Config`. The tick log keeps the
    session's last ten minutes."""
    run(shm_name, driver=cfg.bus.driver, driver_kw=cfg.driver_kw(), settings=cfg.executor,
        limits=cfg.limits, ready=ready, log_path=log_path, release=None, log_max_s=600.0,
        dry_run=dry_run, check=True, guard=cfg.guard, layout=layout_for(cfg))


def run(shm_name: str, *, driver: str, driver_kw: dict, settings: ExecutorSettings,
        limits: JointLimits, status=None, ready=None, log_path: str | None = None,
        release: bool | None = True, log_max_s: float | None = None, dry_run: bool = False,
        check: bool = False, guard: GuardSettings | None = None, layout=None) -> None:
    """The process. Runs until `control.session` reaches STOPPED.

    `check`: run the driver's hardware check first, publish it (`control.report`) and refuse
    to start on any problem. `dry_run`: torque stays OFF and no goal is ever written; the loop
    still walks the ring, shapes and clamps every command and publishes it as `q_cmd` beside the
    measurement, so everything upstream runs for real while the arm stays limp. The lag and
    runaway guards are off (the arm is not meant to follow).

    `ready` (P0's latch) is set once the first goal is on the wire. `release`: turn torque off
    at STOPPED (False: the servos keep holding the last goal after the process ends; None: ask
    the arena, `control.release_torque`, which P0 sets only after a completed exit ramp).
    `guard`: the thresholds and whether a fault recovers (None: hold on any fault, the
    executor test's behaviour)."""
    # Ctrl-C belongs to the person at the terminal, who wants the arm HELD, not dropped: the
    # parent catches it and stops writing targets; this loop keeps holding until STOPPED.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    arena = Arena.attach(shm_name, layout or build())
    r = open_all(arena)
    control, ring, state = r["control"], r["targets"], r["state"]

    def say(kind, *m):
        if kind == "fault" and m:
            control.set_fault(str(m[0]))
            if control.bus in (schema.BUS_RUNNING, schema.BUS_DRY, schema.BUS_RECOVERING):
                control.bus = schema.BUS_HOLDING
        if status is not None:
            status.put((kind, *m))

    log = TickLog(log_max_s, settings.hz)
    drv, connected = None, False
    try:
        try:
            drv = make_driver(driver, **driver_kw)
            if check:
                report = drv.check()
                report["dry_run"] = bool(dry_run)
                control.set_report(json.dumps(report))
                if report["problems"]:
                    raise DriverError("the hardware check failed: "
                                      + "; ".join(report["problems"]))
            q0 = drv.connect(passive=dry_run)
            connected = not dry_run
        except Exception as exc:  # noqa: BLE001 - any startup failure is reported, not raised
            # Not ours to release: a refused connect may mean ANOTHER program holds the arm, and
            # turning its torque off would drop it. (A runaway at enable released inside connect.)
            say("fault", f"could not start the bus: {exc}")
            return
        nan = np.full(N_JOINTS, np.nan)
        speed, load, temp = nan, nan, nan
        t = clock.now()
        state.write(t=t, q=q0, q_cmd=q0, t_slow=np.nan, speed=nan, load=nan, temp=nan,
                    alarm=drv.alarms())
        servo = BusServo(ring=ring, control=control, limits=limits, settings=settings,
                         q0=q0, t0=t)
        q_meas, q_prev = q0, q0.copy()
        guard_settings = guard if guard is not None else GuardSettings(recover=False)
        guard, fault, hold = Guard(guard_settings), None, None
        control.bus = schema.BUS_DRY if dry_run else schema.BUS_RUNNING
        say("ready", q0.tolist())

        pacer = clock.Pacer(settings.hz)
        tick, t_last = 0, t
        while control.session < schema.STOPPED:
            if fault is None:
                q = servo.step(t, q_meas)
                if q is None:            # a segment was latched: iterate again, no wait
                    t = clock.now()
                    continue
            else:
                q = hold
            t_w = clock.now()
            try:
                if not dry_run:
                    drv.write(q)
            except BUS_ERRORS as exc:
                if fault is None:
                    fault, hold = f"bus write failed: {exc}", q_prev
                    say("fault", fault)
            t_r = clock.now()
            slow = tick % settings.slow_every == 0
            try:
                if slow:
                    meas, speed, load, temp = drv.read_full()
                else:
                    meas = drv.read()
            except BUS_ERRORS:
                meas = None              # counted by the guard
            t_end = clock.now()
            if tick == 0 and ready is not None:
                ready.set()              # the first goal is on the wire: P0 may write targets
            if meas is not None:
                q_meas = meas
                state.write(t=0.5 * (t_r + t_end), q=meas, q_cmd=q, t_slow=t_end if slow else
                            np.nan, speed=speed, load=load, temp=temp, alarm=drv.alarms())

            if fault is None and (not dry_run or meas is None):     # dry: only lost reads
                f = guard.check(t, q_prev, meas)
                if f is not None:
                    if f.kind == "runaway":
                        guard.ignore.update(f.joints)
                        try:
                            drv.release(f.joints)
                        except BUS_ERRORS:
                            pass
                    if f.kind != "read" and guard_settings.recover and not guard.recovering:
                        # Keep following the ring: P0 brings the arm home, then to rest.
                        guard.recovering = True
                        control.set_fault(f.message)
                        control.bus = schema.BUS_RECOVERING
                        if status is not None:
                            status.put(("recovering", f.message))
                    else:
                        fault = f.message
                        hold = q_prev.copy() if meas is None else q_meas.copy()
                        say("fault", fault)
            q_prev = np.asarray(q, dtype=np.float64).copy()

            row = log.row()
            row["t"], row["dt_ms"] = t, (t - t_last) * 1e3
            row["write_ms"], row["read_ms"] = (t_r - t_w) * 1e3, (t_end - t_r) * 1e3
            row["phase"] = PHASES["fault"] if fault else PHASES[servo.phase]
            row["k_exec"] = control.k_exec
            row["line"], row["cmd"] = servo.q_line_last, q
            row["meas"] = q_meas
            row["slow"] = slow
            if slow:
                row["speed"], row["load"] = speed, load
            t_last, tick = t, tick + 1
            t = pacer.wait()
    finally:
        if drv is not None:
            if release is None:
                release = control.release_torque
            steps = [("turning torque off", drv.release)] if connected and release else []
            if connected and not release:
                say("held", "torque left ON: the servos hold the last goal")
            for what, call in [*steps, ("closing the port", drv.close)]:
                try:
                    call()
                except BUS_ERRORS as exc:
                    say("fault", f"{what} failed: {exc}")
        path = None
        if log_path and log.n:
            path = str(log.save(Path(log_path)))
        control.bus = schema.BUS_DONE
        say("done", path)
        arena.close()


__all__ = ["LOG", "Guard", "GuardSettings", "TickLog", "load_log", "run", "run_config", "save_log"]
