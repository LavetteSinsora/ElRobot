"""P7 for the ElRobot: the bus process. Interpolates the target ring and streams goal positions.

It replaces g1d's P1 (state), P7 (body) and P8 (hand), because all 8 servos share one
half-duplex wire that only one process may own (PORTING_PLAN.md workstream A). Every tick, at
`settings.hz` (250 Hz agreed 2026-10-17):

  1. `BusServo.step`: walk the target line to now, shape it, clamp it        (pure, ~50 us)
  2. sync_write the 8 goal positions                                          (~0.3 ms on the wire)
  3. sync_read the 8 present positions; every `slow_every`th tick the same read also covers
     speed, load and temperature                                              (~1.4 / 1.75 ms)
  4. one row into the bus state ring (what P6, the phone source and P9 read), and one into
     the in-memory log, written to CSV when the process ends.

Write before read: the command goes out on the tick boundary, and the squeeze bound uses a
measurement one tick (4 ms) old, which is fresh for a jaw.

**The guards are not P7's.** g1d's P7 has none by design: P0 decides when to damp. There is no
P0 for this robot yet, so the loop below checks three things, outside `BusServo.step`, and on
any of them stops following the ring and holds:

  * a joint more than `LAG_RAD` behind its command for `LAG_S` (blocked, or overloaded);
  * a joint more than `RUNAWAY_RAD` off its command at all: it is not obeying the goal (the
    multi-turn offset runaway), so its torque goes off and the others hold;
  * `BUS_FAILS` position reads in a row failing: the arm can't be seen, so it holds the last
    command.

The parent learns of it through `status` (a multiprocessing queue): `("ready", q0)`,
`("fault", message)`, `("done", log_path)`. Torque goes off only when the session reaches
STOPPED, i.e. when the person running it says so: never on a fault, which would drop the arm.
"""

from __future__ import annotations

import csv
import signal
from pathlib import Path

import numpy as np

from elrobot_deploy.robot.elrobot_joints import N_JOINTS, JointLimits
from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.drivers.st3215_driver import BUS_ERRORS, make_driver
from elrobot_deploy.runtime.execution.bus_servo import BusServo, ExecutorSettings
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.bus_arena import bus_layout, open_bus_rings
from elrobot_deploy.runtime.shared_memory.layout import Arena

LAG_RAD = np.radians(8.0)
LAG_S = 0.5
RUNAWAY_RAD = np.radians(20.0)
BUS_FAILS = 5

PHASES = {"hold": 0, "inside": 1, "fault": 2}

LOG = np.dtype([
    ("t", "<f8"), ("dt_ms", "<f4"), ("write_ms", "<f4"), ("read_ms", "<f4"),
    ("phase", "u1"), ("k_exec", "<i8"),
    ("line", "<f4", (N_JOINTS,)), ("cmd", "<f4", (N_JOINTS,)), ("meas", "<f4", (N_JOINTS,)),
    ("slow", "u1"), ("speed", "<f4", (N_JOINTS,)), ("load", "<f4", (N_JOINTS,)),
])


class TickLog:
    """Every tick, kept in memory (writing a file at 250 Hz would put disk latency in the loop)."""

    CHUNK = 250 * 60

    def __init__(self):
        self.chunks = [np.zeros(self.CHUNK, LOG)]
        self.n = 0

    def row(self) -> np.ndarray:
        if self.n == self.CHUNK * len(self.chunks):
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


class Guard:
    """The three checks above. `check()` returns a fault message, or None."""

    def __init__(self):
        self.lag_since = np.full(N_JOINTS, np.nan)
        self.read_fails = 0
        self.runaway: list[int] = []

    def check(self, t: float, cmd_prev: np.ndarray, meas: np.ndarray | None) -> str | None:
        if meas is None:
            self.read_fails += 1
            if self.read_fails >= BUS_FAILS:
                return f"{self.read_fails} position reads in a row failed: holding the last command"
            return None
        self.read_fails = 0
        err = np.abs(meas - cmd_prev)
        wild = np.flatnonzero(err > RUNAWAY_RAD)
        if wild.size:
            self.runaway = [int(i) for i in wild]
            return (f"joints {[int(i) + 1 for i in wild]} are "
                    f"{np.degrees(err[wild]).round(0).tolist()} deg off their command (not "
                    "obeying the goal): their torque is off, the others hold")
        late = err > LAG_RAD
        self.lag_since[~late] = np.nan
        self.lag_since[late & np.isnan(self.lag_since)] = t
        stuck = np.flatnonzero(late & (t - self.lag_since > LAG_S))
        if stuck.size:
            return (f"joints {[int(i) + 1 for i in stuck]} are more than {np.degrees(LAG_RAD):.0f} deg "
                    f"behind their command for {LAG_S} s (blocked?): holding where they are")
        return None


def run(shm_name: str, *, driver: str, driver_kw: dict, settings: ExecutorSettings,
        limits: JointLimits, status=None, log_path: str | None = None) -> None:
    """The process. Runs until `control.session` reaches STOPPED."""
    # Ctrl-C belongs to the person at the terminal, who wants the arm HELD, not dropped: the
    # parent catches it and stops writing targets; this loop keeps holding until STOPPED.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    say = (lambda *m: status.put(m)) if status is not None else (lambda *m: None)

    arena = Arena.attach(shm_name, bus_layout())
    r = open_bus_rings(arena)
    control, ring, state = r["control"], r["targets"], r["state"]
    log = TickLog()
    drv = None
    try:
        try:
            drv = make_driver(driver, **driver_kw)
            q0 = drv.connect()
        except Exception as exc:  # noqa: BLE001 - any startup failure is reported, not raised
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
        guard, fault, hold = Guard(), None, None
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
            if meas is not None:
                q_meas = meas
                state.write(t=0.5 * (t_r + t_end), q=meas, q_cmd=q, t_slow=t_end if slow else
                            np.nan, speed=speed, load=load, temp=temp, alarm=drv.alarms())

            if fault is None:
                msg = guard.check(t, q_prev, meas)
                if msg is not None:
                    fault = msg
                    hold = q_prev.copy() if meas is None else q_meas.copy()
                    if guard.runaway:
                        try:
                            drv.release(guard.runaway)
                        except BUS_ERRORS:
                            pass
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
            for what, call in (("turning torque off", drv.release), ("closing the port", drv.close)):
                try:
                    call()
                except BUS_ERRORS as exc:
                    say("fault", f"{what} failed: {exc}")
        path = None
        if log_path and log.n:
            path = str(log.save(Path(log_path)))
        say("done", path)
        arena.close()


__all__ = ["LOG", "Guard", "TickLog", "load_log", "run", "save_log"]
