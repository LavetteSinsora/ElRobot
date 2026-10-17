"""P0: launcher, homing and exit.

The process `elrobot_deploy run` becomes (`cli/main.py`). It creates the arena, starts every
other process, brings the arm to the pose the run starts from, hands it to the operator, and
brings it back. Once the session says ACTIVE it only waits: there is no orchestrator inside a
rollout.

    INIT    the arena
    CHECK   the bus process (it takes the serial port and holds the measured posture), then
            the phone source; each must prove it produces before the arm moves
    HOME    a ramp to `[poses] home`
    READY   wait for START (a teleop launch starts at once: only the clutch moves the arm)
    ACTIVE  P6 runs; the operator drives while holding the clutch
    (RESET  stop P6, ramp back home, READY again)
    EXITING a ramp to `[poses] exit`, then
    STOPPED every child closes itself; the bus process releases torque if P0 said it may

**Homing and the exit go through the target ring.** P0 writes ordinary target rows and the bus
process executes them with ordinary interpolation: it never learns that homing exists, and the
ramps exercise the same deadlines, filter and clamps as every teleop tick.

**Ticks never restart.** The bus process owns the port for the whole session (restarting it
would mean releasing torque), so `k_exec` counts up across rollouts and every producer anchors
on `k_exec + 2`, the first tick it may own.

**The arm is never dropped by software.** Torque goes off only when the session ends after a
completed exit ramp, with the arm at the exit pose and the profile's `release_at_exit`. On any
other way out (a failure, a fault, a second Ctrl-C) the session still ends, but the bus process
leaves torque ON and the servos hold the last goal; P0 says so, and the operator supports the
arm and runs `python -m elrobot.relax`.
"""

from __future__ import annotations

import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.runtime import timing as _clock
from elrobot_deploy.runtime.launcher.children import _Children, _spawn_process
from elrobot_deploy.runtime.launcher.ramps import cosine_ramp, write_ramp
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all, schema

#: How often P0 polls a word it waits on.
POLL_S = 0.01
#: How long STOPPED waits for the children to close themselves before terminating them.
SHUTDOWN_JOIN_S = 10.0
#: Two Ctrl-C closer together than this are one key press (see `_install_sigint`).
DEBOUNCE_S = 0.5
#: How often a long wait reminds the terminal what it is waiting for.
REMIND_S = 5.0


class _ExitNow(KeyboardInterrupt):
    """The operator's first Ctrl-C (or an EXIT request): run the exit sequence."""


class _SkipRamp(KeyboardInterrupt):
    """The operator's second Ctrl-C: skip the exit ramp. The arm stays held where it is."""


class BusFault(RuntimeError):
    """The bus process stopped following the ring (a guard fired) or died."""


class Launcher:
    """The session sequence. `spawn`, `event` and `clock` are injectable for tests."""

    def __init__(self, cfg, *, phone: bool = True, auto_home: bool = False,
                 auto_start: bool = False, dry_run: bool = False, dashboard: bool = False,
                 terminal: bool = False, arena_name: str | None = None, clock=_clock,
                 spawn=None, event=None, log=print, log_dir: str | Path | None = None):
        self.cfg = cfg
        #: Start the phone source (P11). Without it nothing ever drives the arm: a launch that
        #: only homes and exits (a bring-up check).
        self.phone = phone
        #: Ramp home without waiting for a HOME request (the arm moves right after CHECK).
        self.auto_home = auto_home
        #: READY -> ACTIVE without waiting for a START request.
        self.auto_start = auto_start
        #: Torque stays off; every command is computed and published but never written.
        self.dry_run = dry_run
        #: Start the dashboard (P10).
        self.dashboard = dashboard
        #: Also take HOME / START from the terminal (Enter), beside the dashboard's buttons.
        self.terminal = terminal
        self.name = arena_name or f"elr{os.getpid()}"
        self.clock = clock
        self.log = log
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.children = _Children(spawn or _spawn_process, event)
        self.arena = None
        self.rings = None
        self._interrupts = 0
        self._skip_ramp = False
        self._failure: BaseException | None = None
        self.stamp = time.strftime("%Y%m%d_%H%M%S")

    # --- helpers -------------------------------------------------------------------------

    def _set(self, state: int) -> None:
        self.rings["control"].session = state
        self.log(f"[P0] {schema.SESSION_NAMES[state]}")

    def _wait(self, predicate, what: str = "", remind=None) -> None:
        """Poll until true. No timeout, by decision: every wait ends on the fact it waits for,
        or on the death of the process that would have produced it (the predicate checks)."""
        t_next = self.clock.now() + REMIND_S
        while not predicate():
            now = self.clock.now()
            if what and now >= t_next:
                self.log(f"[P0] still waiting for {what}" + (f" ({remind()})" if remind else ""))
                t_next = now + REMIND_S
            self.clock.sleep_until(now + POLL_S)

    def _take_request(self) -> int:
        """The pending request, consumed (read and cleared in one step). Enter on the terminal
        (`terminal`) is the request P0 is waiting for (`control.wait`): HOME, then START."""
        control = self.rings["control"]
        request = control.request
        if request == schema.REQ_NONE and self._enter_pressed():
            request = {schema.WAIT_HOME: schema.REQ_HOME,
                       schema.WAIT_START: schema.REQ_START}.get(control.wait, schema.REQ_NONE)
        if request != schema.REQ_NONE:
            control.request = schema.REQ_NONE
            self.log(f"[P0] request {schema.REQUEST_NAMES.get(request, request)}")
        return request

    def _enter_pressed(self) -> bool:
        """A line typed on the terminal, without blocking (never in tests)."""
        if not self.terminal:
            return False
        import select
        import sys
        try:
            if not sys.stdin.isatty():
                # Only a person at a terminal presses Enter. A pipe or /dev/null is readable
                # at once (end of file), which must never read as "go".
                self.terminal = False
                return False
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline() != ""          # "" = end of file, not a line
        except (OSError, ValueError):
            self.terminal = False
        return False

    def _await_request(self, wait: int, wanted: int, prompt: str) -> None:
        """Hold until `wanted` is requested (dashboard button, or Enter). EXIT leaves."""
        control = self.rings["control"]
        control.wait = wait
        self.log(f"[P0] {prompt}")
        t_next = self.clock.now() + 30.0
        try:
            while True:
                request = self._take_request()
                if request == wanted:
                    return
                if request == schema.REQ_EXIT:
                    raise _ExitNow
                self._check_bus()
                self._note_phone()
                now = self.clock.now()
                if now >= t_next:
                    self.log(f"[P0] still waiting: {prompt}")
                    t_next = now + 30.0
                self.clock.sleep_until(now + POLL_S)
        finally:
            control.wait = schema.WAIT_NONE

    def _how(self) -> str:
        dash = self.dashboard and self.children.alive("p10_dashboard")
        ways = (["the dashboard's button"] if dash else []) + \
               (["Enter here"] if self.terminal else [])
        return " or ".join(ways) or "a request"

    def _check_bus(self) -> None:
        """Raise if the bus process has stopped following the ring."""
        control = self.rings["control"]
        if control.bus in (schema.BUS_HOLDING, schema.BUS_DONE, schema.BUS_STARTING) or \
                not self.children.alive("p7_bus"):
            raise BusFault(f"the bus process is {schema.BUS_NAMES.get(control.bus, '?')}: "
                           f"{control.fault()[1] or 'it exited'}")

    def _held(self) -> np.ndarray:
        """The 8 joints the bus process is commanding right now (what a ramp starts from)."""
        snap = self.rings["state"].latest()
        if snap is None:
            raise ConfigError("no state rows: the bus process is not reading the arm")
        return np.asarray(snap["q_cmd"], dtype=float).copy()

    def _measured(self) -> np.ndarray:
        snap = self.rings["state"].latest()
        if snap is None:
            raise ConfigError("no state rows: the bus process is not reading the arm")
        return np.asarray(snap["q"], dtype=float).copy()

    def _await_even(self) -> None:
        """Before P0 writes a ramp: the phone source hands back (an even counter) once the
        session leaves ACTIVE. A ramp stamped with an odd counter would be abandoned the moment
        it handed back. Not waited for if the phone source is gone (then the counter can no
        longer change, and the ramp is stamped with whatever it is)."""
        control = self.rings["control"]
        self._wait(lambda: control.intervene % 2 == 0 or not self.children.alive("p11_phone"),
                   "the phone source to release the clutch")

    def _ramp_to(self, q_to, what: str) -> None:
        """A cosine ramp from the held command to `q_to`, forward from `k_exec + 2`, waited
        out to its last deadline."""
        cfg, control = self.cfg, self.rings["control"]
        q_from = self._held()
        qs = cosine_ramp(q_from, q_to, dt=cfg.dt, qdot_max=cfg.qdot_max,
                         min_s=cfg.poses.min_s, margin=cfg.poses.velocity_margin)
        k_first = control.k_exec + 2
        self.log(f"[P0] {what} over {len(qs)} ticks ({len(qs) * cfg.dt:.1f} s)")
        written = write_ramp(self.rings, k_first, qs, t_first=self.clock.now() + 2 * cfg.dt,
                             dt=cfg.dt, lead_ticks=self.rings["targets"].n // 2,
                             clock=self.clock)
        if written:
            end = written[-1][1]
            while self.clock.now() < end:
                self._check_bus()
                self.clock.sleep_until(min(end, self.clock.now() + 0.05))

    # --- the sequence --------------------------------------------------------------------

    def init(self) -> None:
        layout = build()
        self.arena = Arena.create(self.name, layout)
        self.rings = open_all(self.arena)
        self._set(schema.INIT)
        self.log(f"[P0] arena {layout.size / 1e6:.2f} MB")
        if self.dashboard:
            # First, so the operator can watch CHECK. It only reads the arena and stores
            # requests; it never decides anything.
            self.children.start("p10_dashboard",
                                "elrobot_deploy.dashboard.server.run_session",
                                (self.name, self.cfg), dry_run=self.dry_run)
            self.log(f"[P0] dashboard: http://{self.cfg.dashboard.host}:"
                     f"{self.cfg.dashboard.port}")

    def check(self) -> None:
        """Start the producers and confirm they produce, before anything moves."""
        self._set(schema.CHECK)
        log_path = None
        if self.log_dir is not None:
            log_path = str(self.log_dir / f"{self.stamp}_bus.csv")
        self.children.start("p7_bus", "elrobot_deploy.runtime.processes.p7_bus.run_config",
                            (self.name, self.cfg), gated=True, log_path=log_path,
                            dry_run=self.dry_run)
        ready = self.children.ready["p7_bus"]
        control = self.rings["control"]
        self._wait(lambda: ready.is_set() or control.bus == schema.BUS_DONE
                   or not self.children.alive("p7_bus"), "the bus process to connect")
        self._log_report()
        if not ready.is_set():
            raise ConfigError(f"the bus process did not start: {control.fault()[1] or 'it exited'}")
        q = self._measured()
        self.log(f"[P0] bus running ({self.cfg.bus.driver}"
                 f"{', DRY RUN: torque off, nothing is written' if self.dry_run else ''}); "
                 f"arm at {np.round(np.degrees(q), 1).tolist()} deg")
        if self.phone:
            # Started now, waited for never: only the clutch moves the arm, so a phone that is
            # not streaming yet must not hold up the check or homing. The dashboard shows it,
            # and `_note_phone` says it here once it appears.
            self.children.start("p11_phone", "elrobot_deploy.runtime.processes.p11_phone.run",
                                (self.name, self.cfg))
            self.log(f"[P0] phone source listening on UDP {self.cfg.teleop.port}: connect the "
                     f"app to this Mac")

    def _note_phone(self) -> None:
        """Say once when the phone's first packet arrives (and if the phone source died)."""
        if not self.phone or getattr(self, "_phone_seen", False):
            return
        if int(self.rings["intervene"].meta()["samples"]) > 0:
            self._phone_seen = True
            self.log("[P0] phone streaming")
        elif not self.children.alive("p11_phone"):
            raise ConfigError("the phone source exited (its traceback is above)")

    def _log_report(self) -> None:
        """The bus process's hardware check, as a table on the terminal."""
        import json
        seq, text = self.rings["control"].report()
        if not seq:
            return
        rep = json.loads(text)
        self.log("[P0] hardware check:  servo   angle    supply   temp   P  D  torque-limit")
        for r in rep["servos"]:
            if not r.get("ok"):
                self.log(f"[P0]                   {r['id']}   no reply")
                continue
            flags = f"  {', '.join(r['status'])}" if r["status"] else ""
            self.log(f"[P0]                   {r['id']}  {r['deg']:+7.1f}   {r['volts']:5.1f} V"
                     f"  {r['temp']:4.0f} C  {r['p']:2d} {r['d']:2d}  {r['torque_limit']:5d}"
                     f"{flags}")
        for w in rep["warnings"]:
            self.log(f"[P0] warning: {w}")
        for p in rep["problems"]:
            self.log(f"[P0] PROBLEM: {p}")

    def confirm_home(self) -> None:
        """Hold the arm where it was found until the operator has seen where home is."""
        if self.auto_home:
            return
        self._await_request(schema.WAIT_HOME, schema.REQ_HOME,
                            f"the arm is held where it is. Home pose: "
                            f"{np.round(np.degrees(self.cfg.poses.home), 1).tolist()} deg "
                            f"(the dashboard shows it). Press {self._how()} to move there; "
                            f"Ctrl-C to exit.")

    def home(self) -> None:
        self._set(schema.HOME)
        self._await_even()
        self._ramp_to(self.cfg.poses.home, "homing")
        if self.dry_run:
            return                       # the arm was never driven: nothing to check
        q = self._measured()
        off = np.abs(q - self.cfg.poses.home)[:schema.ARM_DOF]
        if np.any(off > self.cfg.poses.home_tol):
            worst = int(np.argmax(off))
            raise ConfigError(
                f"homing ended with joint {worst + 1} {np.degrees(off[worst]):.1f} deg from "
                f"home (tolerance {np.degrees(self.cfg.poses.home_tol):.1f} deg): blocked, or "
                f"sagging under load")

    def ready(self) -> None:
        self._set(schema.READY)
        if self.auto_start:
            return
        self._await_request(schema.WAIT_START, schema.REQ_START,
                            f"READY: the arm holds at home. Press {self._how()} to START "
                            f"(then hold the clutch on the phone to drive); Ctrl-C to exit.")

    def active(self) -> None:
        """Spawn P6 and hand the arm over. Returns on RESET; EXIT leaves by raising."""
        self._set(schema.ACTIVE)
        self.children.start("p6_target", "elrobot_deploy.runtime.processes.p6_target.run",
                            (self.name, self.cfg))
        self.log("[P0] ACTIVE: hold the clutch on the phone to drive the arm. "
                 "Ctrl-C to exit (twice to stop where it is).")
        while True:
            request = self._take_request()
            if request == schema.REQ_RESET:
                return
            if request == schema.REQ_EXIT:
                raise _ExitNow
            self._check_bus()
            self._note_phone()
            if not self.children.alive("p6_target"):
                raise ConfigError("P6 exited (its traceback is above)")
            if self.phone and not self.children.alive("p11_phone"):
                raise ConfigError("the phone source exited (its traceback is above)")
            self.clock.sleep_until(self.clock.now() + POLL_S)

    def _stop_p6(self) -> None:
        # The word has already left ACTIVE, which is how P6 ends itself; two ticks is enough for
        # a solve in flight to land. Terminating it mid-write would leave a seqlock odd.
        self.clock.sleep_until(self.clock.now() + 2 * self.cfg.dt)
        self.children.wait("p6_target", timeout=1.0)
        self.children.stop("p6_target")

    def reset(self) -> None:
        """End the rollout: stop P6, ramp home, READY again (the bus keeps holding throughout)."""
        self._set(schema.HOME)
        self._stop_p6()
        self.home()

    def exit(self) -> None:
        """EXITING -> exit ramp -> STOPPED, in that order, always."""
        if self.rings is None:
            return
        control = self.rings["control"]
        self._set(schema.EXITING)
        ramped = False
        try:
            self._stop_p6()
            if not self._skip_ramp:
                self._check_bus()
                self._await_even()
                self._ramp_to(self.cfg.poses.exit, "exit ramp")
                ramped = True
        except _SkipRamp:
            self.log("[P0] second interrupt: skipping the exit ramp")
        except (BusFault, ConfigError) as exc:
            self.log(f"[P0] no exit ramp: {exc}")
        except Exception:                                   # noqa: BLE001
            self.log(traceback.format_exc())
        release = ramped and self.cfg.poses.release_at_exit
        control.release_torque = release
        # STOPPED before terminating anything: every child loops on `session < STOPPED`, so
        # this word is their exit signal and the only path on which their `finally` runs (the
        # bus process's torque-off and log).
        self._set(schema.STOPPED)
        late = self.children.wait(*list(self.children.handles), timeout=SHUTDOWN_JOIN_S)
        if late:
            self.log(f"[P0] {', '.join(sorted(late))} did not exit within "
                     f"{SHUTDOWN_JOIN_S:.0f} s of STOPPED: terminating")
        self.children.stop_all()
        if self.dry_run:
            self.log("[P0] dry run: torque was never on")
        elif self.cfg.bus.driver != "sim":
            if release and control.bus == schema.BUS_DONE and not control.fault()[1].startswith(
                    "turning torque off failed"):
                self.log("[P0] torque off (confirmed by the bus process)")
            else:
                self.log("[P0] TORQUE IS STILL ON: the servos hold the last goal. Support the "
                         "arm, then run `uv run python -m elrobot.relax`.")

    # --- lifecycle -----------------------------------------------------------------------

    def _install_sigint(self) -> None:
        t_first = []

        def handler(signum, frame):
            now = time.monotonic()
            # One Ctrl-C can arrive twice: under `uv run` both uv (which forwards it) and this
            # process are in the terminal's foreground group. A second one within
            # `DEBOUNCE_S` is that same key press, not a second decision.
            if t_first and now - t_first[0] < DEBOUNCE_S:
                return
            self._interrupts += 1
            if self._interrupts == 1:
                t_first.append(now)
                raise _ExitNow
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise _SkipRamp

        signal.signal(signal.SIGINT, handler)

    def _teardown(self) -> None:
        if self.arena is not None:
            self.arena.close()
            self.arena.unlink()

    def run(self) -> int:
        """The whole session. Returns 0 after a clean exit, 1 after a refusal or a bus fault;
        anything unexpected is re-raised once the session has ended."""
        self.init()
        self._install_sigint()
        try:
            self.check()
            self.confirm_home()
            self.home()
            while True:
                self.ready()
                self.active()
                self.reset()
        except _ExitNow:
            pass
        except _SkipRamp:
            self._skip_ramp = True
        except BaseException as exc:                        # noqa: BLE001
            # Whatever went wrong, do not move the arm on a failure nobody understands: skip
            # the ramp, end the session, leave it held.
            self._skip_ramp = True
            self._failure = exc
            self.log(f"[P0] {type(exc).__name__}: {exc}")
            if not isinstance(exc, (ConfigError, BusFault)):
                self.log(traceback.format_exc())
        finally:
            try:
                self.exit()
            finally:
                self._teardown()
        if self._failure is None:
            return 0
        if isinstance(self._failure, (ConfigError, BusFault)):
            return 1
        raise self._failure


def run(cfg, **kwargs) -> int:
    return Launcher(cfg, **kwargs).run()


__all__ = ["BusFault", "Launcher", "run"]
