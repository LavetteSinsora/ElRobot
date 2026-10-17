"""Test the executor end to end: 60 Hz sine targets into the ring, 250 Hz goals out to the servos.

    uv run python -m elrobot_deploy.cli.executor_test --sim --joints 3,4        # no hardware
    uv run python -m elrobot_deploy.cli.executor_test --joints 6 --dry-run      # plan only
    uv run python -m elrobot_deploy.cli.executor_test --joints 6                # the arm moves

Two processes, the same split as teleop will use:

  * **this process plays P6.** At `--target-hz` (60, the phone's rate) it writes one row per
    tick into the target ring: the joint angles of the planned trajectory at the capture
    instant, due at capture + `--lead-ms` (40). That is exactly how a phone pose solved by the
    IK will arrive;
  * **the bus process** (`runtime/processes/p7_bus.py`) walks that ring at `--hz` (250), shapes
    it with the `--filter-hz` (15) critically damped filter reading `2/w` ahead, clamps it and
    sync_writes the goals, reading the positions back every tick.

The trajectory: ramp from where the arm is to a centre pose (the rest pose, nudged so the sine
stays inside each joint's limits), `--cycles` of a `--amp-deg` sine at `--freq` Hz on the
chosen joints (the others hold), then ramp back to rest. Every ramp is a cosine at most
`--ramp-speed` deg/s. When it ends, the arm holds at rest until you press Enter.

Both logs land in `ElRobotRepo/logs/executor/`, and a summary is printed: loop timing, how far
the measured joints trail the target line (and the command), tracking error, and smoothness.
Ctrl-C stops sending targets; the arm holds where the line ended.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker
from pathlib import Path

import numpy as np

from elrobot_deploy.robot.elrobot_joints import N_JOINTS, JointLimits, urdf_limits
from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.execution.bus_servo import ExecutorSettings
from elrobot_deploy.runtime.processes import p7_bus
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.bus_arena import bus_layout, open_bus_rings
from elrobot_deploy.runtime.shared_memory.layout import Arena

REPO = Path(__file__).resolve().parents[4]
#: `elrobot/poses.py`'s rest, in degrees (copied so `--sim` needs no calibration files).
REST_DEG = np.array([0, -90, 0, 90, 0, 0, 0, 0], dtype=np.float64)
EDGE_DEG = 1.0                 # keep the sine this far inside the limits
MAX_RAMP_DEG = 120             # refuse a longer first ramp unless --far


# --- the trajectory ------------------------------------------------------------------------


@dataclass
class Plan:
    q0: np.ndarray            # rad, where the arm is
    centre: np.ndarray
    end: np.ndarray
    joints: list[int]         # indices with a sine
    amp: float                # rad
    freq: float               # Hz
    cycles: float
    ramp_speed: float         # rad/s, peak of each cosine ramp
    hold_s: float = 0.5
    final_hold_s: float = 1.0

    def __post_init__(self):
        self.t_ramp1 = self._ramp_time(self.q0, self.centre)
        self.t_sine = self.cycles / self.freq
        self.t_ramp2 = self._ramp_time(self.centre, self.end)
        self.edges = np.cumsum([0, self.t_ramp1, self.hold_s, self.t_sine, self.hold_s,
                                self.t_ramp2, self.final_hold_s])
        self.mask = np.zeros(N_JOINTS)
        self.mask[self.joints] = 1.0

    def _ramp_time(self, a, b) -> float:
        # A cosine blend over T peaks at (pi/2) * d / T.
        return max(1.0, 0.5 * math.pi * float(np.max(np.abs(b - a))) / self.ramp_speed)

    @property
    def duration(self) -> float:
        return float(self.edges[-1])

    @property
    def sine_window(self) -> tuple[float, float]:
        return float(self.edges[2]), float(self.edges[3])

    def at(self, t: float) -> np.ndarray:
        e = self.edges
        if t < e[1]:
            return _blend(self.q0, self.centre, t / self.t_ramp1)
        if t < e[2]:
            return self.centre.copy()
        if t < e[3]:
            s = math.sin(2 * math.pi * self.freq * (t - e[2]))
            return self.centre + self.mask * self.amp * s
        if t < e[4]:
            return self.centre.copy()
        if t < e[5]:
            return _blend(self.centre, self.end, (t - e[4]) / self.t_ramp2)
        return self.end.copy()


def _blend(a, b, u):
    s = 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))
    return a + s * (b - a)


def make_plan(q0, limits: JointLimits, joints, amp_deg, freq, cycles, ramp_speed_deg) -> Plan:
    amp = math.radians(amp_deg)
    rest = np.radians(REST_DEG)
    end = limits.clip(rest)
    centre = end.copy()
    edge = math.radians(EDGE_DEG)
    for j in joints:
        lo, hi = limits.lower[j] + amp + edge, limits.upper[j] - amp - edge
        if lo > hi:
            raise ValueError(f"joint {j + 1}: a +-{amp_deg} deg sine does not fit in its range")
        centre[j] = min(max(centre[j], lo), hi)
    return Plan(q0=np.asarray(q0, dtype=np.float64), centre=centre, end=end,
                joints=list(joints), amp=amp, freq=freq, cycles=cycles,
                ramp_speed=math.radians(ramp_speed_deg))


# --- the writer (P6's part) ---------------------------------------------------------------


def write_targets(plan: Plan, ring, control, *, target_hz: float, lead_s: float, status,
                  should_stop=lambda: False) -> tuple[np.ndarray, str | None]:
    """Write the plan into the ring at `target_hz`, each row due `lead_s` after its capture.
    Returns the rows written `(t_capture, deadline, k, q...)` and a fault message if the
    executor reported one."""
    rows, fault, k = [], None, 0
    pacer = clock.Pacer(target_hz)
    t_start = clock.now()
    try:
        while not should_stop():
            t_cap = clock.now()
            tau = t_cap - t_start
            if tau > plan.duration:
                break
            q = plan.at(tau)
            ring.write(k, t_cap + lead_s, q, intervene=control.intervene)
            rows.append((t_cap, t_cap + lead_s, k, *q))
            k += 1
            try:
                kind, *rest = status.get_nowait()
                if kind == "fault":
                    fault = rest[0]
                    break
            except queue.Empty:
                pass
            pacer.wait()
    except KeyboardInterrupt:
        fault = "Ctrl-C: no more targets; the arm holds where the line ended"
    return np.array(rows, dtype=np.float64).reshape(-1, 3 + N_JOINTS), fault


def save_targets(rows: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_capture", "deadline", "k"] + [f"q{j + 1}" for j in range(N_JOINTS)])
        for r in rows:
            w.writerow([f"{r[0]:.6f}", f"{r[1]:.6f}", int(r[2])] + [f"{x:.6f}" for x in r[3:]])


# --- the analysis -------------------------------------------------------------------------


ENCODER_STEP = 2 * math.pi / 4096


def _sine_fit(t, y, freq):
    """Least-squares `A sin(w (t - delay)) + c`: returns (A, delay s)."""
    w = 2 * math.pi * freq
    X = np.c_[np.sin(w * t), np.cos(w * t), np.ones_like(t)]
    (s, c, _), *_ = np.linalg.lstsq(X, y, rcond=None)
    return math.hypot(s, c), -math.atan2(c, s) / w


def _smoothness(log, sel, j, plan) -> tuple[float, float]:
    """(rms deg/s of the speed register around its 5-sample (100 ms) moving average, longest
    run in ms with no encoder change while the command moves at over half its peak speed)."""
    slow = sel & (log["slow"] == 1)
    sp = np.degrees(log["speed"][slow, j])
    ripple = (float(np.std((sp - np.convolve(sp, np.ones(5) / 5, mode="same"))[3:-3]))
              if sp.size > 10 else float("nan"))
    t, cmd, meas = log["t"][sel], log["cmd"][sel, j], log["meas"][sel, j]
    fast = np.abs(np.gradient(cmd, t)) > 0.5 * plan.amp * 2 * math.pi * plan.freq
    longest = run = 0
    for moved, f in zip(np.diff(meas) != 0, fast[1:]):
        run = 0 if (moved or not f) else run + 1
        longest = max(longest, run)
    period_ms = float(np.median(np.diff(t))) * 1e3 if t.size > 1 else 4.0
    return ripple, (longest + 1) * period_ms


def _reversal_dead_zone(cmd, meas, amp) -> float:
    """Median, over the sine's turns, of how far (deg) the command travels back before the
    joint has moved one encoder step the new way."""
    s = np.sign(np.diff(cmd))
    turns = np.flatnonzero(s[1:] * s[:-1] < 0) + 1
    centre = 0.5 * (cmd.max() + cmd.min())
    out = []
    for r in turns:
        if abs(cmd[r] - centre) < 0.5 * amp:          # only the sine's own peaks
            continue
        k = np.flatnonzero(s[r] * (meas[r:] - meas[r]) >= 0.99 * ENCODER_STEP)
        if k.size:
            out.append(abs(cmd[r + k[0]] - cmd[r]))
    return math.degrees(float(np.median(out))) if out else float("nan")


def analyze(log: np.ndarray, targets: np.ndarray, plan: Plan, t_start: float, *, hz: float,
            lead_s: float) -> str:
    out = []
    period_ms = 1e3 / hz
    dt = log["dt_ms"][5:]
    if dt.size:
        late = int(np.sum(dt > 1.5 * period_ms))
        out.append(f"loop      {hz:.0f} Hz target: period median {np.median(dt):.2f} ms, "
                   f"p99 {np.percentile(dt, 99):.2f}, max {dt.max():.2f}; "
                   f"{late} ticks ({100 * late / dt.size:.1f} %) over {1.5 * period_ms:.1f} ms")
        busy = log["write_ms"] + log["read_ms"]
        out.append(f"bus       write median {np.median(log['write_ms']):.2f} ms, read median "
                   f"{np.median(log['read_ms']):.2f} ms (p99 {np.percentile(log['read_ms'], 99):.2f}); "
                   f"busy {100 * np.median(busy) / period_ms:.0f} % of a period")
    if targets.size:
        gaps = np.diff(targets[:, 0]) * 1e3
        out.append(f"targets   {len(targets)} rows, every {np.median(gaps):.1f} ms "
                   f"(p99 {np.percentile(gaps, 99):.1f}), lead {lead_s * 1e3:.0f} ms")
    s0, s1 = plan.sine_window
    w0, w1 = t_start + s0 + lead_s + 0.25, t_start + s1 + lead_s
    sel = (log["t"] >= w0) & (log["t"] <= w1)
    if not sel.any() or not targets.size:
        out.append("no sine section in the log (stopped early?)")
        return "\n".join(out)
    t = log["t"][sel]
    h0, h1 = t_start + plan.edges[1] + lead_s + 0.2, t_start + plan.edges[2] + lead_s
    hold = (log["t"] >= h0) & (log["t"] <= h1)       # the centre hold just before the sine
    deg = np.degrees
    out.append("")
    out.append("joint  amp meas/cmd  delay cmd->meas  reversal dead zone  hold err  "
               "err vs line rms/max  cmd step max")
    for j in plan.joints:
        y_line = np.interp(t, targets[:, 1], targets[:, 3 + j])     # the line, at its deadlines
        cmd, meas = log["cmd"][sel, j], log["meas"][sel, j]
        a_cmd, d_cmd = _sine_fit(t, cmd, plan.freq)
        a_meas, d_meas = _sine_fit(t, meas, plan.freq)
        dz = _reversal_dead_zone(cmd, meas, plan.amp)
        herr = deg(np.mean(log["meas"][hold, j] - log["cmd"][hold, j])) if hold.any() else np.nan
        err = deg(meas - y_line)
        step = deg(np.max(np.abs(np.diff(cmd)))) if cmd.size > 1 else 0.0
        out.append(f"  {j + 1:<4} {a_meas / a_cmd:>10.3f} {1e3 * (d_meas - d_cmd):>13.0f} ms "
                   f"{dz:>15.2f} deg {herr:>+7.2f} deg {np.sqrt(np.mean(err ** 2)):>10.2f} / "
                   f"{np.max(np.abs(err)):<5.2f} deg {step:>8.3f} deg")
        ripple, stall = _smoothness(log, sel, j, plan)
        out.append(f"       smoothness: speed swing {ripple:.1f} deg/s rms around its 100 ms "
                   f"average (peak speed {math.degrees(plan.amp * 2 * math.pi * plan.freq):.1f}); "
                   f"longest stall while it should move fast {stall:.0f} ms")
    out.append("")
    out.append("delay: the servo's phase lag on the sine, as time. reversal dead zone: how far "
               "the command comes back after a turn before the joint follows by one encoder step "
               "(friction against a P-only servo shows here, and inflates the delay). hold err: "
               "measured minus commanded while holding still. err vs line: measured minus the "
               "target line at its deadlines (capture -> arm = lead + delay).")
    return "\n".join(out)


# --- main ----------------------------------------------------------------------------------


def _parse_joints(text: str) -> list[int]:
    js = sorted({int(x) for x in text.replace(" ", "").split(",") if x})
    if not js or js[0] < 1 or js[-1] > N_JOINTS:
        raise argparse.ArgumentTypeError(f"joints are 1..{N_JOINTS}, e.g. 3,4")
    return [j - 1 for j in js]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joints", type=_parse_joints, required=True, help="e.g. 6 or 3,4 (1-8)")
    ap.add_argument("--amp-deg", type=float, default=5.0)
    ap.add_argument("--freq", type=float, default=0.25, help="sine frequency, Hz")
    ap.add_argument("--cycles", type=float, default=3)
    ap.add_argument("--hz", type=float, default=250.0, help="executor write rate")
    ap.add_argument("--target-hz", type=float, default=60.0, help="rows into the ring per second")
    ap.add_argument("--lead-ms", type=float, default=40.0)
    ap.add_argument("--filter-hz", type=float, default=15.0, help="0 = no filter")
    ap.add_argument("--no-filter-lead", action="store_true")
    ap.add_argument("--velocity-scale", type=float, default=0.5)
    ap.add_argument("--ramp-speed", type=float, default=15.0, help="deg/s, peak of the ramps")
    ap.add_argument("--acceleration", type=int, default=0,
                    help="servo ACCELERATION register while streaming (0 = no servo ramp)")
    ap.add_argument("--p-coef", type=int, default=None,
                    help="P gain for the TESTED joints only, this run only (RAM, restored at the "
                         "end; the servos' own is 32)")
    ap.add_argument("--goal-speed", type=int, default=None,
                    help="GOAL_SPEED cap (steps/s, 1 step = 0.088 deg) for the TESTED joints, "
                         "instead of the high ceiling: near the commanded speed, the servo ramps "
                         "between whole-step goals itself. The sine and ramps must stay under 90 %%")
    ap.add_argument("--sim", action="store_true", help="simulated servos, no hardware")
    ap.add_argument("--port")
    ap.add_argument("--far", action="store_true",
                    help=f"allow a first ramp longer than {MAX_RAMP_DEG} deg")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
    ap.add_argument("--yes", action="store_true", help="don't ask for 'go' (only with --sim)")
    args = ap.parse_args(argv)

    limits = urdf_limits()
    settings = ExecutorSettings(hz=args.hz, filter_hz=args.filter_hz,
                                filter_lead=not args.no_filter_lead,
                                velocity_scale=args.velocity_scale)
    qdot_max = limits.velocity * settings.velocity_scale
    peak = math.radians(args.amp_deg) * 2 * math.pi * args.freq
    if peak > 0.8 * qdot_max.min():
        print(f"refusing: the sine peaks at {math.degrees(peak):.0f} deg/s, over 80 % of the "
              f"executor's limit ({math.degrees(qdot_max.min()):.0f} deg/s)")
        return 1
    if args.sim:
        driver, driver_kw = "sim", {"q0": limits.clip(np.radians(REST_DEG)).tolist()}
    else:
        driver, driver_kw = "st3215", {"port": args.port, "qdot_max": qdot_max.tolist(),
                                       "acceleration": args.acceleration}
        if args.p_coef is not None:
            if not 1 <= args.p_coef <= 128:
                print("refusing: --p-coef is 1..128 (the servos ship with 32)")
                return 1
            driver_kw["p_coef"] = {j + 1: args.p_coef for j in args.joints}
        if args.goal_speed is not None:
            if not 1 <= args.goal_speed <= 3400:
                print("refusing: --goal-speed is 1..3400 steps/s")
                return 1
            driver_kw["goal_speed"] = {j + 1: args.goal_speed for j in args.joints}

    # The current posture, read once before anything is powered (hardware), for the plan.
    if args.sim:
        q_now = np.asarray(driver_kw["q0"])
    else:
        from elrobot_deploy.runtime.drivers.st3215_driver import (
            BUS_ERRORS,
            St3215Driver,
        )
        try:
            probe = St3215Driver(port=args.port)
            q_now = probe.read()
            probe.close()
        except BUS_ERRORS as exc:
            print(f"error: could not read the arm: {exc}", file=sys.stderr)
            return 2
    try:
        plan = make_plan(q_now, limits, args.joints, args.amp_deg, args.freq, args.cycles,
                         args.ramp_speed)
    except ValueError as exc:
        print(f"refusing: {exc}")
        return 1
    if args.goal_speed is not None:
        # The cap must not be what limits the motion, or the joint falls behind its command.
        cap = args.goal_speed * ENCODER_STEP
        for j in plan.joints:
            need = max(plan.amp * 2 * math.pi * plan.freq,
                       0.5 * math.pi * abs(plan.centre[j] - plan.q0[j]) / plan.t_ramp1,
                       0.5 * math.pi * abs(plan.end[j] - plan.centre[j]) / plan.t_ramp2)
            if need > 0.9 * cap:
                print(f"refusing: joint {j + 1} needs up to {math.degrees(need):.1f} deg/s "
                      f"({need / ENCODER_STEP:.0f} steps/s), over 90 % of --goal-speed "
                      f"{args.goal_speed}. Raise it, or lower --ramp-speed / start nearer rest.")
                return 1

    deg = np.degrees
    print(f"{'joint':<7}{'now':>8}{'centre':>8}{'sine':>16}{'end':>8}   limits")
    for j in range(N_JOINTS):
        sine = (f"{deg(plan.centre[j] - plan.amp):+.1f}..{deg(plan.centre[j] + plan.amp):+.1f}"
                if j in plan.joints else "hold")
        print(f"{j + 1:<7}{deg(q_now[j]):>+8.1f}{deg(plan.centre[j]):>+8.1f}{sine:>16}"
              f"{deg(plan.end[j]):>+8.1f}   {deg(limits.lower[j]):+.1f}..{deg(limits.upper[j]):+.1f}")
    ramp = float(np.max(np.abs(deg(plan.centre - plan.q0))))
    print(f"\n{plan.duration:.1f} s: ramp {plan.t_ramp1:.1f} s (largest {ramp:.0f} deg), "
          f"{args.cycles:g} x {args.amp_deg:g} deg at {args.freq:g} Hz, ramp back {plan.t_ramp2:.1f} s."
          f"\nTargets {args.target_hz:g} Hz, lead {args.lead_ms:g} ms -> executor {args.hz:g} Hz, "
          f"filter {args.filter_hz:g} Hz{'' if settings.filter_lead else ' (no look-ahead)'}, "
          f"{'SIMULATED servos' if args.sim else 'REAL servos'}.")
    if args.dry_run:
        return 0
    if ramp > MAX_RAMP_DEG and not args.far:
        print(f"Refusing: the first ramp moves a joint {ramp:.0f} deg. Move the arm nearer rest "
              "by hand, or pass --far.")
        return 1
    if not (args.sim and args.yes):
        if not args.sim:
            print("Clear space around the arm, hand near the 12 V switch. Ctrl-C holds the arm.")
        if input("Type 'go' to start: ").strip().lower() != "go":
            print("cancelled, nothing moved")
            return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = ("sim_" if args.sim else "") + "j" + "".join(str(j + 1) for j in plan.joints)
    if args.p_coef is not None:
        tag += f"_p{args.p_coef}"
    if args.goal_speed is not None:
        tag += f"_gs{args.goal_speed}"
    log_dir = REPO / "logs" / "executor"
    exec_path = log_dir / f"{stamp}_{tag}_exec.csv"
    target_path = log_dir / f"{stamp}_{tag}_targets.csv"

    name = f"elx{os.getpid()}"
    arena = Arena.create(name, bus_layout())
    r = open_bus_rings(arena)
    control, ring = r["control"], r["targets"]
    control.session = schema.ACTIVE
    ctx = mp.get_context("spawn")
    status = ctx.Queue()
    proc = ctx.Process(target=p7_bus.run, args=(name,), name="p7_bus", daemon=False,
                       kwargs={"driver": driver, "driver_kw": driver_kw, "settings": settings,
                               "limits": limits, "status": status, "log_path": str(exec_path)})
    proc.start()
    code, rows, t_start = 0, np.zeros((0, 3 + N_JOINTS)), clock.now()
    try:
        kind, *rest = status.get(timeout=15)
        # A spawned child shares this process's resource tracker, and `Arena.attach` unregisters
        # the segment there (right for g1d's separately launched processes). Register it back,
        # so this process's `unlink()` below is the one that balances it.
        resource_tracker.register(arena.shm._name, "shared_memory")
        if kind != "ready":
            print(f"stopped: {rest[0]}")
            code = 1
        else:
            print("executor running; streaming targets ...")
            rows, fault = write_targets(plan, ring, control, target_hz=args.target_hz,
                                        lead_s=args.lead_ms / 1e3, status=status)
            if len(rows):
                t_start = rows[0, 0]          # the plan's t = 0 is the first capture
            if fault:
                print(f"stopped: {fault}")
                code = 1
            else:
                print("done; holding at rest.")
            if not args.sim:
                try:
                    input("Support the arm, then press Enter to turn torque off: ")
                except (KeyboardInterrupt, EOFError):
                    print()
    except queue.Empty:
        print("stopped: the executor did not start within 15 s")
        code = 1
    finally:
        control.session = schema.STOPPED
        saved = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                kind, *rest = status.get(timeout=0.5)
            except queue.Empty:
                if not proc.is_alive():
                    break
                continue
            if kind == "done":
                saved = rest[0]
                break
            print(f"executor: {rest[0]}")
        proc.join(timeout=5)
        arena.close()
        arena.unlink()
    if not args.sim:
        print("Torque off.")
    if len(rows):
        save_targets(rows, target_path)
        print(f"Targets: {target_path}")
    if saved:
        print(f"Executor log: {saved}\n")
        print(analyze(p7_bus.load_log(Path(saved)), rows, plan, t_start, hz=args.hz,
                      lead_s=args.lead_ms / 1e3))
    return code


if __name__ == "__main__":
    sys.exit(main())
