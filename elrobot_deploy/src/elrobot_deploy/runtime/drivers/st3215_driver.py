"""The wire under the executor: 8 ST3215 servos on one serial bus, or a simulation of them.

Everything above this file speaks radians in the URDF's convention, in servo order. Everything
below it is `elrobot.st3215` (packets) and `elrobot.calibration` (reading <-> angle). One
process owns the port: the bus is half-duplex, so reads and writes cannot come from different
processes the way g1d's P1 and P7 used separate DDS topics (PORTING_PLAN.md workstream A).

The servo settings while streaming, and why:

  * **acceleration 0**: the servo's own speed ramp is off. The executor already shapes the
    motion; a ramp restarted on every goal is what made `elrobot.move --stream` stop and start
    (2026-10-11 log). On the STS series 0 is documented as "no ramp"; the first log confirms it.
  * **goal speed = a ceiling**, above the executor's own limit (`goal_speed_factor` x its
    `qdot_max`). Never 0, which on this servo means full speed.
  * **torque limit** per joint, as `elrobot.move` uses (30-50 %), until the arm has been run
    harder on purpose.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
from elrobot.st3215.bus import BusError

from elrobot_deploy.robot.elrobot_joints import N_JOINTS, SERVO_IDS

#: 0.1 % of stall torque, per servo ID: `elrobot.move`'s values.
TORQUE_LIMIT = {1: 500, 2: 500, 3: 500, 4: 500, 5: 400, 6: 400, 7: 400, 8: 300}
ENABLE_WATCH_S = 0.3        # after torque on, how long every joint is watched for a runaway
RUNAWAY_RAD = np.radians(1.5)

# The hardware check (`St3215Driver.check`), before anything is written.
VOLTAGE_RANGE = (10.5, 13.2)    # V: a 12 V supply, unloaded
TEMP_WARN_C, TEMP_MAX_C = 50.0, 65.0
# How far outside its calibrated range a joint may read and still be accepted: the ranges are
# the measured stops pulled in by 3-5 deg (`elrobot/calibration.py`), and the arm at rest lies
# ON its stops (the shoulder; the gripper fully open). The multi-turn trap is off by far more.
RANGE_MARGIN_STEPS = 91         # 8 deg


class DriverError(RuntimeError):
    pass


#: What a call on the wire can raise: a servo that didn't answer (`BusError`), or the port
#: itself (pyserial's `SerialException` is an `OSError`).
BUS_ERRORS = (BusError, OSError, DriverError)


class ServoDriver(Protocol):
    def check(self) -> dict:
        """Read every servo before anything is written: `{"servos": [one dict per servo],
        "problems": [...], "warnings": [...]}`. Any problem refuses the start."""

    def connect(self, passive: bool = False) -> np.ndarray:
        """Torque on, holding the present posture. Returns it, (8,) rad. `passive`: the same
        checks, but torque stays OFF and nothing is written (a dry run)."""

    def write(self, q: np.ndarray) -> None:
        """Goal positions, (8,) rad. No reply is awaited."""

    def read(self) -> np.ndarray:
        """Present positions, (8,) rad."""

    def read_full(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(position rad, speed rad/s, load fraction, temperature deg C), each (8,), in ONE
        transaction: a second read for the slow values would nearly fill a 4 ms period."""

    def alarms(self) -> np.ndarray:
        """(8,) uint8: each servo's status byte from its latest reply."""

    def release(self, joints: list[int] | None = None) -> None:
        """Torque off (all joints, or these indices)."""

    def close(self) -> None: ...


class St3215Driver:
    """The real bus. `bus` and `cals` are injectable for tests (the repo's fake serial bus,
    `ElRobotRepo/tests/fake_bus.py`)."""

    def __init__(self, *, port: str | None = None, bus=None, cals=None, calibration=None,
                 qdot_max=None,
                 goal_speed_factor: float = 1.5, acceleration: int = 0,
                 torque_limit: dict[int, int] | None = None, p_coef: dict[int, int] | None = None,
                 d_coef: dict[int, int] | None = None,
                 goal_speed: dict[int, int] | None = None, rewrite_every: int = 1,
                 clock=time.monotonic, sleep=time.sleep):
        from elrobot.calibration import RAD_PER_STEP, load
        from elrobot.st3215 import registers as R
        from elrobot.st3215.bus import Bus

        self.R = R
        self.bus = bus if bus is not None else Bus(port)
        if cals is None:
            cals = load() if calibration is None else load(Path(calibration))
        self.cals = cals
        self.ids = list(SERVO_IDS)
        if sorted(self.cals) != self.ids:
            raise DriverError(f"calibration has joints {sorted(self.cals)}, expected {self.ids}")
        c = [self.cals[j] for j in self.ids]
        self.sign = np.array([x.sign for x in c], dtype=np.float64)
        self.zero = np.array([x.zero_reading for x in c], dtype=np.float64)
        self.lo = np.array([min(x.min_reading, x.max_reading) for x in c])
        self.hi = np.array([max(x.min_reading, x.max_reading) for x in c])
        self.rad_per_step = RAD_PER_STEP
        qdot = np.full(N_JOINTS, 2.4) if qdot_max is None else np.asarray(qdot_max)
        # 3400 steps/s is the register's documented top (above the servo's own no-load speed).
        self.goal_speed = {j: min(3400, max(1, round(goal_speed_factor * v / RAD_PER_STEP)))
                           for j, v in zip(self.ids, qdot)}
        # Explicit caps (steps/s) per servo ID, replacing the ceiling: the servo moves its own
        # setpoint toward each goal at most this fast, so a cap near the commanded speed turns
        # the goal's whole-step staircase into a ramp (executor_test --goal-speed).
        for j, v in (goal_speed or {}).items():
            if not 1 <= int(v) <= 3400:
                raise DriverError(f"goal speed for servo {j}: {v} is not 1..3400 steps/s")
            self.goal_speed[int(j)] = int(v)
        self.acceleration = int(acceleration)
        self.torque_limit = dict(TORQUE_LIMIT if torque_limit is None else torque_limit)
        # P and D per servo ID, for this session only: written with the EEPROM locked, so it lives in
        # RAM until power-off, and `release()` puts the original back (as `elrobot.hold_test`).
        self.gains = {(R.P_COEF, int(j)): int(v) for j, v in (p_coef or {}).items()}
        self.gains.update({(R.D_COEF, int(j)): int(v) for j, v in (d_coef or {}).items()})
        self._gain_original: dict = {}
        self.clock, self.sleep = clock, sleep
        self.torque_on = False
        self._goal_encode = R.GOAL_POSITION.encode
        self._goal_addr, self._goal_size = R.GOAL_POSITION.addr, R.GOAL_POSITION.size
        # 1: every servo's goal goes out every tick. N > 1: a servo is sent its goal only when
        # the reading changes, and everyone is re-sent every Nth write (a lost sync_write has no
        # reply to notice it by). Tests whether re-writing an unchanged goal disturbs the servo.
        self.rewrite_every = int(rewrite_every)
        self._last_goal: np.ndarray | None = None
        self._writes = 0

    # --- conversion ----------------------------------------------------------------------

    def to_rad(self, readings: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(readings, dtype=np.float64) - self.zero) * self.rad_per_step

    def to_readings(self, q: np.ndarray) -> np.ndarray:
        r = np.rint(self.zero + self.sign * np.asarray(q, dtype=np.float64) / self.rad_per_step)
        # A second clamp, in the servo's own units: the calibrated range can never be left even
        # if a caller skipped the executor's.
        return np.clip(r, self.lo, self.hi).astype(np.int64)

    # --- the wire ------------------------------------------------------------------------

    def _read_raw(self) -> np.ndarray:
        got = self.bus.sync_read_reg(self.ids, self.R.PRESENT_POSITION)
        return np.array([got[j] for j in self.ids], dtype=np.int64)

    def read(self) -> np.ndarray:
        return self.to_rad(self._read_raw())

    def write(self, q: np.ndarray) -> None:
        enc = self._goal_encode
        r = self.to_readings(q)
        self._writes += 1
        if self.rewrite_every <= 1 or self._last_goal is None or \
                self._writes % self.rewrite_every == 0:
            send = range(len(self.ids))              # everyone (also the periodic refresh)
        else:
            send = np.flatnonzero(r != self._last_goal)
            if not send.size:
                return
        self.bus.sync_write(self._goal_addr, self._goal_size,
                            {self.ids[i]: enc(int(r[i])) for i in send})
        self._last_goal = r

    def read_full(self):
        R = self.R
        start = R.PRESENT_POSITION.addr          # 56..63: position, speed, load, volts, temp
        raw = self.bus.sync_read(self.ids, start, R.PRESENT_TEMPERATURE.addr + 1 - start)
        blk = [R.decode_block(start, raw[j]) for j in self.ids]
        q = self.to_rad(np.array([b["present_position"] for b in blk]))
        speed = self.sign * np.array([b["present_speed"] for b in blk]) * self.rad_per_step
        # Load's sign follows the servo's own direction, like speed's; flipped onto the URDF's.
        load = self.sign * np.array([b["present_load"] for b in blk]) / 1000.0
        temp = np.array([b["present_temperature"] for b in blk], dtype=np.float64)
        return q, speed, load, temp

    def alarms(self) -> np.ndarray:
        return np.array([self.bus.alarms.get(j, 0) for j in self.ids], dtype=np.uint8)

    # --- torque --------------------------------------------------------------------------

    def check(self) -> dict:
        """Every servo's whole register table, read once with torque off, checked: it answers,
        it is an STS3215 in position mode, its offset is the calibrated one, torque is off, the
        supply voltage and temperature are sane, no alarm bit is set, and it reads inside its
        calibrated range (outside it means a wrong offset or the multi-turn trap)."""
        R = self.R
        from elrobot.st3215.registers import STS3215_MODEL, TABLE_END, status_flags
        servos, problems, warnings = [], [], []
        for i, j in enumerate(self.ids):
            try:
                reg = R.decode_block(0, self.bus.read(j, 0, TABLE_END))
            except BUS_ERRORS as exc:
                problems.append(f"servo {j} does not answer ({exc}): is the 12 V supply on, "
                                f"the cable seated?")
                servos.append({"id": j, "ok": False})
                continue
            raw = int(reg["present_position"])
            volts = reg["present_voltage"] / 10.0
            temp = float(reg["present_temperature"])
            flags = status_flags(int(reg["status"]))
            deg = np.degrees(self.sign[i] * (raw - self.zero[i]) * self.rad_per_step)
            row = {"id": j, "ok": True, "deg": round(float(deg), 1),
                   "volts": volts, "temp": temp, "status": flags,
                   "torque": int(reg["torque_enable"]), "mode": int(reg["mode"]),
                   "p": int(reg["p_coef"]), "d": int(reg["d_coef"]),
                   "torque_limit": int(reg["torque_limit"]),
                   "firmware": f"{reg['firmware_major']}.{reg['firmware_minor']}"}
            servos.append(row)
            bad = []
            if int(reg["model"]) != STS3215_MODEL:
                bad.append(f"model {reg['model']}, not an STS3215 ({STS3215_MODEL})")
            if int(reg["mode"]) != 0:
                bad.append(f"mode {reg['mode']}, not position mode (0)")
            if int(reg["offset"]) != self.cals[j].offset:
                bad.append(f"offset {reg['offset']}, calibration.json says "
                           f"{self.cals[j].offset}: recalibrate")
            if int(reg["torque_enable"]):
                bad.append("torque is already on: another program holds the arm "
                           "(stop it, or `uv run python -m elrobot.relax`)")
            if not VOLTAGE_RANGE[0] <= volts <= VOLTAGE_RANGE[1]:
                bad.append(f"supply {volts:.1f} V, outside {VOLTAGE_RANGE[0]}..{VOLTAGE_RANGE[1]} V")
            if temp >= TEMP_MAX_C:
                bad.append(f"{temp:.0f} C: too hot (let it cool)")
            elif temp >= TEMP_WARN_C:
                warnings.append(f"servo {j}: {temp:.0f} C")
            if flags:
                bad.append(f"alarm: {', '.join(flags)}")
            if not (self.lo[i] - RANGE_MARGIN_STEPS <= raw <= self.hi[i] + RANGE_MARGIN_STEPS):
                bad.append(f"reads {raw}, outside its calibrated range {int(self.lo[i])}.."
                           f"{int(self.hi[i])} (a wrong offset, or the multi-turn trap: "
                           f"power-cycle and check before anything moves)")
            problems += [f"servo {j}: {b}" for b in bad]
        return {"servos": servos, "problems": problems, "warnings": warnings}

    def connect(self, passive: bool = False) -> np.ndarray:
        """`elrobot.move`'s enable sequence: check, hold the present posture, watch, return it.
        `passive`: the checks only; torque stays off and nothing is written (a dry run)."""
        R, bus = self.R, self.bus
        on = [j for j, t in bus.sync_read_reg(self.ids, R.TORQUE_ENABLE).items() if t]
        if on:
            raise DriverError(f"torque is already on for {on}: another program holds the arm. "
                              "Stop it, or run `uv run python -m elrobot.relax`.")
        off = {j: bus.read_reg(j, R.OFFSET) for j in self.ids}
        wrong = [j for j in self.ids if off[j] != self.cals[j].offset]
        if wrong:
            raise DriverError(f"offsets of {wrong} differ from calibration.json; recalibrate first")

        start = self._read_raw()
        if passive:
            return self.to_rad(start)
        bus.sync_write_reg(R.GOAL_POSITION, {j: int(v) for j, v in zip(self.ids, start)})
        bus.sync_write_reg(R.GOAL_SPEED, self.goal_speed)
        bus.sync_write_reg(R.ACCELERATION, {j: self.acceleration for j in self.ids})
        bus.sync_write_reg(R.TORQUE_LIMIT, {j: self.torque_limit[j] for j in self.ids})
        for (reg, j), v in self.gains.items():
            self._gain_original.setdefault((reg, j), bus.read_reg(j, reg))
            bus.write_reg(j, reg, v)
        bus.sync_write_reg(R.TORQUE_ENABLE, {j: 1 for j in self.ids})
        self.torque_on = True
        q0 = self.to_rad(start)
        t0 = self.clock()
        while self.clock() - t0 < ENABLE_WATCH_S:
            moved = np.flatnonzero(np.abs(self.read() - q0) > RUNAWAY_RAD)
            if moved.size:
                try:
                    self.release()
                except DriverError as exc:
                    raise DriverError(f"a joint ran away when torque came on, and then: {exc}. "
                                      "Switch the 12 V supply off.") from exc
                raise DriverError(
                    f"joints {[self.ids[i] for i in moved]} started moving on their own when "
                    "torque came on (the controller disagrees with the reading). Torque is off "
                    "everywhere. Power-cycle before trying again.")
            self.sleep(0.01)
        return q0

    def release(self, joints: list[int] | None = None) -> None:
        """Torque off, CONFIRMED: a sync_write has no reply, and one sent just before the port
        closes can be lost (2026-10-17: "Torque off." printed, the next run found all 8 on). So
        read it back and resend, and raise if a servo still holds."""
        R = self.R
        ids = [self.ids[i] for i in (range(N_JOINTS) if joints is None else joints)]
        still = ids
        for _ in range(3):
            self.bus.sync_write_reg(R.TORQUE_ENABLE, {j: 0 for j in still})
            on = self.bus.sync_read_reg(still, R.TORQUE_ENABLE)
            still = [j for j in still if on[j]]
            if not still:
                break
        if joints is None:
            for (reg, j), v in self._gain_original.items():
                self.bus.write_reg(j, reg, v)
            self._gain_original.clear()
        if still:
            raise DriverError(f"torque is still on for {still} after three tries")
        if joints is None:
            self.torque_on = False

    def close(self) -> None:
        flush = getattr(self.bus.ser, "flush", None)
        if flush is not None:
            flush()                          # anything still queued goes out before the port shuts
        self.bus.close()


class SimDriver:
    """Eight first-order lags with the real servo's resolution and a speed cap.

    `io_s` = (write, read, full read) seconds each call blocks, to give the loop the real bus's
    timing (sync_write ~0.3 ms on the wire; sync_read of 8 positions 1.40 ms, of position +
    speed + load 1.73 ms, measured 2026-10-11). `clock` is the time the lag integrates over;
    with a fake clock, pass `sleep_io=False`.
    """

    STEP = 2 * np.pi / 4096

    def __init__(self, q0=None, *, tau_s: float = 0.03, qdot_max: float = 5.0,
                 io_s=(0.0003, 0.0014, 0.00175), clock=time.monotonic, sleep_io: bool = True):
        self.q = np.zeros(N_JOINTS) if q0 is None else np.asarray(q0, dtype=np.float64).copy()
        self.goal = self.q.copy()
        self.v = np.zeros(N_JOINTS)
        self.tau, self.vmax = float(tau_s), float(qdot_max)
        self.io_s, self.clock, self.sleep_io = io_s, clock, sleep_io
        self.t = clock()
        self.torque_on = False

    def _advance(self):
        t = self.clock()
        dt, self.t = max(0.0, t - self.t), t
        if not self.torque_on or dt == 0:
            self.v[:] = 0
            return
        v = np.clip((self.goal - self.q) / self.tau, -self.vmax, self.vmax)
        self.q += v * dt
        self.v = v

    def _io(self, i):
        if self.sleep_io and self.io_s[i] > 0:
            t_end = time.monotonic() + self.io_s[i]
            while time.monotonic() < t_end:
                pass

    def check(self):
        q = self._position()
        return {"servos": [{"id": i + 1, "ok": True, "deg": round(float(np.degrees(v)), 1),
                            "volts": 12.0, "temp": 30.0, "status": [], "torque": 0, "mode": 0,
                            "p": 32, "d": 32, "torque_limit": 1000, "firmware": "sim"}
                           for i, v in enumerate(q)], "problems": [], "warnings": []}

    def connect(self, passive=False):
        self._advance()
        self.goal = self.q.copy()
        self.torque_on = not passive
        return self.read()

    def write(self, q):
        self._io(0)
        self._advance()
        self.goal = np.asarray(q, dtype=np.float64).copy()

    def _position(self):
        return np.round(self.q / self.STEP) * self.STEP

    def read(self):
        self._io(1)
        self._advance()
        return self._position()

    def read_full(self):
        self._io(2)
        self._advance()
        return self._position(), self.v.copy(), np.zeros(N_JOINTS), np.full(N_JOINTS, 30.0)

    def alarms(self):
        return np.zeros(N_JOINTS, dtype=np.uint8)

    def release(self, joints=None):
        self._advance()
        if joints is None:
            self.torque_on = False

    def close(self):
        pass


def make_driver(kind: str, **kw) -> ServoDriver:
    """By name, so a spawned process can be told which wire to open."""
    if kind == "st3215":
        return St3215Driver(**kw)
    if kind == "sim":
        return SimDriver(**kw)
    raise ValueError(f"unknown driver {kind!r} (st3215, sim)")


__all__ = [
    "BUS_ERRORS",
    "TORQUE_LIMIT",
    "DriverError",
    "ServoDriver",
    "SimDriver",
    "St3215Driver",
    "make_driver",
]
