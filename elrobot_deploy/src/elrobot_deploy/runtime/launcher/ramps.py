"""Homing, reset and exit ramps: cosine joint-space moves P0 writes into the target ring, at
ticks the bus process executes like any other. The poses are the profile's `[poses]`."""

from __future__ import annotations

import numpy as np

from elrobot_deploy.runtime import timing as _clock


def ramp_duration(q_from, q_to, *, qdot_max, min_s: float, margin: float) -> float:
    """`max(min_s, max_j |dq_j| / (qdot_max_j * margin))`: derived, never configured.

    The slowest joint sets the duration, so the ramp is inside every joint's velocity limit by
    construction and the executor's slew clamp never has to act during it.
    """
    d = np.abs(np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float))
    reach = float(np.max(d / (np.asarray(qdot_max, dtype=float) * margin)))
    return max(float(min_s), reach)


def cosine_samples(a_from, a_to, n: int) -> np.ndarray:
    """`n` samples of a cosine ease-in-out, `(n,) + a_from.shape`. The first sample is the
    first that moves (`a_from` is where the arm already is); the last is exactly `a_to`."""
    a_from = np.asarray(a_from, dtype=float)
    a_to = np.asarray(a_to, dtype=float)
    u = (np.arange(1, n + 1, dtype=float) / n).reshape((n,) + (1,) * a_from.ndim)
    out = a_from + (a_to - a_from) * (0.5 * (1.0 - np.cos(np.pi * u)))
    out[-1] = a_to
    return out


def cosine_ramp(q_from, q_to, *, dt: float, qdot_max, min_s: float, margin: float) -> np.ndarray:
    """`(n, 8)` samples at the target rate, row `i` being tick `k_first + i`.

    A cosine, so velocity is continuous at both ends. Its peak speed is `pi/2` times its mean,
    which is why the margin must stay at or below `2/pi` (`config.load` refuses more).
    """
    duration = ramp_duration(q_from, q_to, qdot_max=qdot_max, min_s=min_s, margin=margin)
    return cosine_samples(q_from, q_to, max(1, int(np.ceil(duration / dt))))


def write_ramp(rings, k_first: int, qs, *, t_first: float, dt: float, lead_ticks: int,
               clock=_clock) -> list[tuple[int, float]]:
    """Write `qs` as ticks `k_first, k_first + 1, ...` due `t_first + i * dt`. Returns the
    `(k, deadline)` written.

    Two disciplines, both P6's too:

    * **Pacing.** A ramp may be longer than the ring, so each tick is written only once it is
      due within `lead_ticks` ticks: the ring's capacity, in time.
    * **The gate.** Never write into `{k_exec - 1, k_exec}`: the bus process may be mid-segment
      on them.

    Every row is stamped with the LIVE intervention counter, re-read per tick: after a rollout
    in which somebody drove, the counter is even but not zero, and a row stamped 0 would never
    be executed.
    """
    targets, control = rings["targets"], rings["control"]
    written: list[tuple[int, float]] = []
    for i, q in enumerate(qs):
        k = int(k_first) + i
        deadline = t_first + i * dt
        clock.sleep_until(deadline - lead_ticks * dt)
        k_exec = control.k_exec
        if k in (k_exec, k_exec - 1):
            continue
        targets.write(k, deadline, q, intervene=control.intervene)
        written.append((k, deadline))
    return written


__all__ = ["cosine_ramp", "cosine_samples", "ramp_duration", "write_ramp"]
