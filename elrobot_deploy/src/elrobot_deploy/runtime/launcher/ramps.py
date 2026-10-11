"""Homing and exit ramps: cosine joint-space moves written by P0 into the
target ring at ticks P7 and P8 execute like any other. The poses come from the
deploy file (`docs/deploying_a_policy/deploy_toml.md`, "`[home]` and `[exit]`:
where the arm starts and ends")."""

from __future__ import annotations

import numpy as np

from elrobot_deploy.runtime import timing as _clock


def ramp_duration(q_from, q_to, *, qdot_max, min_s: float,
                  margin: float) -> float:
    """`max(min_s, ‖Δq‖∞ / (qdot_max · margin))` — derived, never configured.

    The ramp is inside the velocity limit *by construction*, so a slew clamp
    during homing is a genuine bug rather than an expected clip. `qdot_max` is
    per joint, so the infinity norm is taken over `Δq_j / qdot_max_j` rather
    than over `Δq` alone: the slowest joint sets the duration, and a wrist that
    may move twice as fast does not get to hide a waist that may not.
    """
    d = np.abs(np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float))
    reach = float(np.max(d / (np.asarray(qdot_max, dtype=float) * margin)))
    return max(float(min_s), reach)


def cosine_samples(a_from, a_to, n: int) -> np.ndarray:
    """`n` samples of a cosine ease-in-out, `(n,) + a_from.shape`.

    The SHAPE of a ramp, with no opinion about its duration — the caller
    supplies `n`. Split out from `cosine_ramp` so the end effector can ride the
    arm's tick grid: `qdot_max` is 29 joint velocity limits and there is no
    equivalent for a gripper (P8 has no velocity clamp), so the hand cannot derive a duration of its
    own. It does not need to. Both payloads of a tick share one `k` and one
    deadline, so they must share one duration, and the arm's is the one
    that is derived from a real limit.

    `a_from` itself is not a row. It is where the robot already is — P7 and P8
    are holding it — so the first sample is the first that moves.
    """
    a_from = np.asarray(a_from, dtype=float)
    a_to = np.asarray(a_to, dtype=float)
    # One trailing axis per dimension of the payload, so this broadcasts over a
    # (17,) body vector and a (2, 6) hand command alike.
    u = (np.arange(1, n + 1, dtype=float) / n).reshape((n,) + (1,) * a_from.ndim)
    out = a_from + (a_to - a_from) * (0.5 * (1.0 - np.cos(np.pi * u)))
    # Assign the endpoint rather than trusting `a + (b - a) * 1.0` to round back
    # onto `b`: "homing ends at home_q" is an exact claim.
    out[-1] = a_to
    return out


def cosine_ramp(q_from, q_to, *, dt: float, qdot_max, min_s: float,
                margin: float) -> np.ndarray:
    """`(n, 17)` samples at the action rate, row `i` being tick `k_first + i`.

    A cosine ease-in-out, so velocity is continuous at both ends: the robot is
    stationary before the ramp and must be stationary at `home_q`, and a linear
    ramp would step the commanded velocity at both boundaries.

    The profile's peak speed is `π/2` times its mean, which is why
    `home_velocity_margin` defaults to 0.5: at any margin above `2/π` the peak
    of the cosine exceeds `qdot_max` even though the mean does not.
    """
    duration = ramp_duration(q_from, q_to, qdot_max=qdot_max, min_s=min_s,
                             margin=margin)
    return cosine_samples(q_from, q_to, max(1, int(np.ceil(duration / dt))))


def home_ticks(n: int) -> np.ndarray:
    """`[1 - n … 0]` — homing occupies ticks `k < 0` and ends at `k = 0`.

    Tick 0 *is* the end of homing, which is why P6's first epoch stamp
    (`deadline(0) = now`) is true rather than approximately true.
    """
    return np.arange(1 - n, 1, dtype=np.int64)


def write_ramp(rings, ks, qs, hands, *, t_first: float, dt: float,
               lead_ticks: int, clock=_clock,
               cursor_live: bool = False) -> list[tuple[int, float]]:
    """Write a ramp into both target rings. Returns the `(k, deadline)` written.

    `qs` is `(n, 17)` and `hands` is `(n, 2, 6)` — a ramp per payload, sampled
    on the same tick grid. `hands` must be a ramp too: ONE held command
    broadcast over every tick would be a step dressed as a ramp — P8 would take
    the final aperture in its first segment and drive there in one action tick,
    at whatever speed the jaw can manage, while the arm eases over seconds.

    Deadlines are `t_first + i·dt`, computed by P0 directly: during homing no
    epoch exists yet, and during EXITING P6 is gone.

    Two disciplines, both P6's:

    * **Pacing.** The ring holds `N_body` ticks, and a ramp may be longer than
      that, so P0 may not write it all at once. It stays `lead_ticks` ahead in
      *time* — the ring's own capacity — rather than counting on `k_exec`,
      which means the pacing is correct before P7 has taken its first target.
    * **The gate.** Never write into `{k_exec − 1, k_exec}`: the value must not
      perturb the segment P7 is mid-interpolation on. With a lead of the whole
      ring this can only fire if P7 has run far ahead of its own deadlines, so
      it is in practice an assertion — but it is the same assertion P6 makes,
      and homing is meant to be an ordinary producer.

    `k_exec` is consulted only once it is *live* — once it has moved off the
    value it held before this ramp existed. During HOME it reads 0 while the
    ramp is writing negative ticks, because P7 has not taken a target yet, and
    a gate that believed that stale zero would refuse to write tick 0 — the one
    tick homing must end on. The exit ramp passes `cursor_live` because it is
    anchored on `k_exec + 2` and so cannot be started against a stale one.
    """
    ks = np.asarray(ks, dtype=np.int64)
    if not (len(ks) == len(qs) == len(hands)):
        # Body and hand are written together per tick, so a length mismatch is
        # not a short ramp — it is a run where one payload stops being written
        # partway and P8 holds an aperture nobody chose.
        raise ValueError(f"ramp lengths disagree: {len(ks)} ticks, {len(qs)} "
                         f"body samples, {len(hands)} hand samples")

    targets, control = rings["targets"], rings["control"]
    stale = control.k_exec
    written: list[tuple[int, float]] = []
    for i, (k, q, cmd) in enumerate(zip(ks, qs, hands)):
        k = int(k)
        deadline = t_first + i * dt
        clock.sleep_until(deadline - lead_ticks * dt)
        k_exec = control.k_exec
        if (cursor_live or k_exec != stale) and k in (k_exec, k_exec - 1):
            continue
        # One row: the ramp's joints and its end-effector command cannot be
        # written apart (`schema.TARGET`).
        #
        # Stamped with the LIVE counter, not with a zero. P0's ramps
        # run outside any intervention, but "outside an intervention" is an EVEN
        # counter, not necessarily a zero one: after a rollout in which somebody
        # took over even once, it is 2, or 4. A ramp stamped 0 would be filtered
        # out of P7's segment advance, and the exit ramp — the one that brings
        # the arm down at the end of the run — would never execute at all.
        # Re-read per tick rather than hoisted: these loops sleep between ticks,
        # and P11 is still running.
        targets.write(k, deadline, q, cmd, intervene=control.intervene)
        written.append((k, deadline))
    return written


__all__ = ["cosine_ramp", "cosine_samples", "home_ticks", "ramp_duration",
           "write_ramp"]
