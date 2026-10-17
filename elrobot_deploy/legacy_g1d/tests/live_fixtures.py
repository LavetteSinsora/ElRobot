"""A synthetic live run: an arena filled the way P1, P2, P5, P6 and P7 fill it,
for the live dashboard's tests (and for looking at the page in a browser
without a robot).

Everything is written through the ring classes' own writers, so the readers
under test see exactly the layout and the counter discipline a live run has.
Times are on `clock.now()`'s axis, ending at `now`, because the modes read
"the last half second" relative to the clock.

What is planted, so a test can ask for it by name (`FakeRun`):

* an arm that moves smoothly on a sinusoid, with a **branch switch** — the left
  elbow commanded 0.15 rad per tick for `SWITCH_TICKS` ticks, ~4.5 rad/s
  against a pi rad/s threshold — at tick `switch_at`;
* poses solved exactly (TCP error 0) except one tick, `off_tick`, whose pose
  is 20 mm off at both flanges;
* a measured arm that lags the command by `LAG_S`, so the servo view has a
  tracking error to report;
* P7 publishing at 500 Hz, holding `k_exec` for `STALL_S` once (a stall);
* chunks every P ticks with their six timing stamps, the newest written `CHUNK_TOTAL_S`
  after its trigger;
* cameras at 30 fps, the second one's source dropping every fifth frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elrobot_deploy.runtime import timing as clock
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.robot.kinematics import ForwardKinematics
from elrobot_deploy.config_loading.rollout_settings import RolloutSettings

SWITCH_TICKS = 3
SWITCH_STEP = 0.15          # rad per tick, left elbow
LAG_S = 0.012               # measured lags commanded by this much
STALL_S = 0.090             # one k_exec hold in the command log
CHUNK_TOTAL_S = 0.060       # trigger -> written
CHUNK_INFER_S = 0.040       # cpu_processed -> chunk_received
CHUNK_OBS_S = -0.004        # anchor - trigger
OFF_M = 0.020               # the one mis-solved tick's pose offset


@dataclass
class FakeRun:
    now: float
    t0: float               # deadline of tick 0
    dt: float
    k_exec: int
    k_last: int             # newest tick written
    switch_at: int
    off_tick: int
    generation: int
    chunks: int


def arm_q(t) -> np.ndarray:
    """The smooth commanded arm, (14,), at time `t` (seconds on the run)."""
    base = np.array([0.2, 0.3, 0.0, 0.9, 0.0, 0.0, 0.0,
                     0.2, -0.3, 0.0, 0.9, 0.0, 0.0, 0.0])
    return base + 0.05 * np.sin(2 * np.pi * 0.5 * t + np.arange(14))


def fake_run(rings, cfg, hw, *, now: float | None = None, n_ticks: int | None = None,
             session: int = schema.ACTIVE, generation: int = 1,
             joint_mode: bool = False, cameras: bool = True) -> FakeRun:
    """Fill `rings` with a run that is `n_ticks` action ticks old at `now`."""
    now = clock.now() if now is None else now
    control = rings["control"]
    launch = RolloutSettings.of(cfg).to_record()
    control.request_config(launch)
    control.activate(launch, "pick up the cube")
    control.rollout_generation = generation
    control.session = session

    dt = cfg.dt
    P = cfg.P
    n_slots = rings["targets"].n
    n_ticks = n_ticks or max(n_slots - 2, 12)
    # Tick k's deadline; the executing tick is the one due next, with P ticks
    # written ahead of it, as P6 runs.
    t0 = now - (n_ticks - P) * dt
    k_exec = n_ticks - P
    k_last = n_ticks
    switch_at = max(2, n_ticks // 2)
    off_tick = switch_at + SWITCH_TICKS + 2

    # --- targets (P6) ---------------------------------------------------------
    fk = ForwardKinematics(hw.urdf)
    waist_ik = np.array([0.0, 0.039, 0.0])
    q_line = {}
    extra = 0.0
    for k in range(1, k_last + 1):
        q14 = arm_q(k * dt)
        if switch_at <= k < switch_at + SWITCH_TICKS:
            extra += SWITCH_STEP
        q14 = q14.copy()
        q14[3] += extra
        q_line[k] = q14
        body = np.concatenate([q14, np.zeros(schema.WAIST_DOF)])
        if joint_mode:
            rings["targets"].write(k, t0 + k * dt, body,
                                   np.zeros((schema.N_HANDS, schema.HAND_DIM)),
                                   intervene=0)
            continue
        pose = fk.flange(q14, waist_ik).copy()
        err = 0.0005
        if k == off_tick:
            pose[:, 0] += OFF_M
            err = OFF_M
        rings["targets"].write(k, t0 + k * dt, body,
                               np.zeros((schema.N_HANDS, schema.HAND_DIM)),
                               intervene=0, err=err, pose=pose,
                               waist_ik=waist_ik)
    control.k_exec = k_exec

    def commanded(t):
        """The linear walk between tick deadlines, (14,)."""
        x = (t - t0) / dt
        k = int(np.clip(np.floor(x), 1, k_last - 1))
        u = float(np.clip(x - k, 0.0, 1.0))
        return (1 - u) * q_line[k] + u * q_line[k + 1]

    # --- state (P1) ------------------------------------------------------------
    arm = list(hw.joints_arm)
    waist = list(hw.joints_waist)
    span = min(cfg.state_history_s, 2.0)
    for t in np.arange(now - span, now, 1e-3):
        q29 = np.zeros(schema.N_JOINTS)
        dq29 = np.zeros(schema.N_JOINTS)
        q29[arm] = commanded(t - LAG_S)
        dq29[arm] = (commanded(t - LAG_S + 1e-3) - commanded(t - LAG_S)) / 1e-3
        q29[waist] = waist_ik
        rings["state"].write(t_recv=float(t), q=q29, dq=dq29,
                             flange=np.tile([0.3, 0.0, 0.2, 1.0, 0, 0, 0],
                                            (schema.N_HANDS, 1)),
                             grip_meas=np.zeros(schema.GRIP_DIM),
                             t_grip_recv=float(t))

    # --- The command log (P7) --------------------------------------------------
    n_cmd = rings["command"].n
    t_cmd = now - (n_cmd - 1) * 0.002 + np.arange(n_cmd) * 0.002
    stall_from = t_cmd[len(t_cmd) // 3]
    for t in t_cmd:
        k_now = int(np.floor((t - t0) / dt)) + 1
        if stall_from <= t < stall_from + STALL_S:
            k_now = int(np.floor((stall_from - t0) / dt)) + 1
        q = np.concatenate([commanded(t), np.zeros(schema.WAIST_DOF)])
        dq = np.concatenate([(commanded(t + 1e-3) - commanded(t)) / 1e-3,
                             np.zeros(schema.WAIST_DOF)])
        tau_ff = 0.1 * dq
        rings["command"].write(t=float(t), k_exec=k_now, q=q, dq=dq,
                               tau=tau_ff, tau_ff=tau_ff, q_lin=q)

    # --- chunks (P5) -----------------------------------------------------------
    n_chunks = 0
    chunk_id = 0
    for kf in range(1, k_last + 1, P):
        chunk_id += 1
        n_chunks += 1
        n = min(P, cfg.H)
        trig = t0 + kf * dt - CHUNK_TOTAL_S - 2 * dt
        kw = dict(t_trigger=trig, t_anchor=trig + CHUNK_OBS_S,
                  t_obs_received=trig + 0.002, t_cpu_processed=trig + 0.008,
                  t_chunk_received=trig + 0.008 + CHUNK_INFER_S,
                  t_chunk_written=trig + CHUNK_TOTAL_S)
        hand = np.zeros((n, schema.N_HANDS, schema.HAND_DIM))
        if joint_mode:
            rings["chunks"].write(chunk_id=chunk_id, k_first_action=kf,
                                  n_actions=n, hand=hand,
                                  arm_q=np.zeros((n, schema.N_HANDS, 7)),
                                  waist=np.zeros((n, schema.WAIST_DOF)), **kw)
        else:
            rings["chunks"].write(chunk_id=chunk_id, k_first_action=kf,
                                  n_actions=n, hand=hand,
                                  pose=np.zeros((n, schema.N_HANDS,
                                                 schema.POSE_DIM)), **kw)
    control.newest_chunk_id = chunk_id

    # --- cameras (P2) ----------------------------------------------------------
    jpeg = bytes([0xFF, 0xD8, 0xFF, 0xD9])
    views = sorted(rings["cameras"].items()) if cameras else []
    for c, (name, ring) in enumerate(views):
        hz = cfg.views[name].hz
        src = 0
        for t in np.arange(now - 1.0, now, 1.0 / hz):
            src += 2 if (c == 1 and src % 5 == 4) else 1
            ring.spare()
            ring.publish(float(t), jpeg, seq=src)

    return FakeRun(now=now, t0=t0, dt=dt, k_exec=k_exec, k_last=k_last,
                   switch_at=switch_at, off_tick=off_tick,
                   generation=generation, chunks=n_chunks)


__all__ = ["FakeRun", "LAG_S", "OFF_M", "STALL_S", "SWITCH_STEP",
           "SWITCH_TICKS", "arm_q", "fake_run"]
