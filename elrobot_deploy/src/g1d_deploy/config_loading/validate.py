"""Startup consistency checks on the resolved settings: refuse, before anything
moves, a configuration that cannot mean what it says.

`validate` runs once at launch (P0, `check-policy`); `validate_rollout` again
at every Start against the operator's requested settings. What each refusal
protects is in its own message; the user-facing rules are in
`docs/deploying_a_policy/deploy_toml.md`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from g1d_deploy.config_loading.derive import P_MIN
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading.resolved import (CAMERA_SOURCES, DeployConfig, Hardware,
                                                missing_camera_driver, not_a_view_hint)
from g1d_deploy.runtime import cpu_pinning as affinity, ik as ik_solvers
from g1d_deploy.runtime.shared_memory import schema

#: The largest arm gain scale `validate` accepts. kp 80 -> 200 on the shoulders;
#: arXiv:2606.15915 runs the G1 arm at 400, so this is inside what the actuator
#: has been shown to take, and far enough below it to stay a tuning experiment.
ARM_GAIN_SCALE_MAX = 2.5


def check_servo(cfg: DeployConfig, hw: Hardware) -> None:
    """Refuse a `ServoShaping` that cannot mean what it says."""
    sv = cfg.servo
    for name in ("filter_hz", "velocity_ff", "torque_ff", "arm_gain_scale"):
        v = float(getattr(sv, name))
        if not np.isfinite(v) or v < 0:
            raise ConfigError(f"servo.{name} = {v}, must be finite and >= 0")
    if sv.filter_hz and not 1.0 <= sv.filter_hz <= 50.0:
        raise ConfigError(f"servo.filter_hz = {sv.filter_hz}: outside [1, 50] Hz")
    if sv.velocity_ff > 1.5:
        raise ConfigError(f"servo.velocity_ff = {sv.velocity_ff}: 1.5 is the ceiling")
    if sv.torque_ff > 1.5:
        raise ConfigError(f"servo.torque_ff = {sv.torque_ff}: 1.5 is the ceiling")
    if not 0.0 < sv.arm_gain_scale <= ARM_GAIN_SCALE_MAX:
        raise ConfigError(f"servo.arm_gain_scale = {sv.arm_gain_scale}: must be in "
                          f"(0, {ARM_GAIN_SCALE_MAX}]")
    if sv.torque_ff and hw.servo_model is None and not cfg.servo_feedforward:
        raise ConfigError(
            f"servo.torque_ff = {sv.torque_ff} needs this robot's [servo_model] "
            f"in the robot file. Add it, or run with --torque-ff 0.")


def intervene_ik_refusal(ik: str) -> str:
    """The message for `--intervene` with any IK but `arm_qp`. One string, so
    the CLI's exit and the launcher's `ConfigError` cannot say different things."""
    return (f"--intervene with ik = {ik!r} is currently unsupported. The "
            f"intervention has only been built and tested against the per-tick "
            f"arm QP; run with ik = \"{ik_solvers.INTERVENE}\" (drop --ik, or "
            f"pass --ik {ik_solvers.INTERVENE}).")


def check_intervene_ik(cfg: DeployConfig, *, intervene: bool) -> None:
    """Refuse an intervention on anything but `arm_qp`.

    Refused, not merged. The whole-body path plans a chunk's trunk once, when
    the chunk lands, and commands that plan tick by tick; the intervention
    swaps chunk rows for operator rows mid-chunk and preempts whatever was
    latched. Nobody has decided what a planned trunk should do when the chunk
    it was planned for is abandoned under it, and an unspecified waist motion
    under a human's hands is not a thing to find out about on the robot.

    The arm PLANNER is refused for the same kind of reason, although it
    commands no trunk: it keeps a per-chunk branch plan and an escape that
    pulls the arm toward a planned posture when the executed arm misses, and
    under an operator's targets that plan belongs to an abandoned chunk. The
    per-tick QP answers each target from the last answer and nothing else,
    which is what the takeover was built and tested against. `__main__` makes
    `arm_qp` the default under `--intervene` (as it makes the servo linear),
    so this bites only on an explicit other choice.

    Both paths call this: `__main__` before anything is loaded, so the refusal
    is the first line printed, and `Launcher` so a programmatic launch cannot
    step round it.
    """
    if intervene and cfg.ik != ik_solvers.INTERVENE:
        raise ConfigError(intervene_ik_refusal(cfg.ik))


def validate_rollout(cfg: DeployConfig, hw: Hardware) -> None:
    """The refusals that depend on a SWITCHABLE setting
    (`rollout_settings.RolloutSettings`).

    `validate` runs these at launch; P0 runs them again on every START against
    the operator's requested settings (`rollout_settings.check`), and P10 before it
    stores a request, so a dashboard gets its answer at once. Pure: no file,
    no machine, no import that costs anything for a shipped IK solver.
    """
    if not (np.isfinite(float(cfg.control_hz)) and float(cfg.control_hz) > 0):
        raise ConfigError(f"control_hz = {cfg.control_hz}, must be finite and > 0")
    if cfg.P < P_MIN:
        raise ConfigError(
            f"replan_period_ticks = {cfg.P}, must be >= 2. At P = 1 the window "
            f"P5 scans is empty and t_next_inference runs backwards.")
    if cfg.H < cfg.P:
        raise ConfigError(
            f"action_horizon = {cfg.H} < replan_period_ticks = {cfg.P}: the "
            f"chunk would run out before the next replan.")
    if cfg.sample_steps < 1:
        raise ConfigError(
            f"sample_steps = {cfg.sample_steps}, must be >= 1. It is the number of "
            f"integration steps the policy's sampler takes per chunk.")
    if cfg.plan_lookahead_ticks < 0:
        raise ConfigError(
            f"plan_lookahead_ticks = {cfg.plan_lookahead_ticks}, must be >= 0. "
            f"It EXTENDS what the trunk planner sees past the replan period; "
            f"there is no way to show it less than the chunk it must solve.")
    if cfg.plan_lookahead_ticks:
        if not cfg.sync:
            # Refused rather than ignored, and for the reason a waist solver
            # is in `_check_ik`: the lookahead rows extend ONE chunk's trajectory, and in
            # async two chunks overlap by design, so the rows appended past
            # tick `kf + n_actions` are a region another live chunk is already
            # contributing to. The planner would optimize against a future the
            # blend is about to overwrite.
            raise ConfigError(
                f"plan_lookahead_ticks = {cfg.plan_lookahead_ticks} with "
                f"sync = false. The lookahead extends a single chunk past its "
                f"own actions, which only means something when that chunk owns "
                f"the ticks after them. Run sync, or leave it 0.")
        if cfg.P + cfg.plan_lookahead_ticks > cfg.H:
            raise ConfigError(
                f"replan_period_ticks = {cfg.P} + plan_lookahead_ticks = "
                f"{cfg.plan_lookahead_ticks} exceeds action_horizon = {cfg.H}. "
                f"The lookahead rows are decoded actions the chunk will not "
                f"run, so they can only come from the horizon the policy "
                f"already produces — there is nothing past {cfg.H} to publish.")
    check_servo(cfg, hw)
    _check_ik(cfg)


def check_async_timing(cfg: DeployConfig, latency_s: float) -> None:
    """Can an async rollout keep up with an inference of `latency_s`?

    Async runs one inference every `P` ticks: triggered at tick `k`'s deadline,
    its chunk covers ticks `k+1 .. k+H` and P6 adopts it at the tick being
    executed when it lands, plus two. Two things must hold, and neither is
    visible in the config alone:

    * **`P·dt > L`.** Otherwise every trigger is already in the past when it is
      scheduled, each chunk lands later than the one before, and soon covers
      only ticks whose deadlines have passed. P7 then has nothing to execute:
      the arm holds, forever, and nothing else says why (measured in
      simulation: P=5, H=50, L=200 ms froze at tick 260).
    * **`H - P >= ceil(L/dt) + 2`.** The previous chunk must still cover the
      ticks that run while this one is being inferred, plus P6's two ticks of
      margin; otherwise every seam leaves a hole and skips actions.

    `latency_s` is P5's `Control.latency_s`; 0 (not measured) checks nothing.
    Sync has neither problem: the arm waits for each chunk.
    """
    if cfg.sync or latency_s <= 0:
        return
    dt = cfg.dt
    span = math.ceil(latency_s / dt - 1e-9)          # ticks one inference spans
    p_min = math.floor(latency_s / dt) + 1           # smallest P with P·dt > L
    p_max = cfg.H - span - 2                         # largest P leaving cover
    ok_range = (f"replan_period {p_min}..{p_max}" if p_min <= p_max else
                f"no replan period at action_horizon = {cfg.H}; run sync, "
                f"or use a faster policy")
    measured = (f"inference measured at {latency_s * 1e3:.0f} ms "
                f"({latency_s / dt:.1f} ticks at {cfg.control_hz:g} Hz)")
    if cfg.P * dt <= latency_s:
        raise ConfigError(
            f"async with replan_period = {cfg.P} ({cfg.P * dt * 1e3:.0f} ms): "
            f"{measured}, longer than a replan period, so each chunk would land "
            f"later than the last until none can run and the arm stops. "
            f"Works: {ok_range}.")
    if cfg.H - cfg.P < span + 2:
        raise ConfigError(
            f"async with replan_period = {cfg.P}, action_horizon = {cfg.H}: "
            f"{measured}; a chunk must still cover the {span + 2} ticks after "
            f"the next trigger, but only {cfg.H - cfg.P} remain, so every seam "
            f"would skip actions. Works: {ok_range}.")


def _check_servo_modules(cfg: DeployConfig) -> None:
    """`[control.servo] smoother` / `feedforward` import in THIS interpreter —
    P7 runs in it, not in the policy's."""
    from g1d_deploy.runtime.execution import servo_filter
    for key, spec in (("smoother", cfg.servo_smoother),
                      ("feedforward", cfg.servo_feedforward)):
        if spec:
            try:
                servo_filter.load(spec)
            except ValueError as exc:
                raise ConfigError(
                    f"[control.servo] {key}: {exc}. It runs in P7, under "
                    f"g1d_deploy's own Python, so its module must import "
                    f"there.") from exc


def _check_blend(cfg: DeployConfig) -> None:
    """`blend` resolves: a shipped name, or a `module:function` that imports
    in THIS interpreter — P6 runs in it, not in the policy's."""
    from g1d_deploy.runtime.execution import chunk_blend
    m = float(cfg.temporal_ensemble_m)
    if not (np.isfinite(m) and m >= 0):
        raise ConfigError(f"[control] temporal_ensemble_m = {m}: must be finite "
                          f"and >= 0 (0 = a plain mean)")
    try:
        chunk_blend.get(cfg.blend, cfg)
    except ValueError as exc:
        raise ConfigError(
            f"[control] blend = {cfg.blend!r}: {exc}. A custom blend runs in "
            f"P6, under g1d_deploy's own Python, so its module must import "
            f"there.") from exc


def _check_ik(cfg: DeployConfig) -> None:
    """`validate_rollout`'s IK half: the solver exists, and it can run here.

    Pure for the four shipped names — no import that costs anything. A
    `module:factory` is imported to read its mode; that is the research path
    and it pays for itself.
    """
    try:
        mode = ik_solvers.mode_of(cfg.ik)
    except ValueError as exc:
        raise ConfigError(
            f"ik = {cfg.ik!r}: {exc}. It names the IK solver P6 runs; an "
            f"unknown name would otherwise surface inside P6 after the arena "
            f"is mapped.") from exc
    if mode == "waist" and not cfg.sync:
        # Refused rather than ignored. `p6_target` falls back to the per-tick
        # path in async — the plan assumes sync's one contributor per tick — and
        # a run that asked for the trunk and quietly did not get it is the worst
        # of the three outcomes: the record names a waist solver, the trunk
        # never moves, and nothing anywhere says why.
        raise ConfigError(
            f"ik = {cfg.ik!r} (a waist-mode solver) with sync = false. Planning "
            f"a chunk's trunk needs all P targets to belong to one epoch, which "
            f"is a sync-only guarantee; in async two chunks overlap by "
            f"design and the trunk would be planned against a blend that "
            f"changes underneath it. Run sync, or use an arm solver "
            f"(ik = \"arm_planner\").")


def validate(cfg: DeployConfig, hw: Hardware) -> None:
    """Every startup refusal about the configuration, in one place."""
    validate_rollout(cfg, hw)
    _check_blend(cfg)
    _check_servo_modules(cfg)
    for key, value in cfg.labels.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ConfigError(f"labels must map str -> str, got {key!r}: {value!r}")
    # Every policy-facing camera name is a VIEW name (`DeployConfig.views`):
    # a side_by_side camera exists only as its two halves. Built first, because
    # building it is also what refuses a split that collides with a declared
    # camera's name.
    views = cfg.views
    if cfg.primary_camera not in views:
        raise ConfigError(
            f"primary_camera = {cfg.primary_camera!r} is not one of the cameras "
            f"{sorted(views)}. It anchors every other stream."
            + not_a_view_hint(cfg.primary_camera, cfg.cameras))
    for name, seconds in cfg.camera_history_s.items():
        if name not in views:
            raise ConfigError(
                f"camera_history_s names {name!r}, which is not a camera "
                f"({sorted(views)})." + not_a_view_hint(name, cfg.cameras))
        if not (np.isfinite(seconds) and seconds >= 0):
            raise ConfigError(f"camera_history_s[{name!r}] = {seconds!r}; want "
                              f"finite, non-negative seconds")
    # `observed_cameras` names a SUBSET of what exists. A stranger here is the
    # mistake worth catching early: it reads as "this camera feeds the policy"
    # and would instead be a KeyError deep inside P5's first anchor, after the
    # arena is mapped and the policy has warmed up.
    strangers = set(cfg.observed) - set(views)
    if strangers:
        raise ConfigError(
            f"observed_cameras names {sorted(strangers)}, which are not "
            f"declared cameras ({sorted(views)}). It selects which of "
            f"`cameras` the policy sees; it cannot introduce one."
            + "".join(not_a_view_hint(n, cfg.cameras) for n in sorted(strangers)))
    # The primary defines `t_anchor` itself, so a primary the policy does not
    # observe is an instant taken off an image no model reads — and, since the
    # anchor set would then not contain it at all, a KeyError rather than a
    # wrong answer. Both are worth refusing here by name.
    if cfg.primary_camera not in cfg.observed:
        raise ConfigError(
            f"primary_camera = {cfg.primary_camera!r} is not in "
            f"observed_cameras {sorted(cfg.observed)}. The primary's chosen "
            f"frame IS t_anchor, so it cannot be a camera that is "
            f"only recorded.")
    for name, spec in cfg.cameras.items():
        if spec.source not in CAMERA_SOURCES:
            raise ConfigError(
                f"camera {name!r} has source {spec.source!r}, which is not one "
                f"of {list(CAMERA_SOURCES)}")
        # Only the ZMQ source speaks to the robot board, so only it needs a port
        # there. A camera on the deploy PC's own USB has no port at all, and
        # demanding a placeholder one would be demanding a number that means
        # nothing and can be set wrong.
        if spec.source == "zmq" and name not in hw.camera_ports:
            raise ConfigError(
                f"camera {name!r} has source 'zmq' but no port in "
                f"the robot file ([cameras.{name}] port)")
        missing = missing_camera_driver(name, spec.source)
        if missing:
            raise ConfigError(missing)
        if spec.source == "v4l2" and not spec.device:
            raise ConfigError(
                f"camera {name!r} has source 'v4l2' and no device. It must name "
                f"a /dev/v4l/by-path/... node: the fisheye units report "
                f"duplicate serials so by-id collides, and /dev/videoN is not "
                f"stable across reboots. `ls -l /dev/v4l/by-path/` on the "
                f"deploy PC lists them.")
    for label, arr, n in (("home_q", cfg.home_q, schema.ARM_DOF),
                          ("exit_q", cfg.exit_q, schema.ARM_DOF),
                          ("home_waist", cfg.home_waist, schema.WAIST_DOF),
                          ("exit_waist", cfg.exit_waist, schema.WAIST_DOF)):
        if np.shape(arr) != (n,):
            raise ConfigError(f"{label} must have shape ({n},), got {np.shape(arr)}")
    for label, arr in (("home_hand", cfg.home_hand), ("exit_hand", cfg.exit_hand)):
        if np.shape(arr) != (schema.N_HANDS, schema.HAND_DIM):
            raise ConfigError(
                f"{label} must have shape "
                f"({schema.N_HANDS}, {schema.HAND_DIM}), got {np.shape(arr)}")
    if len(hw.joints_arm) != schema.ARM_DOF:
        raise ConfigError(f"robot file joints.arm must list {schema.ARM_DOF} indices")
    if len(hw.joints_waist) != schema.WAIST_DOF:
        raise ConfigError(f"robot file joints.waist must list {schema.WAIST_DOF}")
    # The CPU core reservation, checked rather than described
    # (`docs/setup/machine_toml.md`, "CPU pinning").
    affinity.validate_pinning(hw, error=ConfigError)
    # Last, because it is the only check that touches the filesystem. An empty
    # checkpoint is legal — that is the EchoPolicy dry run. A non-empty one that
    # does not exist is refused HERE, before the arena is mapped and ten
    # processes spawn, rather than inside P5 after WARMUP has begun. `~` is
    # expanded first: this is a hand-written path.
    if cfg.checkpoint and not Path(os.path.expanduser(cfg.checkpoint)).exists():
        raise ConfigError(
            f"checkpoint {cfg.checkpoint!r} does not exist. Point [checkpoint] "
            f"at what the policy loads (a step directory or a file), or leave "
            f"it out for a policy that loads nothing.")
    # P5 is spawned with it: a typo must cost a second here, not surface as a
    # P5 that never warms up.
    if cfg.policy_python and not os.access(cfg.policy_python, os.X_OK):
        raise ConfigError(
            f"[policy] python = {cfg.policy_python!r} is not an executable "
            f"interpreter (docs/deploying_a_policy/policy_contract.md, "
            f"\"The adapter's own Python\").")


__all__ = ["ARM_GAIN_SCALE_MAX", "check_intervene_ik", "check_servo",
           "intervene_ik_refusal", "validate", "validate_rollout"]
