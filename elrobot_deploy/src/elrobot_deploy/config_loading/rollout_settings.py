"""Rollout settings — what may change between rollouts of one launch.

Design: `docs/deploying_a_policy/deploy_toml.md`, "Per launch or per rollout",
and `docs/deploying_a_policy/running_and_live_dashboard.md`, "Changing things
between rollouts".

`RolloutSettings` is the Python face of `schema.SETTINGS`, the record the
control region holds twice:

    requested_config   P10 (the dashboard) writes it, any time
    active_config      ONLY P0 writes it, by copying the request at the START
                       it accepts, after `check` — then bumps
                       `rollout_generation`

and every process that cares **latches `active_config` when a rollout starts
and holds it until the rollout ends** (`latched`). Nothing but P0 reads the
request, so a change can never land mid-rollout, and every process agrees on
the rollout's settings by construction.

The settings are exactly the ones whose change needs no new arena and no
reload of the policy: sync/async, replan period `P`, control rate, IK solver, servo shaping
the policy's sample steps and the async blend. The IK solver switches among
the four shipped names (`schema.IK_OPTIONS`); a research `module:factory`
named at launch is carried as `schema.IK_LAUNCH` ("this launch's own") and
cannot be switched TO. The blend works the same way (`schema.BLEND_OPTIONS`,
`schema.BLEND_LAUNCH`). Robot, policy, checkpoint, `H`, history and the camera set are fixed per
launch — they size the arena (`derive.derive` sizes
the rings for the worst `P` precisely so `P` can be in this list) or load the
policy.

Where each field lands, per process:

    P0   `dt` for its own two-tick waits; validation (`check`)
    P5   `P`, `sync`, `dt` for the schedule (a new `PolicyLoop` per rollout);
         `sample_steps` through `Policy.on_settings`; the prompt through
         `Policy.set_prompt`
    P6   `P`, `sync`, `dt`, `ik`, `blend` — it is spawned per rollout, so it
         latches at birth and builds its solver and blend from the result
    P7   servo shaping and `arm_gain_scale`, re-latched on each new generation
    P9   everything, into the rollout's `manifest.json`
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime import ik as ik_solvers
from elrobot_deploy.config_loading.resolved import IK_SOLVERS, DeployConfig, ServoShaping
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import check_intervene_ik, validate_rollout

assert tuple(IK_SOLVERS) == schema.IK_OPTIONS, \
    "schema.IK_OPTIONS must be ik_solvers.SOLVERS, in its order"


def _shipped_blends() -> tuple:
    from elrobot_deploy.runtime.execution import chunk_blend
    return tuple(chunk_blend._BLENDS)


assert _shipped_blends() == schema.BLEND_OPTIONS, \
    "schema.BLEND_OPTIONS must be chunk_blend._BLENDS, in its order"

#: What `from_record` names a `BLEND_LAUNCH` record: "the launch's own blend",
#: which `apply` replaces with the launch config's `blend`.
LAUNCH_BLEND = "(launch)"

#: What `from_record` names an `IK_LAUNCH` record: "the launch's own solver",
#: which `apply` replaces with the launch config's `ik` (a `module:factory`).
LAUNCH_IK = "(launch)"

#: The servo fields the record carries, in `ServoShaping`'s own names.
SERVO_FIELDS = ("filter_hz", "filter_lead", "velocity_ff", "torque_ff",
                "arm_gain_scale")


class SettingsError(ConfigError, ValueError):
    """A settings request that cannot mean what it says."""


@dataclass(frozen=True)
class RolloutSettings:
    sync: bool
    replan_period: int
    control_hz: float
    ik: str
    sample_steps: int
    servo: ServoShaping
    blend: str = "temporal_ensemble"

    def __post_init__(self):
        # The mode shorthands name their default solver, as in `DeployConfig`.
        object.__setattr__(self, "ik", ik_solvers.normalize(str(self.ik)))

    # --- to and from the config ---------------------------------------------

    @classmethod
    def of(cls, cfg: DeployConfig) -> "RolloutSettings":
        """The settings a config carries — the launch's, before any START."""
        return cls(sync=bool(cfg.sync), replan_period=int(cfg.P),
                   control_hz=float(cfg.control_hz), ik=str(cfg.ik),
                   sample_steps=int(cfg.sample_steps), servo=cfg.servo,
                   blend=str(cfg.blend))

    def apply(self, cfg: DeployConfig) -> DeployConfig:
        """`cfg` with these settings in place. Everything else is untouched.
        `LAUNCH_IK` becomes the launch's own solver (`cfg.ik`)."""
        return dataclasses.replace(
            cfg, sync=bool(self.sync), replan_period_ticks=int(self.replan_period),
            control_hz=float(self.control_hz),
            ik=cfg.ik if self.ik == LAUNCH_IK else self.ik,
            sample_steps=int(self.sample_steps), servo=self.servo,
            blend=cfg.blend if self.blend == LAUNCH_BLEND else self.blend)

    # --- to and from the arena ----------------------------------------------

    def to_record(self) -> np.ndarray:
        rec = np.zeros((), dtype=schema.SETTINGS)
        rec["sync"] = int(bool(self.sync))
        rec["replan_period"] = int(self.replan_period)
        rec["control_hz"] = float(self.control_hz)
        # A shipped name is its index; anything else is the launch's own
        # `module:factory` — `check` refuses any OTHER module before a request
        # is stored, so a record never names a solver nobody launched.
        rec["ik"] = (schema.IK_OPTIONS.index(self.ik) if self.ik in schema.IK_OPTIONS
                     else schema.IK_LAUNCH)
        rec["sample_steps"] = int(self.sample_steps)
        rec["blend"] = (schema.BLEND_OPTIONS.index(self.blend)
                        if self.blend in schema.BLEND_OPTIONS else schema.BLEND_LAUNCH)
        for name in SERVO_FIELDS:
            rec[name] = getattr(self.servo, name)
        return rec

    @classmethod
    def from_record(cls, rec) -> "RolloutSettings":
        ik = int(rec["ik"])
        blend = int(rec["blend"])
        return cls(
            sync=bool(int(rec["sync"])),
            replan_period=int(rec["replan_period"]),
            control_hz=float(rec["control_hz"]),
            # An index the schema does not know is kept visible rather than
            # mapped to a default: `check` then refuses it by name.
            ik=(schema.IK_OPTIONS[ik] if ik < len(schema.IK_OPTIONS)
                else LAUNCH_IK if ik == schema.IK_LAUNCH else f"#{ik}"),
            sample_steps=int(rec["sample_steps"]),
            blend=(schema.BLEND_OPTIONS[blend] if blend < len(schema.BLEND_OPTIONS)
                   else LAUNCH_BLEND if blend == schema.BLEND_LAUNCH else f"#{blend}"),
            servo=ServoShaping(
                filter_hz=float(rec["filter_hz"]),
                filter_lead=bool(int(rec["filter_lead"])),
                velocity_ff=float(rec["velocity_ff"]),
                torque_ff=float(rec["torque_ff"]),
                arm_gain_scale=float(rec["arm_gain_scale"])))

    # --- to and from JSON (the dashboard, the manifest) ---------------------

    def as_dict(self) -> dict:
        return {"sync": bool(self.sync), "replan_period": int(self.replan_period),
                "control_hz": float(self.control_hz), "ik": self.ik,
                "sample_steps": int(self.sample_steps), "blend": self.blend,
                "servo": {name: getattr(self.servo, name) for name in SERVO_FIELDS}}

    def updated(self, changes: dict) -> "RolloutSettings":
        """These settings with `changes` (a partial `as_dict`) applied.

        Types are checked here, ranges are not — those are `check`'s, which
        knows the horizon and the robot. Unknown keys are refused by name: a
        typo silently ignored is a setting the operator believes they changed.
        """
        if not isinstance(changes, dict):
            raise SettingsError(f"settings must be an object, got {changes!r}")
        top = {"sync", "replan_period", "control_hz", "ik", "sample_steps", "servo",
               "blend"}
        unknown = sorted(set(changes) - top)
        if unknown:
            raise SettingsError(f"unknown settings {unknown}; known: {sorted(top)}")
        servo = changes.get("servo", {})
        if not isinstance(servo, dict):
            raise SettingsError(f"servo must be an object, got {servo!r}")
        bad = sorted(set(servo) - set(SERVO_FIELDS))
        if bad:
            raise SettingsError(f"unknown servo settings {bad}; known: "
                                f"{list(SERVO_FIELDS)}")

        def boolean(name, v):
            if not isinstance(v, bool):
                raise SettingsError(f"{name} must be true or false, got {v!r}")
            return v

        def integer(name, v):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v) or int(v) != v:
                raise SettingsError(f"{name} must be a whole number, got {v!r}")
            return int(v)

        def number(name, v):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v):
                raise SettingsError(f"{name} must be a finite number, got {v!r}")
            return float(v)

        kw = {}
        if "sync" in changes:
            kw["sync"] = boolean("sync", changes["sync"])
        if "replan_period" in changes:
            kw["replan_period"] = integer("replan_period", changes["replan_period"])
        if "control_hz" in changes:
            kw["control_hz"] = number("control_hz", changes["control_hz"])
        if "sample_steps" in changes:
            kw["sample_steps"] = integer("sample_steps", changes["sample_steps"])
        if "ik" in changes:
            # A shipped name (or a mode shorthand for one), or the launch's
            # own solver by `LAUNCH_IK` or by its `module:factory` — which of
            # those this launch has is `check`'s to say.
            ik = changes["ik"]
            if not isinstance(ik, str) or not ik:
                raise SettingsError(f"ik must be a solver name, got {ik!r}")
            ik = ik_solvers.normalize(ik)
            if ik not in schema.IK_OPTIONS and ik != LAUNCH_IK and ":" not in ik:
                raise SettingsError(f"ik = {ik!r} is not one of "
                                    f"{list(schema.IK_OPTIONS)}")
            kw["ik"] = ik
        if "blend" in changes:
            # A shipped name, or the launch's own (`LAUNCH_BLEND` or its
            # `module:function`) — `check` says which this launch has.
            b = changes["blend"]
            if not isinstance(b, str) or not b:
                raise SettingsError(f"blend must be a blend name, got {b!r}")
            if b not in schema.BLEND_OPTIONS and b != LAUNCH_BLEND and ":" not in b:
                raise SettingsError(f"blend = {b!r} is not one of "
                                    f"{list(schema.BLEND_OPTIONS)}")
            kw["blend"] = b
        if servo:
            sv = {}
            for name, v in servo.items():
                sv[name] = (boolean(f"servo.{name}", v) if name == "filter_lead"
                            else number(f"servo.{name}", v))
            kw["servo"] = dataclasses.replace(self.servo, **sv)
        # Negative whole numbers survive `integer` and would wrap in the u8
        # record; refused here, with the range checks left to `check`.
        for name in ("replan_period", "sample_steps"):
            if name in kw and kw[name] < 0:
                raise SettingsError(f"{name} = {kw[name]}, must be >= 0")
        return dataclasses.replace(self, **kw)


def check(settings: RolloutSettings, cfg: DeployConfig, hw, *,
          intervene: bool = False) -> DeployConfig:
    """Would a rollout with `settings` be accepted? Returns the rollout config.

    The launch refusals that depend on a switchable setting
    (`validate.validate_rollout`: `P` in `[2, H]`, sample steps, servo ranges,
    the IK mode and its sync requirement, the lookahead) plus the two that are
    about this launch:

    * `--intervene` runs the `arm_qp` IK and the plain linear servo — the
      combination the takeover was built and tested against (`__main__`);
    * a `module:factory` IK solver is the launch's: a rollout may run it
      only if the launch named it, and may switch away from it to a shipped
      one, but not to another module.

    Raises `ConfigError`. Called by P10 before it stores a request (so the
    dashboard gets its answer at once) and by P0 at START (the authority).
    """
    rollout = settings.apply(cfg)
    if ik_solvers.is_custom(rollout.ik) and rollout.ik != cfg.ik:
        raise ConfigError(
            f"ik = {rollout.ik!r} is not a shipped solver "
            f"({list(schema.IK_OPTIONS)}) and not this launch's own "
            f"({cfg.ik!r}); a module:factory solver is chosen at launch "
            f"(--ik), not between rollouts")
    if rollout.blend not in schema.BLEND_OPTIONS and rollout.blend != cfg.blend:
        raise ConfigError(
            f"blend = {rollout.blend!r} is not a shipped blend "
            f"({list(schema.BLEND_OPTIONS)}) and not this launch's own "
            f"({cfg.blend!r}); a module:function blend is chosen at launch "
            f"([control] blend), not between rollouts")
    validate_rollout(rollout, hw)
    check_intervene_ik(rollout, intervene=intervene)
    if intervene and rollout.servo != ServoShaping.linear():
        raise ConfigError(
            "--intervene runs the plain linear servo (filter_hz 0, torque_ff "
            "0): a takeover must stop the arm NOW, and the shaped servo has not "
            "been built or tested against it. Leave the servo settings alone.")
    return rollout


def latched(control, cfg: DeployConfig) -> tuple[DeployConfig, str, int]:
    """`(rollout config, prompt, generation)` from the ACTIVE block.

    What every process calls when a rollout starts (or, for P6, when it is
    born into one). Generation 0 means no START has been accepted yet: the
    launch settings, by definition — the block P0 writes at INIT says the same
    for the dashboard, but a reader does not depend on it having done so.

    The generation is read on both sides of the block and the read retried if
    it moved, so the settings returned are the ones that generation names.
    """
    while True:
        gen = control.rollout_generation
        if gen == 0:
            return cfg, cfg.prompt, 0
        rec, prompt = control.active()
        if control.rollout_generation == gen:
            return RolloutSettings.from_record(rec).apply(cfg), prompt, gen


__all__ = ["LAUNCH_IK", "RolloutSettings", "SERVO_FIELDS", "SettingsError",
           "check", "latched"]
