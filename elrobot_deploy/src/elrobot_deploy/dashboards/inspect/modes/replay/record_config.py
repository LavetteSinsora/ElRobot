"""The `DeployConfig` a rollout ran with, rebuilt from its manifest.

P9 dumps EVERY `DeployConfig` field into `manifest["deploy_config"]`
(`p9_record._jsonable`: arrays as lists, dataclasses as dicts, tuples as
lists). This is the inverse: each field read back to the type the dataclass
declares, so the replay validates, shifts and cuts the policy's actions with
the numbers the rollout itself used — `P`, `H`, `dt`, `sync`, the cameras.

A field a record does not carry (an older record, a hand-written one) takes the
dataclass default, or a neutral value for the required ones. `camera_history_s`
is not read back: a replay serves history from the record, which holds every
frame, so the rings' sizes do not bound it (`widen_history`).

Used by the replay producer (`run.py`) and the replay mode's settings check.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.config_loading.resolved import CameraSpec, DeployConfig, ServoShaping

#: Required fields a record might lack, and what stands in for them.
_NEUTRAL = {
    "prompt": "", "checkpoint": "", "sync": True, "replan_period_ticks": 25,
    "action_horizon": 50, "control_hz": 30.0, "cameras": {}, "primary_camera": "",
    "home_q": np.zeros(schema.ARM_DOF), "home_waist": np.zeros(schema.WAIST_DOF),
    "home_hand": np.zeros((schema.N_HANDS, schema.HAND_DIM)),
    "exit_q": np.zeros(schema.ARM_DOF), "exit_waist": np.zeros(schema.WAIST_DOF),
    "exit_hand": np.zeros((schema.N_HANDS, schema.HAND_DIM)),
}

_TUPLE_FIELDS = {"crop", "lens_circle", "fallback_lens_circle"}


def _camera(raw: dict) -> CameraSpec:
    names = {f.name for f in dataclasses.fields(CameraSpec)}
    kw = {}
    for k, v in (raw or {}).items():
        if k not in names:
            continue
        kw[k] = tuple(v) if (k in _TUPLE_FIELDS and v is not None) else v
    return CameraSpec(**kw)


def _tuple(v):
    return tuple(_tuple(x) for x in v) if isinstance(v, (list, tuple)) else v


def deploy_config(manifest: dict) -> DeployConfig:
    """The rollout's `DeployConfig` (its settings already applied: P9 dumps the
    latched per-rollout config)."""
    dumped = dict(manifest.get("deploy_config") or {})
    kw = {}
    for f in dataclasses.fields(DeployConfig):
        name = f.name
        if name not in dumped:
            if name in _NEUTRAL:
                kw[name] = _NEUTRAL[name]
            continue
        v = dumped[name]
        if name == "cameras":
            v = {c: _camera(spec) for c, spec in (v or {}).items()}
        elif name == "servo":
            fields = {x.name for x in dataclasses.fields(ServoShaping)}
            v = ServoShaping(**{k: val for k, val in (v or {}).items() if k in fields})
        elif name in ("camera_history_s",):
            v = {}
        elif isinstance(_NEUTRAL.get(name), np.ndarray):
            v = np.asarray(v, dtype=np.float64).reshape(_NEUTRAL[name].shape)
        elif isinstance(f.default, tuple) or name == "observed_cameras":
            v = _tuple(v or ())
        elif name in ("policy_env", "labels"):
            v = dict(v or {})
        kw[name] = v
    return DeployConfig(**kw)


def widen_history(cfg: DeployConfig, history: dict | None) -> DeployConfig:
    """`cfg` with rings as deep as `history` asks. Offline the record holds
    every frame, so `check_against`'s ring-depth refusal would be about a ring
    that does not exist here."""
    history = dict(history or {})
    cams = {k: float(v) for k, v in history.items() if k != "state"}
    state_s = max(float(cfg.state_history_s), float(history.get("state", 0.0)))
    return dataclasses.replace(cfg, camera_history_s=cams, state_history_s=state_s)


__all__ = ["deploy_config", "widen_history"]
