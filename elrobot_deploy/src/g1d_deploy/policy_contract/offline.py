"""An adapter, built offline exactly as P5 builds it — for an adapter author's
own checks (conversions against reference data, a diff against an older
serving path, recorded samples through `infer`). Described in
`skills/deploy-a-policy/SKILL.md`, "Part 3: verify without the robot".

    # in the ADAPTER's interpreter ([policy] python):
    from g1d_deploy.policy_contract.offline import build
    policy, cfg, hw = build("path/to/deploy.toml")            # constructed + bound
    policy, cfg, hw = build("path/to/deploy.toml", warmup=True)   # + warmup()

What it does is P5's startup up to the rollouts: resolve the deploy file (the
same layers and refusals as `run`), apply `[policy.env]`, construct the class
(`make_policy`), and bind what the infra binds — the dummy observation, a
full-length dummy reader, `Policy.cameras` (outside a live run no lens is
fitted: a fisheye's square is the robot file's `fallback_lens_circle`'s) and `Policy.settings`. With
`warmup=True` it then calls `policy.warmup()`, which takes the GPU if the
policy uses one. `policy.dummy_observation()` is the dummy it was bound with:
black frames, zero joints, both flanges at the base origin with identity
orientation, a closed hand, `t_anchor = 0`.

No robot, no arena: `reader.arena` stays None, as in check-policy.
"""

from __future__ import annotations

from pathlib import Path


def build(deploy_toml: str | Path, *, warmup: bool = False, robot: str | None = None,
          machine: str | None = None, sets=()):
    """-> `(policy, cfg, hw)`. `robot` / `sets` are `--robot` / `--set`."""
    from g1d_deploy.config_loading import deploy_file
    from g1d_deploy.config_loading.rollout_settings import RolloutSettings
    from g1d_deploy.policy_contract import (DummyReader, bind_runtime, camera_info,
                                            dummy_observation, make_policy)
    from g1d_deploy.policy_contract.base import observation_cameras
    from g1d_deploy.runtime.processes.p5_policy import apply_policy_env

    r = deploy_file.resolve(deploy_toml, flags={"robot": robot} if robot else None,
                            sets=tuple(sets), machine=machine, record=False)
    cfg, hw = r.cfg, r.hw
    apply_policy_env(cfg)
    policy = make_policy(cfg, hw)
    views = cfg.views
    cams = observation_cameras(type(policy), cfg)
    dummy = dummy_observation({c: views[c] for c in cams}, cfg.prompt)
    reader = DummyReader(views, dict(policy.history or {}), state_hz=hw.lowstate_hz)
    bind_runtime(policy, dummy, reader, camera_info(cfg), RolloutSettings.of(cfg))
    if warmup:
        policy.warmup()
    return policy, cfg, hw


__all__ = ["build"]
