"""`g1d_deploy check-policy`'s second half: the policy itself, under its own
interpreter, with no robot and no arena.

    <policy python> -m g1d_deploy.runtime.processes.policy_host
        (stdin: {"dotted": "g1d_deploy.policy_contract.check.run_check",
                 "args": [cfg, hw]})

Run by `cli/check_policy.py` through `runtime/processes/policy_host.py`.
Design: `docs/deploying_a_policy/running_and_live_dashboard.md`, "The commands".

**Its goal: the policy loads and an inference completes.** It does what
P5 does before a rollout, in the same order and with the same objects:
`[policy.env]` applied, the class built from the deploy config
(`make_policy`), the prompt judged (`check_prompt_length`), the infra's dummy
observation and full-length dummy reader bound, `warmup()`, then one `infer`
on that dummy.

**What it does NOT judge: the action values.** The dummy is black frames and
a zero state, so what a real policy returns for it is meaningless — an
out-of-travel gripper or a non-unit quaternion on that input says nothing about
the adapter on real input. Only what is independent of the input fails the
check: `infer` returning the declared action type (`EefActions` for "eef",
`JointActions` for "joint", shapes enforced by their constructors) and an
action count the run can use (`1 <= n <= H`, `n >= P` in sync). The value
checks P5 applies live (finite, unit quaternions, hand travel, joint limits)
are run and reported as NOTES, never as failures.

numpy + stdlib + the core, like P5.
"""

from __future__ import annotations

import time
import traceback

import numpy as np

OK, FAIL, NOTE = "[  ok  ]", "[ FAIL ]", "[ note ]"


def _say(tag: str, what: str, detail: str = "") -> None:
    print(f"{tag} {what}" + (f"  —  {detail}" if detail else ""), flush=True)


def _summary(actions) -> str:
    """What came back, in the contract's own terms."""
    name = type(actions).__name__
    parts = [f"{name}, n={actions.n}"]
    if hasattr(actions, "pose"):
        xyz = np.asarray(actions.pose)[..., :3]
        parts.append("on the dummy: pose xyz range m "
                     f"[{xyz.min():.3f}, {xyz.max():.3f}]")
    if hasattr(actions, "arm_q"):
        q = np.asarray(actions.arm_q)
        parts.append(f"arm_q range rad [{q.min():.3f}, {q.max():.3f}]")
        w = np.asarray(actions.waist_q)
        parts.append(f"waist_q range rad [{w.min():.3f}, {w.max():.3f}]")
    h = np.asarray(actions.hand)
    parts.append(f"hand range [{h.min():.3f}, {h.max():.3f}]")
    return "; ".join(parts)


def _check_structure(actions, mode: str, cfg) -> None:
    """What does not depend on the input: the declared type (its constructor
    already enforced the shapes) and an action count the run can use."""
    from g1d_deploy.policy_contract.actions import (EEF, ChunkError, EefActions,
                                                    JointActions)
    want = EefActions if mode == EEF else JointActions
    if not isinstance(actions, want):
        raise ChunkError(f"action_mode = {mode!r} but infer() returned "
                         f"{type(actions).__name__}; it must return {want.__name__}")
    n = min(actions.n, cfg.H)          # P5 cuts a longer chunk to H
    if n < 1:
        raise ChunkError("infer() returned no actions")
    if cfg.sync and n < cfg.P:
        raise ChunkError(f"infer() returned {n} actions; sync needs at least "
                         f"replan_period = {cfg.P}. Return the whole chunk; the "
                         f"infra cuts it")


def run_construct(cfg, hw) -> None:
    """`check-policy` level 1's last step, in the policy's interpreter with the
    GPU hidden: construct the class (its `__init__`: config, tokenizer, stats,
    whatever it checks) and judge the prompt — what the inspection dashboard's
    "Check prompt" does. No `warmup`, no `infer`. Exit 1 at a failure."""
    import os

    from g1d_deploy.policy_contract import make_policy
    from g1d_deploy.policy_contract.describe import CPU_ONLY_ENV
    from g1d_deploy.runtime.processes.p5_policy import apply_policy_env

    step = "construct the policy (GPU hidden)"
    try:
        # As P5 constructs it: `[policy.env]` first (an adapter may read a
        # variable in `__init__`), then the GPU hidden again whatever it set.
        apply_policy_env(cfg)
        os.environ.update(CPU_ONLY_ENV)
        t0 = time.perf_counter()
        policy = make_policy(cfg, hw)
        _say(OK, step, f"{type(policy).__name__} in {time.perf_counter() - t0:.1f} s")
        step = "judge the prompt"
        policy.check_prompt_length(cfg.prompt)
        n_tok = policy.count_prompt_tokens(cfg.prompt)
        _say(OK, step, f"{cfg.prompt!r}"
                       + (f", {n_tok} tokens" if n_tok is not None else "")
                       + (f" (max {policy.max_prompt_tokens})"
                          if getattr(policy, "max_prompt_tokens", None) else ""))
    except SystemExit:
        raise
    except BaseException as exc:                     # noqa: BLE001 - reported
        _say(FAIL, step, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from None


def run_check(cfg, hw) -> None:
    """Construct, warm up and run the policy once on the infra's dummy
    observation; exit 0 when every step passed, 1 at the first that did not."""
    from g1d_deploy.policy_contract import (DummyReader, bind_runtime,
                                            camera_info, dummy_observation,
                                            make_policy)
    from g1d_deploy.policy_contract.actions import (Limits, first_actions,
                                                    validate_actions)
    from g1d_deploy.policy_contract.base import observation_cameras
    from g1d_deploy.runtime.processes.p5_policy import apply_policy_env

    step = "apply [policy.env]"
    try:
        apply_policy_env(cfg)

        step = "construct the policy"
        t0 = time.perf_counter()
        policy = make_policy(cfg, hw)
        _say(OK, step, f"{type(policy).__name__}("
                       f"checkpoint={cfg.checkpoint or None!r}) in "
                       f"{time.perf_counter() - t0:.1f} s")

        step = "judge the prompt"
        policy.check_prompt_length(cfg.prompt)
        n_tok = policy.count_prompt_tokens(cfg.prompt)
        _say(OK, step, f"{cfg.prompt!r}"
                       + (f", {n_tok} tokens" if n_tok is not None else "")
                       + (f" (max {policy.max_prompt_tokens})"
                          if getattr(policy, "max_prompt_tokens", None) else ""))

        step = "bind the dummy observation and reader"
        cams = observation_cameras(type(policy), cfg)
        views = cfg.views
        dummy = dummy_observation({c: views[c] for c in cams}, cfg.prompt)
        reader = DummyReader(views, dict(policy.history or {}),
                             state_hz=hw.lowstate_hz)
        from g1d_deploy.config_loading.rollout_settings import RolloutSettings
        bind_runtime(policy, dummy, reader, camera_info(cfg),
                     RolloutSettings.of(cfg))

        step = "warmup()"
        t0 = time.perf_counter()
        policy.warmup()
        _say(OK, step, f"{time.perf_counter() - t0:.1f} s")

        step = "infer() on the dummy observation"
        t0 = time.perf_counter()
        actions = policy.infer(dummy, reader)
        dt = time.perf_counter() - t0
        _say(OK, step, f"{dt * 1e3:.0f} ms")

        step = "the action type and count"
        _check_structure(actions, type(policy).action_mode, cfg)
        used = first_actions(actions, cfg.H)
        _say(OK, step, _summary(used) + (f" (cut from {actions.n} to action_horizon)"
                                         if used.n < actions.n else ""))

        # The values: what P5 would refuse LIVE. On the dummy they are a
        # policy's answer to black frames, so a refusal here is a note to read,
        # not a failure: an adapter that neither clips nor normalises would be
        # refused on the robot too, and this is the only offline hint of it.
        try:
            validate_actions(first_actions(actions, cfg.H),
                             type(policy).action_mode, cfg, Limits.from_robot(hw))
        except Exception as exc:                      # noqa: BLE001 - reported
            _say(NOTE, "on the dummy input, P5 would refuse these values",
                 f"{exc}. Meaningless for black frames; but does the adapter "
                 f"clip the hand to its travel and normalise quaternions? P5 "
                 f"refuses, it never clips")
    except SystemExit:
        raise
    except BaseException as exc:                     # noqa: BLE001 - reported
        _say(FAIL, step, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from None
    print("\nThe policy loads, warms up and completes an inference of the "
          "declared action type.\nNot checked: the action VALUES. The dummy is "
          "black frames and a zero state, so they mean nothing; whether the "
          "frame, units, quaternion order, arm order and TCP-vs-flange are right "
          "is for a replay on real observations (g1d_deploy inspect, 'policy "
          "checkpoint') and a supervised run.", flush=True)


__all__ = ["run_check", "run_construct"]
