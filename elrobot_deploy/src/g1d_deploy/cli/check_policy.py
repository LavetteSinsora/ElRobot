"""`g1d_deploy check-policy deploy.toml` — is this adapter ready to run?
No robot, no arena, nothing moves.

    g1d_deploy check-policy deploy.toml [--machine PATH] [--robot NAME|PATH]
        [--checkpoint PATH | --step N] [--prompt TEXT] [--set KEY=VALUE ...]
        [--no-run]

Two levels, each what `run` itself would do first:

1. **The deploy file resolves.** The robot file, the machine file and the
   policy class (described in its own interpreter, never imported here) are
   read and layered exactly as `run` layers them, and every startup refusal
   (`config_loading.validate`) is applied: a camera the robot does not have,
   a named checkpoint that does not exist, a `[policy] python` that is not
   executable, a class whose declarations break the contract.
2. **The policy loads and an inference completes.** Under
   `[policy] python`, the class is constructed (from its checkpoint, if it has
   one), the prompt
   is judged, `warmup()` runs, and one `infer` on the infra's dummy
   observation must return the declared action type with a usable count
   (`policy_contract/check.py`). The action VALUES are not judged — the dummy
   is black frames — only noted.

   **For a policy with weights this loads them onto the GPU** (with
   `[policy.env]`'s allocator flags: a JAX policy may preallocate most of the
   card), for as long as a warmup takes. It is refused while a live deployment holds the GPU (its
   arena in /dev/shm), and anything else on the card — training, another
   server — can OOM it or be OOMed by it. `--no-run` stops after level 1.

Passing both means the run will get through startup and warmup. It does not
mean the actions are RIGHT: see `policy_contract/check.py`.

For a user: `docs/deploying_a_policy/running_and_live_dashboard.md`, "The
commands", and `skills/deploy-a-policy/SKILL.md`, "Part 3: verify without the
robot".
"""

from __future__ import annotations

import os
import subprocess
import sys

from g1d_deploy.config_loading.errors import ConfigError

OK, FAIL = "[  ok  ]", "[ FAIL ]"


def add_parser(sub) -> None:
    p = sub.add_parser(
        "check-policy",
        help="check a deploy file and its policy offline (no robot)",
        description="Resolve a deploy file the way `run` does, then load the "
                    "policy under its own interpreter, warm it up, run one "
                    "inference on a dummy observation and check the actions "
                    "against the contract. Nothing talks to the robot.")
    p.add_argument("deploy_toml", help="path to a deploy.toml")
    p.add_argument("--robot", metavar="NAME|PATH",
                   help="robot file, overriding the deploy file's `robot`")
    p.add_argument("--machine", metavar="PATH", help=(
        "machine file (default: $G1D_MACHINE, then <project root>/machine.toml)"))
    p.add_argument("--prompt", help="the prompt to judge and infer with")
    p.add_argument("--checkpoint", metavar="PATH",
                   help="a step directory, replacing the deploy file's [checkpoint]")
    p.add_argument("--step", help="a step in the deploy file's [checkpoint] run")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   dest="sets", help="set any deploy-file key, as for `run`")
    p.add_argument("--no-run", action="store_true",
                   help="stop after level 1: do not warm the policy up or take "
                        "the GPU")


def _flags(args) -> dict:
    flags = dict.fromkeys((
        "prompt", "robot", "step", "out", "home_tol_arm", "home_tol_waist",
        "sample_steps", "replan_period", "action_horizon", "control_hz", "ik",
        "servo_filter_hz", "velocity_ff", "torque_ff", "arm_gain_scale"))
    flags.update(prompt=args.prompt, robot=args.robot, step=args.step)
    return flags


def run(args) -> int:
    from g1d_deploy.config_loading import deploy_file
    from g1d_deploy.config_loading.validate import validate

    print("--- 1. the deploy file resolves")
    try:
        r = deploy_file.resolve(args.deploy_toml, flags=_flags(args),
                                sets=list(args.sets), checkpoint=args.checkpoint,
                                machine=args.machine, record=False)
        cfg, hw = r.cfg, r.hw
        validate(cfg, hw)
    except ConfigError as exc:
        print(f"{FAIL} {exc}")
        return 1
    print(f"{OK} {r.deploy_path}")
    print(f"         robot       {r.robot.describe()}")
    print(f"         policy      {cfg.policy_entry or '(none: EchoPolicy)'}"
          f"  [{cfg.policy_action_mode or 'eef'} actions]")
    print(f"         python      {cfg.policy_python or sys.executable}")
    print(f"         checkpoint  {cfg.checkpoint or '(none)'}")
    print(f"         prompt      {cfg.prompt!r}")
    print(f"         cameras     anchor={cfg.primary_camera}  "
          f"observed={list(cfg.observed)}")
    print(f"         timing      control_hz={cfg.control_hz:g}  H={cfg.H}  "
          f"P={cfg.P}  {'sync' if cfg.sync else 'async'}")
    if r.trained:
        print(f"         trained     {r.trained}")
    for w in r.warnings:
        print(f"WARNING  {w}")

    # Still level 1: the class's own __init__ and prompt judgement, where they
    # will run, with the GPU hidden — so a failure there costs seconds and no
    # GPU, and never waits for level 2.
    from g1d_deploy.config_loading import json_transport as codec
    from g1d_deploy.policy_contract.describe import CPU_ONLY_ENV
    python = cfg.policy_python or sys.executable
    sys.stdout.flush()
    proc = subprocess.run(
        [python, "-m", "g1d_deploy.runtime.processes.policy_host"],
        input=codec.dumps({"dotted": "g1d_deploy.policy_contract.check.run_construct",
                           "args": (cfg, hw), "kwargs": {}}),
        text=True, env={**os.environ, **CPU_ONLY_ENV})
    if proc.returncode != 0:
        return 1
    if args.no_run:
        return 0

    from g1d_deploy.dashboards.web import gpu
    why = gpu.refusal()
    if why:
        print(f"\n{FAIL} level 2 not run: {why}")
        return 1
    print("\n--- 2. the policy loads and an inference completes "
          "(its own interpreter; takes the GPU)", flush=True)
    payload = codec.dumps({"dotted": "g1d_deploy.policy_contract.check.run_check",
                           "args": (cfg, hw), "kwargs": {}})
    proc = subprocess.run(
        [python, "-m", "g1d_deploy.runtime.processes.policy_host"],
        input=payload, text=True)
    return 0 if proc.returncode == 0 else 1


__all__ = ["add_parser", "run"]
