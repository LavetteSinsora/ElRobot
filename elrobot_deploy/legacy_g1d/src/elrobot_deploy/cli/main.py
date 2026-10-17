"""`elrobot_deploy <command>` — the entry point (also `python -m elrobot_deploy`).

    elrobot_deploy run deploy.toml [--step N] [--prompt ...] [--set k=v ...]
    elrobot_deploy check [--robot NAME|PATH] [--machine PATH] [--no-machine] [--no-robot]
    elrobot_deploy check-policy deploy.toml [--no-run] [...]
    elrobot_deploy inspect <root-or-record> [--port N]

and one command inspection spawns, callable by hand but not listed in --help:

    elrobot_deploy replay-policy --record DIR [--entry E] [--checkpoint C]

`run` is deliberately thin. It resolves the deploy file against its robot file,
the machine file and the command line (`config_loading/deploy_file.py::resolve`),
refuses an invalid config before anything is created, and hands over to P0;
every decision after that is in `runtime/launcher/`.

`check` is the one-time setup's: first whether THIS box can deploy at all
(packages, the machine file, GPU, /dev/shm, the NIC, the cores), then whether
the robot answers — every camera the robot file declares, and the state
streams. Read-only (`cli/check.py`).

The commands and flags for a user:
`docs/deploying_a_policy/running_and_live_dashboard.md`, "The commands" and
"Run flags".
"""

from __future__ import annotations

import argparse
import sys

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.derive import derive
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import intervene_ik_refusal, validate

#: Resolved lazily so `check` never imports the launcher's world.
Launcher = None


def _launcher():
    global Launcher
    if Launcher is None:
        from elrobot_deploy.runtime.launcher import Launcher as _L
        Launcher = _L
    return Launcher


def _add_run(sub) -> None:
    p = sub.add_parser(
        "run", help="run a deploy file on the robot",
        description="Run a deploy file on the robot. Every setting below "
                    "overrides the deploy file for this run only; the result is "
                    "printed with where each value came from and recorded in "
                    "every rollout's manifest.")
    p.add_argument("deploy_toml", help="path to a deploy.toml (examples/deploy.toml)")
    p.add_argument("--robot", metavar="NAME|PATH", help=(
        "robot file: a shipped profile name (g1d_fisheye_dex1, g1d_normal_dex1, "
        "...) or a path. Overrides the deploy file's `robot`."))
    p.add_argument("--machine", metavar="PATH", help=(
        "machine file (default: $ELROBOT_MACHINE, then <project root>/machine.toml)"))
    p.add_argument("--prompt", help="the task prompt for this run")
    p.add_argument("--checkpoint", metavar="PATH", help=(
        "serve this step directory, replacing the deploy file's [checkpoint]"))
    p.add_argument("--step", help=(
        "a step in the deploy file's [checkpoint] run: a number, `latest`, or "
        "an alias the policy's `resolve_checkpoint` understands"))
    p.add_argument("--out", metavar="DIR", help=(
        "record root; one directory per rollout inside it. Default: the deploy "
        "file's [record] out, else the machine file's [paths] records/<task>, "
        "else rollout_record/ beside the deploy file"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   dest="sets", help=(
        "set any deploy-file key, dotted: --set control.replan_period=10 "
        "--set home.tol_arm=0.2. VALUE is read as TOML when it parses "
        "(10, true, [1, 2], \"x\") and as a plain string otherwise. Repeatable; "
        "applied after the named flags."))
    p.add_argument("--intervene", action="store_true", help=(
        "start P11 and let a PICO operator take over mid-rollout. "
        "Off by default, and off means off: without it the intervention counter "
        "never leaves 0."))
    p.add_argument("--no-record", action="store_true",
                   help="run without P9. The default records every rollout: "
                        "a recorder you have to remember is one that is "
                        "missing from the run you most wanted")
    p.add_argument("--home-tol-arm", type=float, metavar="RAD",
                   help="[home] tol_arm: how far any arm joint may sit from home "
                        "when homing ends before P0 refuses (default 0.15)")
    p.add_argument("--home-tol-waist", type=float, metavar="RAD",
                   help="[home] tol_waist (default 0.05). Tighter than the arm's "
                        "on purpose — raise it only by what you measured")
    p.add_argument("--ik", metavar="SOLVER", default=None, help=(
        "[control] ik: the IK solver — " + " | ".join(cfgmod.IK_SOLVERS)
        + " (\"arm\" / \"waist\" = each mode's default: arm_planner, "
          "waist_planner), or module:factory for a research solver. Default "
          "arm_planner; arm_qp under --intervene"))
    p.add_argument("--sample-steps", type=int, metavar="N",
                   help="[control] sample_steps, handed to the policy")
    p.add_argument("--replan-period", type=int, metavar="TICKS",
                   help="[control] replan_period (P); 2 <= P <= H")
    p.add_argument("--action-horizon", type=int, metavar="TICKS",
                   help="[control] action_horizon (H); never above the policy's "
                        "declared trained_action_horizon")
    p.add_argument("--control-hz", type=float, metavar="HZ",
                   help="[control] control_hz; a different rate from the policy's "
                        "declared one is warned about, not refused")
    servo = p.add_argument_group(
        "servo shaping", "[control.servo]: how P7 turns the 30 Hz target line "
                         "into 500 Hz commands (config.ServoShaping)")
    servo.add_argument("--servo-filter-hz", type=float, metavar="HZ",
                       help="command filter cutoff (default 4; 0 = straight line)")
    servo.add_argument("--servo-no-lead", action="store_true",
                       help="do not read the target line ahead by the filter lag")
    servo.add_argument("--velocity-ff", type=float, metavar="SCALE",
                       help="scale on the dq feedforward (default 1)")
    servo.add_argument("--torque-ff", type=float, metavar="SCALE",
                       help="scale on the torque feedforward (default 0.5)")
    servo.add_argument("--arm-gain-scale", type=float, metavar="SCALE",
                       help="scale on the arm kp; kd by its square root (default 1)")


def _add_check(sub) -> None:
    c = sub.add_parser(
        "check", help="is this box ready, and does the robot answer?",
        description="First the machine: packages, the machine file, GPU, "
                    "/dev/shm, the network interface, the CPU table. Then the "
                    "robot: every camera the robot file declares and the state "
                    "streams. Every check runs and is reported; the exit status "
                    "is 0 only if none failed. Read-only: never publishes, "
                    "never moves anything.")
    c.add_argument("--robot", metavar="NAME|PATH", default=None,
                   help="robot file (default: g1d_fisheye_dex1)")
    c.add_argument("--machine", metavar="PATH", help=(
        "machine file (default: $ELROBOT_MACHINE, then <project root>/machine.toml)"))
    c.add_argument("--window", type=float, default=2.0, metavar="S",
                   help="how long to listen to each stream (default 2 s)")
    c.add_argument("--no-machine", action="store_true",
                   help="skip the machine checks")
    c.add_argument("--no-robot", action="store_true",
                   help="skip the robot checks (a box with no robot attached)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elrobot_deploy",
        description="Deployment runtime for the Unitree G1-D.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add_run(sub)
    _add_check(sub)
    from elrobot_deploy.cli import check_policy
    check_policy.add_parser(sub)
    # Offline, no robot: the inspection dashboard and the policy replay it
    # spawns (docs/deploying_a_policy/rollout_inspection.md). Registered by
    # `cli/inspect.py`;
    # the replay is plumbing, so it is left out of --help.
    from elrobot_deploy.cli import inspect as inspect_cli
    inspect_cli.add_inspect(sub)
    inspect_cli.add_replay(sub, hidden=True)
    return parser


def _flags(args) -> dict:
    return {name: getattr(args, name) for name in (
        "prompt", "robot", "step", "out", "home_tol_arm", "home_tol_waist",
        "sample_steps", "replan_period", "action_horizon", "control_hz", "ik",
        "servo_filter_hz", "velocity_ff", "torque_ff", "arm_gain_scale")}


def run(args, parser, argv) -> None:
    from elrobot_deploy.config_loading import deploy_file as deployfile
    from elrobot_deploy.rollout_record_utils import manifest as provenance

    for name in ("home_tol_arm", "home_tol_waist", "sample_steps",
                 "replan_period", "action_horizon", "control_hz"):
        value = getattr(args, name)
        if value is not None and not value > 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive, got {value}")
    sets = list(args.sets)
    if args.servo_no_lead:
        sets.insert(0, "control.servo.filter_lead=false")
    if args.intervene:
        # A takeover must stop the arm NOW, and the shaped servo has not
        # been built or tested against it: intervention runs the plain servo.
        # Written as overrides so the record's `sources` says why.
        linear = cfgmod.ServoShaping.linear()
        sets += [f"control.servo.{f}={str(getattr(linear, f)).lower()}"
                 for f in ("filter_hz", "filter_lead", "velocity_ff",
                           "torque_ff", "arm_gain_scale")]

    try:
        # The intervene/IK refusal comes FIRST, before any file beyond the
        # deploy file is read: `validate.check_intervene_ik` says why.
        _path, _text, tree = deployfile.load_tree(args.deploy_toml)
        final = deployfile.apply_overrides(tree, _flags(args), sets, args.checkpoint)
        if args.intervene:
            from elrobot_deploy.runtime import ik as ik_solvers

            # The takeover runs `arm_qp` (`validate.check_intervene_ik`). An ik
            # left unsaid — or the bare "arm", which names the arm mode's
            # default — becomes it, written as an override like the servo
            # above so the record's `sources` says why; an explicit other
            # solver is refused.
            ik = final.get("control", {}).get("ik")
            if ik in (None, "arm", ik_solvers.INTERVENE):
                sets.append(f"control.ik={ik_solvers.INTERVENE}")
            else:
                parser.exit(2, f"elrobot_deploy: error: "
                               f"{intervene_ik_refusal(ik)}\n")

        r = deployfile.resolve(args.deploy_toml, flags=_flags(args), sets=sets,
                               checkpoint=args.checkpoint, machine=args.machine,
                               record=not args.no_record)
        cfg, hw = r.cfg, r.hw
        # Before the arena, before a process, before the robot moves: every
        # startup refusal about the configuration is in here.
        validate(cfg, hw)
        if args.intervene:
            from elrobot_deploy.runtime.processes.p11_pico import (sdk_available,
                                                               sdk_missing_message)
            if not sdk_available():
                raise ConfigError(sdk_missing_message())
    except ConfigError as exc:
        parser.exit(2, f"elrobot_deploy: error: {exc}\n")

    derived = derive(cfg, hw)
    src = r.sources

    def tag(*names):
        got = sorted({src.get(n, "?") for n in names} - {"deploy", "infra"})
        return f"   ({', '.join(got)})" if got else ""

    print(f"deploy      {r.deploy_path}")
    print(f"robot       {r.robot.describe()}")
    print(f"machine     {r.machine.describe()}")
    print(f"policy      {cfg.policy_entry or '(none: EchoPolicy)'}")
    print(f"prompt      {cfg.prompt!r}{tag('prompt')}")
    print(f"checkpoint  {cfg.checkpoint or '(none)'}{tag('checkpoint')}")
    print(f"mode        {'sync' if cfg.sync else 'async'}  P={cfg.P}  H={cfg.H}  "
          f"control_hz={cfg.control_hz:g}  sample_steps={cfg.sample_steps}"
          f"{tag('sync', 'replan_period_ticks', 'action_horizon', 'control_hz', 'sample_steps')}")
    if r.trained:
        print(f"trained     {r.trained}")
    print("ik          not used (joint actions)"
          if cfg.policy_action_mode == "joint"
          else f"ik          {cfg.ik} ({cfg.ik_mode} mode){tag('ik')}")
    print(f"derived     {derived.describe()}")
    print(f"servo       {cfg.servo.describe()}{tag('servo')}"
          + ("   (--intervene: the straight-line servo)" if args.intervene else ""))
    if cfg.labels:
        print(f"labels      {cfg.labels}")
    print(f"cameras     anchor={cfg.primary_camera}  observed={list(cfg.observed)}  "
          f"{hw.camera_host} {hw.camera_ports}")
    print(f"hand        {hw.hand}  {list(hw.topic_hand_cmd)}  "
          f"travel [{hw.hand_min}, {hw.hand_max}]  open={hw.hand_open}")
    print(f"home tol    arm {cfg.home_tol_arm} rad  waist {cfg.home_tol_waist} rad"
          f"{tag('home_tol_arm', 'home_tol_waist')}")
    for w in r.warnings:
        print(f"WARNING     {w}")
    # The BASE, not a stamped directory: one session serves many rollouts on
    # one warmup, and P9 stamps each as it starts.
    if r.record_dir is not None:
        print(f"recording   {r.record_dir}/<per rollout>")

    # The policy travels as `cfg.policy_entry`; P5 builds it from there
    # (`policy_contract.make_policy`).
    _launcher()(cfg, hw, record_dir=r.record_dir, pico=args.intervene,
                provenance=provenance.build(r, argv=list(sys.argv[:1]) + list(argv))
                ).run()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args, parser, argv)
        return
    if args.command in ("inspect", "replay-policy"):
        from elrobot_deploy.cli import inspect as inspect_cli
        raise SystemExit(inspect_cli.run_inspect(args) if args.command == "inspect"
                         else inspect_cli.run_replay(args))
    if args.command == "check-policy":
        from elrobot_deploy.cli import check_policy
        raise SystemExit(check_policy.run(args))
    if args.command == "check":
        from elrobot_deploy.cli import check as preflight
        if args.no_machine and args.no_robot:
            parser.error("check: --no-machine and --no-robot leave nothing to check")
        raise SystemExit(preflight.check(
            robot=args.robot, machine=args.machine, window_s=args.window,
            do_machine=not args.no_machine, do_robot=not args.no_robot))


if __name__ == "__main__":
    main()
