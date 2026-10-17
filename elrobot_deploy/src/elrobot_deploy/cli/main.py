"""`elrobot_deploy <command>`, the entry point (also `python -m elrobot_deploy`).

    elrobot_deploy run [--sim | --dry-run] [--no-phone] [--profile NAME|PATH] [--set k=v ...]
    elrobot_deploy view [--profile ...]                      the URDF and the poses, no robot
    elrobot_deploy config [--profile ...] [--set ...]        print the resolved settings
    elrobot_deploy phone-bench                               check the phone grip, no robot
    elrobot_deploy fake-phone ...                            a synthetic phone (for --sim)
    elrobot_deploy executor-test ...                         the executor's sine test
    elrobot_deploy camera list|probe|test ...                one camera on the bench, no robot

`run` is thin on purpose: it resolves the profile and the overrides (`config_loading.config`),
prints what was resolved, and hands over to P0 (`runtime/launcher/`), which owns every decision
after that.
"""

from __future__ import annotations

import argparse
import sys

from elrobot_deploy.config_loading.config import REPO, load
from elrobot_deploy.config_loading.errors import ConfigError


def _add_profile(p) -> None:
    p.add_argument("--profile", metavar="NAME|PATH", default=None,
                   help="robot profile: a shipped name (elrobot) or a TOML path")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   dest="sets", help="override one profile key for this run, e.g. "
                                     "--set teleop.lead_s=0.05 (VALUE is TOML). Repeatable.")
    p.add_argument("--sim", action="store_true",
                   help="simulated servos (no hardware): the same as --set bus.driver=sim")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elrobot_deploy",
        description="Deployment runtime for the ElRobot follower arm.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    r = sub.add_parser("run", help="run a session: check, home, teleoperate, exit",
                       description="Check the hardware, wait for Home (the dashboard shows the "
                                   "home pose), home the arm, wait for Start, then hand it to the "
                                   "phone. Home and Start come from the dashboard's buttons or "
                                   "Enter in this terminal. Ctrl-C ends the session through the "
                                   "exit ramp; a second Ctrl-C skips the ramp and leaves the arm "
                                   "held.")
    _add_profile(r)
    r.add_argument("--dry-run", action="store_true",
                   help="the real arm, torque OFF: it is read, every command is computed and "
                        "shown on the dashboard (the blue ghost), none is written")
    r.add_argument("--no-phone", action="store_true",
                   help="do not start the phone source: home, hold, and exit (a bring-up check)")
    r.add_argument("--auto-home", action="store_true",
                   help="home right after the check, without waiting for Home")
    r.add_argument("--auto-start", action="store_true",
                   help="start the rollout right after homing, without waiting for Start")
    r.add_argument("--runaway-only", action="store_true",
                   help="of the bus guards, keep only the runaway rule (a joint far off its "
                        "command and moving away: its torque goes off); no 'blocked' rule. Lost "
                        "position reads still hold the arm (torque on). Same as --set "
                        "safety.blocked=false")
    r.add_argument("--no-dashboard", action="store_true",
                   help="do not start the dashboard (Home and Start then come from Enter)")
    cams = r.add_mutually_exclusive_group()
    cams.add_argument("--no-cameras", action="store_true",
                      help="start no camera process (the profile's [cameras.*] are ignored)")
    cams.add_argument("--fake-cameras", action="store_true",
                      help="synthetic frames in place of every camera (no camera needed)")
    r.add_argument("--log-dir", default=str(REPO / "logs" / "sessions"),
                   help="where the bus process's tick log goes (default: ElRobot/logs/sessions)")

    c = sub.add_parser("config", help="print the resolved profile and exit")
    _add_profile(c)

    v = sub.add_parser("view", help="the dashboard without a robot: the URDF at any pose",
                       description="Serve the dashboard's 3D view with no session: the home and "
                                   "exit poses as ghosts, and sliders to pose the arm.")
    _add_profile(v)

    b = sub.add_parser("phone-bench", help="print the phone's motion in your axes; no robot",
                       description="Receive the phone's packets and, while ENGAGE is held, "
                                   "print its motion since the press in the operator's "
                                   "forward / left / up. Check the grip before driving.")
    _add_profile(b)
    b.add_argument("--port", type=int, default=None)

    sub.add_parser("fake-phone", help="send synthetic phone packets (its own --help)",
                   add_help=False)
    sub.add_parser("camera", help="one camera on the bench: list, probe, test (its own --help)",
                   add_help=False)
    sub.add_parser("executor-test", help="the executor's sine test (its own --help)",
                   add_help=False)
    return parser


def _config(args):
    sets = list(args.sets) + (["bus.driver=sim"] if args.sim else [])
    if getattr(args, "runaway_only", False):
        sets.append("safety.blocked=false")
    return load(args.profile, sets=sets).with_cameras(
        fake=getattr(args, "fake_cameras", False), none=getattr(args, "no_cameras", False))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["executor-test"]:
        from elrobot_deploy.cli import executor_test
        return executor_test.main(argv[1:])
    if argv[:1] == ["camera"]:
        from elrobot_deploy.cli import camera
        return camera.main(argv[1:])
    if argv[:1] == ["fake-phone"]:
        from elrobot_deploy.runtime.teleop import fake_phone
        return fake_phone.main(argv[1:])
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cfg = _config(args)
    except ConfigError as exc:
        parser.exit(2, f"elrobot_deploy: error: {exc}\n")
    if args.command == "phone-bench":
        from elrobot_deploy.runtime.processes.p11_phone import bench
        bench(cfg, port=args.port)
        return 0
    if args.command == "view":
        from elrobot_deploy.dashboard.server import run_view
        run_view(cfg)
        return 0
    print(cfg.describe())
    if args.command == "config":
        return 0
    if args.command == "run":
        from elrobot_deploy.runtime.launcher import Launcher
        if args.dry_run and args.sim:
            parser.error("--dry-run reads the real arm; --sim has none. Use one.")
        if args.dry_run:
            print("\nDRY RUN: the real arm is read, torque stays off, nothing is written.")
        elif cfg.bus.driver != "sim":
            print("\nREAL servos. Clear space around the arm, hand near the 12 V switch.")
        return Launcher(cfg, phone=not args.no_phone, auto_home=args.auto_home,
                        auto_start=args.auto_start, dry_run=args.dry_run,
                        dashboard=not args.no_dashboard, terminal=True,
                        log_dir=args.log_dir).run()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
