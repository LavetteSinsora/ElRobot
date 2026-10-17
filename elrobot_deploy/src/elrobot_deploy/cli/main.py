"""`elrobot_deploy <command>`, the entry point (also `python -m elrobot_deploy`).

    elrobot_deploy run [--sim] [--no-phone] [--profile NAME|PATH] [--set section.key=value ...]
    elrobot_deploy config [--profile ...] [--set ...]        print the resolved settings
    elrobot_deploy executor-test ...                         the executor's sine test

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

    r = sub.add_parser("run", help="run a session: home, teleoperate, exit",
                       description="Start the bus process, home the arm, and hand it to the "
                                   "phone. Ctrl-C ends the session through the exit ramp; a "
                                   "second Ctrl-C skips the ramp and leaves the arm held.")
    _add_profile(r)
    r.add_argument("--no-phone", action="store_true",
                   help="do not start the phone source: home, hold, and exit (a bring-up check)")
    r.add_argument("--wait-start", action="store_true",
                   help="at READY, wait for a START request instead of starting at once")
    r.add_argument("--log-dir", default=str(REPO / "logs" / "sessions"),
                   help="where the bus process's tick log goes (default: ElRobot/logs/sessions)")

    c = sub.add_parser("config", help="print the resolved profile and exit")
    _add_profile(c)

    sub.add_parser("executor-test", help="the executor's sine test (its own --help)",
                   add_help=False)
    return parser


def _config(args):
    sets = list(args.sets) + (["bus.driver=sim"] if args.sim else [])
    return load(args.profile, sets=sets)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["executor-test"]:
        from elrobot_deploy.cli import executor_test
        return executor_test.main(argv[1:])
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cfg = _config(args)
    except ConfigError as exc:
        parser.exit(2, f"elrobot_deploy: error: {exc}\n")
    print(cfg.describe())
    if args.command == "config":
        return 0
    if args.command == "run":
        from elrobot_deploy.runtime.launcher import Launcher
        if cfg.bus.driver != "sim":
            print("\nREAL servos. Clear space around the arm, hand near the 12 V switch.")
        return Launcher(cfg, phone=not args.no_phone, auto_start=not args.wait_start,
                        log_dir=args.log_dir).run()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
