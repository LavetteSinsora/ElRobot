"""The two command-line entries inspection adds to `elrobot_deploy`.

    elrobot_deploy inspect [root-or-record] [--port N] [--mode pkg.mod:Mode]
    elrobot_deploy replay-policy --record DIR [--entry E] [--checkpoint C] ...

`inspect` serves the dashboard (`dashboards/inspect/server.py`); `replay-policy` is the
policy replay producer the dashboard spawns (`dashboards/inspect/modes/replay/run.py`) as
`<this python> -m elrobot_deploy replay-policy`, runnable by hand as well but not
listed in `elrobot_deploy --help`. Both are registered here so `cli/main.py`
only has to hand them a subparser each.

For a user: `docs/deploying_a_policy/rollout_inspection.md`.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser


def add_inspect(sub) -> None:
    p = sub.add_parser(
        "inspect", help="look at recorded rollouts in a browser (no robot)",
        description="Serve rollout inspection over every record under ROOT "
                    "(any directory holding a manifest.json P9 wrote), or over "
                    "one record directory. Read-only against the records; "
                    "replays and benches it starts write beside them. Options "
                    "may also be set in the machine file's [inspect] table.")
    p.add_argument("root", nargs="?", default=None,
                   help="a directory to search for records, or one record "
                        "(default: the machine file's [paths] records)")
    p.add_argument("--port", type=int, default=None,
                   help="HTTP port (default: [inspect] port, else 8770)")
    p.add_argument("--host", default=None,
                   help="bind address (default 127.0.0.1: ssh -L to reach it)")
    p.add_argument("--machine", metavar="PATH",
                   help="machine file for its [inspect] table (default: "
                        "$ELROBOT_MACHINE, then <project root>/machine.toml; optional here)")
    p.add_argument("--mode", action="append", default=[], metavar="PKG.MOD:CLASS",
                   dest="modes", help="mount an extra mode (repeatable); see "
                                      "docs/deploying_a_policy/rollout_inspection.md, "
                                      "'Writing a mode'")
    p.add_argument("--executor-producer", metavar="PKG.MOD",
                   help="module the executor mode runs as `python -m PKG.MOD "
                        "--source RECORD --variants a,b` to make replays")
    p.add_argument("--runs", action="append", default=[], metavar="DIR",
                   help="a directory the policy replay's run picker starts "
                        "from (repeatable)")
    p.add_argument("--open", action="store_true", help="open a browser")


def run_inspect(args) -> int:
    from elrobot_deploy.dashboards.inspect import options as options_mod
    from elrobot_deploy.dashboards.inspect import server

    from elrobot_deploy.config_loading.errors import ConfigError
    try:
        opts = options_mod.resolve(
            args.root, machine=args.machine, host=args.host, port=args.port,
            modes=args.modes, executor_producer=args.executor_producer,
            runs=args.runs)
    except ConfigError as exc:
        raise SystemExit(f"elrobot_deploy inspect: {exc}") from None
    if not opts.root.exists():
        raise SystemExit(f"elrobot_deploy inspect: {opts.root} does not exist")
    print(f"records     {opts.root}")
    if opts.machine_file:
        print(f"machine     {opts.machine_file}")
    if opts.modes:
        print(f"extra modes {', '.join(opts.modes)}")
    if opts.executor_producer:
        print(f"executor    {opts.executor_producer}")
    if args.open:
        url = f"http://{opts.host}:{opts.port}"
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve(opts)
    return 0


def add_replay(sub, *, hidden: bool = False) -> None:
    """`hidden`: register it without a help line, so argparse leaves it out of
    the parent's command list (it still parses and has its own --help)."""
    from elrobot_deploy.dashboards.inspect.modes.replay import run as replay_run

    kw = {} if hidden else {"help": "re-run a rollout's observations through a policy"}
    p = sub.add_parser("replay-policy", **kw,
                       description=replay_run.__doc__.split("\n\n")[0])
    replay_run.add_arguments(p)


def run_replay(args) -> int:
    from elrobot_deploy.dashboards.inspect.modes.replay import run as replay_run

    return replay_run.main_args(args)


def main(argv=None) -> int:
    """`python -m elrobot_deploy.dashboards.inspect ...` — the same two commands."""
    parser = argparse.ArgumentParser(prog="python -m elrobot_deploy.dashboards.inspect")
    sub = parser.add_subparsers(dest="command", required=True)
    add_inspect(sub)
    add_replay(sub)
    args = parser.parse_args(argv)
    return run_inspect(args) if args.command == "inspect" else run_replay(args)


__all__ = ["add_inspect", "add_replay", "main", "run_inspect", "run_replay"]
