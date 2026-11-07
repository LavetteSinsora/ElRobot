"""Browse recorded ElRobot rollouts without connecting to hardware."""

import argparse
import time

from elrobot_deploy.config_loading.config import REPO


def main(argv=None):
    p = argparse.ArgumentParser(prog="elrobot_deploy inspect",
                                description="Offline cameras, robot poses and joint traces.")
    p.add_argument("root", nargs="?", default=str(REPO / "logs" / "records"),
                   help="recording directory or directory containing recordings")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--episodes", default=str(REPO / "astra" / "episodes"),
                   help="Astra episodes directory, to associate tasks and actions")
    args = p.parse_args(argv)
    from elrobot_deploy.inspection.server import serve
    try:
        server = serve(args.root, episodes=args.episodes, port=args.port)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(f"Rollout inspector: http://127.0.0.1:{server.server_address[1]} "
          "(recordings only; Ctrl-C to stop)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0
