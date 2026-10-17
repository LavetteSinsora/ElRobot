"""`python -m elrobot_deploy.dashboards.inspect inspect <root>` — the same as
`elrobot_deploy inspect <root>`; see `cli/inspect.py`."""

from __future__ import annotations

from elrobot_deploy.cli.inspect import main

if __name__ == "__main__":
    raise SystemExit(main())
