"""`python -m g1d_deploy.dashboards.inspect inspect <root>` — the same as
`g1d_deploy inspect <root>`; see `cli/inspect.py`."""

from __future__ import annotations

from g1d_deploy.cli.inspect import main

if __name__ == "__main__":
    raise SystemExit(main())
