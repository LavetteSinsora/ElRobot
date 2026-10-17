"""A task folder built from the shipped examples, for tests that edit one.

`examples/deploy.toml` is the fully commented deploy file (every table is
there to be edited); the adapters are `examples/policies/*.py`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_ENTRY = 'entry = "policies/hold.py:Policy"'


def task(dst: Path, policy: str | None = "sine_eef") -> Path:
    """`dst/deploy.toml` (the commented file, its entry `policy.py:Policy`) and,
    unless `policy` is None, `dst/policy.py` (a copy of that example adapter).
    Returns the deploy file."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    text = (EXAMPLES / "deploy.toml").read_text()
    assert text.count(_ENTRY) == 1
    (dst / "deploy.toml").write_text(text.replace(_ENTRY, 'entry = "policy.py:Policy"'))
    if policy is not None:
        shutil.copy(EXAMPLES / "policies" / f"{policy}.py", dst / "policy.py")
    return dst / "deploy.toml"
