"""elrobot_deploy names nothing outside itself.

Nothing in `elrobot_deploy/` imports or names the repository it is
developed in, its `experiments/`, the frozen `deploy_v2/`, its `envs/` machine
profiles or the box they were written for — not in an import, not in a string,
a regex, a launcher path, a comment or a doc — because the package is about to
become its own repository, where none of them exist. This is a TEXT search over
every source file, doc and example, with no allow-list.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The parent repository (and its `EGO2G1_*` variables), `experiments` as a
#: word, the two other trees, the machine profiles, and the box's GPU name.
FORBIDDEN = re.compile(r"ego2g1"
                       r"|(?<![A-Za-z0-9_])experiments(?![A-Za-z0-9_])"
                       r"|deploy_v2|jincheng_experiments|envs/"
                       r"|(?<![0-9])4090(?![0-9])", re.IGNORECASE)

TEXT = {".py", ".js", ".html", ".css", ".toml", ".md", ".yaml", ".yml",
        ".json", ".sh", ".txt"}


def _ignored(paths: list[Path]) -> set[Path]:
    """The gitignored ones — a rollout record written by a real run holds this
    checkout's absolute paths, and it is never shipped."""
    try:
        out = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                             input="\n".join(map(str, paths)), text=True,
                             capture_output=True, check=False).stdout
    except FileNotFoundError:           # no git: scan everything
        return set()
    return {Path(line) for line in out.splitlines()}


def _files():
    found = []
    for base in (ROOT / "src", ROOT / "docs", ROOT / "examples", ROOT / "scripts",
                 ROOT / "skills"):
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in TEXT and "__pycache__" not in p.parts:
                found.append(p)
    ignored = _ignored(found)
    yield from (p for p in found if p not in ignored)
    yield ROOT / "README.md"
    yield ROOT / "pyproject.toml"


def test_nothing_names_what_elrobot_deploy_leaves_behind():
    bad = []
    for p in _files():
        rel = p.relative_to(ROOT).as_posix()
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if FORBIDDEN.search(line):
                bad.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not bad, "elrobot_deploy names what it may not:\n" + "\n".join(bad)


def test_the_pattern_catches_what_it_must_and_nothing_else():
    for hit in ("from experiments.exp_006 import x", '"experiments/exp_004"',
                "python -m deploy_v2", "jincheng_experiments/", "experiments",
                "ego2g1_v2", "EGO2G1_STEP", "source envs/4090.sh", "the 4090"):
        assert FORBIDDEN.search(hit), hit
    for miss in ("deployment_optimization_experiments/04", "my_experiments_dir",
                 "experimentsX", "14090", "40901"):
        assert not FORBIDDEN.search(miss), miss
