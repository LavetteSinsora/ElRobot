"""Every code reference in the docs, the skill AND the source still points at
something.

The docs and `skills/` name files (`src/g1d_deploy/runtime/ik/arm_qp.py`),
symbols in them (`...::DualArmSolver`), other docs (`docs/infra_logic/ik_solvers.md`),
folders (`examples/custom_modules/ik_solver.py`) and modules run with `python -m`. So do the
comments, docstrings and messages of the source, the examples, the scripts and
the tests, which also name files relative to the package
(`runtime/drivers/cameras.py`). A relayout that moves any of them turns those
into directions to nowhere, which a coding agent follows literally; this makes
it a test failure instead.

What is checked, in every file below:

* a path rooted at the checkout (`docs/`, `skills/`, `examples/`, `scripts/`,
  `src/`, `tests/`) exists;
* in source files, a path rooted at the package (`runtime/`, `robot/`,
  `config_loading/`, ...) exists under `src/g1d_deploy/`, and a bare
  `assets/...` (what a robot profile writes, relative to the package's
  `robot/` directory) exists there;
* `path.py::symbol` is defined in that file;
* a module run with `python -m` (a `g1d_deploy` one) is importable.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "g1d_deploy"

DOCS = sorted([*ROOT.glob("docs/**/*.md"), *ROOT.glob("skills/**/*.md"),
               ROOT / "README.md"])

#: A path rooted at the checkout: under one of these folders.
_PATH = re.compile(r"(?<![\w/.-])((?:src|examples|tests|skills|scripts|docs)/"
                   r"[\w./-]*[\w/])(?:::([A-Za-z_]\w*))?")
_MODULE = re.compile(r"python -m (g1d_deploy(?:\.\w+)+)")

#: The package's own top-level folders: how source comments name a sibling
#: module (`runtime/ik/arm_qp.py`). Read off the tree, so a new folder is
#: checked without an edit here.
_PKG_DIRS = sorted(p.name for p in PKG.iterdir()
                   if p.is_dir() and p.name != "__pycache__")
_PKG_PATH = re.compile(r"(?<![\w/.-])((?:" + "|".join(_PKG_DIRS) + r")/"
                       r"[\w./-]*[\w/])(?:::([A-Za-z_]\w*))?")
#: `assets/...` as a robot profile writes it: relative to the package's
#: `robot/` directory (`config_loading/resolved.py::MODEL_ROOT`).
_ASSET_PATH = re.compile(r"(?<![\w/.-])(assets/[\w./-]*[\w/])")

#: Text files of the source tree that carry comments worth checking.
_SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".md", ".js", ".html",
                    ".css", ".sh"}


def _refs(text: str):
    for m in _PATH.finditer(text):
        yield m.group(1), m.group(2)


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:         # a parent package that does not exist
        return False


def _defines(path: Path, symbol: str) -> bool:
    src = path.read_text()
    pattern = (rf"^\s*(?:class|def)\s+{symbol}\b|^\s*{symbol}\s*(?::[^=\n]*)?="
               rf"|^\s*@\w+\s*\n\s*def\s+{symbol}\b")
    return re.search(pattern, src, re.MULTILINE) is not None


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_path_and_symbol_a_doc_names_exists(doc):
    text = doc.read_text()
    missing = []
    for rel, symbol in _refs(text):
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif symbol and path.is_file() and not _defines(path, symbol):
            missing.append(f"{rel}::{symbol}")
    assert not missing, f"{doc.name} names what does not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_module_a_doc_runs_exists(doc):
    missing = [m for m in _MODULE.findall(doc.read_text()) if not _importable(m)]
    assert not missing, f"{doc.name} runs modules that do not exist: {missing}"


def test_the_docs_link_each_other_only_to_docs_that_exist():
    names = {p.relative_to(ROOT / "docs").as_posix() for p in ROOT.glob("docs/**/*.md")}
    for doc in DOCS:
        for name in re.findall(r"docs/([\w/-]+\.md)", doc.read_text()):
            assert name in names, f"{doc.name} links docs/{name}"


# --- the source: comments, docstrings, messages, toml comments ----------------


def _ignored(paths: list[Path]) -> set[Path]:
    """The gitignored ones (a rollout record under `examples/`, a build
    directory): never shipped, and full of this machine's own paths. As
    `tests/test_selfcontained.py` does it."""
    try:
        out = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                             input="\n".join(map(str, paths)), text=True,
                             capture_output=True, check=False).stdout
    except FileNotFoundError:           # no git: scan everything
        return set()
    return {Path(line) for line in out.splitlines()}


def _source_files() -> list[Path]:
    found = []
    for base, suffixes in ((ROOT / "src", _SOURCE_SUFFIXES),
                           (ROOT / "examples", _SOURCE_SUFFIXES),
                           (ROOT / "scripts", _SOURCE_SUFFIXES),
                           (ROOT / "tests", {".py", ".toml"})):
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
                found.append(p)
    ignored = _ignored(found)
    return [p for p in found if p not in ignored] + [
        ROOT / "machine.example.toml", ROOT / "pyproject.toml"]


SOURCES = _source_files()


def _missing(text: str, pattern: re.Pattern, bases: tuple[Path, ...]):
    """`(line number, reference)` for every match of `pattern` that exists
    under none of `bases`, or whose `::symbol` its file does not define."""
    for n, line in enumerate(text.splitlines(), 1):
        for m in pattern.finditer(line):
            rel = m.group(1)
            symbol = m.group(2) if pattern.groups > 1 else None
            hits = [b / rel for b in bases if (b / rel).exists()]
            if not hits:
                yield n, rel
            elif symbol and hits[0].suffix == ".py" and not _defines(hits[0], symbol):
                yield n, f"{rel}::{symbol}"


def test_every_path_the_source_names_exists():
    """A `docs/...`, `skills/...`, `examples/...`, `scripts/...`, `src/...` or
    `tests/...` path in a comment, a docstring, a message or a toml comment."""
    bad = [f"{p.relative_to(ROOT)}:{n}: {ref}"
           for p in SOURCES
           for n, ref in _missing(p.read_text(errors="replace"), _PATH, (ROOT,))]
    assert not bad, "the source names what does not exist:\n" + "\n".join(bad)


def test_every_package_relative_path_the_source_names_exists():
    """`runtime/drivers/cameras.py`, `robot/g1d_model.py::BASE_BODY`: how the
    source names a sibling module, relative to `src/g1d_deploy/`."""
    bad = []
    for p in SOURCES:
        text = p.read_text(errors="replace")
        bad += [f"{p.relative_to(ROOT)}:{n}: {ref}"
                for n, ref in _missing(text, _PKG_PATH, (PKG,))]
        # A robot profile's `urdf = "assets/..."` is relative to `robot/`.
        bad += [f"{p.relative_to(ROOT)}:{n}: {ref} (relative to robot/)"
                for n, ref in _missing(text, _ASSET_PATH, (PKG / "robot",))]
    assert not bad, ("the source names package paths that do not exist:\n"
                     + "\n".join(bad))


def test_every_module_the_source_runs_exists():
    bad = [f"{p.relative_to(ROOT)}: python -m {m}"
           for p in SOURCES
           for m in _MODULE.findall(p.read_text(errors="replace"))
           if not _importable(m)]
    assert not bad, "the source runs modules that do not exist:\n" + "\n".join(bad)


def test_the_source_check_catches_a_stale_pointer_and_passes_a_live_one():
    # Spelled in pieces, so that this file passes its own check.
    doc, mod, asset = "docs" + "/gone.md", "runtime" + "/gone.py", "assets" + "/gone.urdf"
    sym = "runtime" + "/timing.py::no_such_name"
    stale = f"see {doc} and `{mod}`, or\n`{sym}`, urdf = \"{asset}\""
    assert [r for _, r in _missing(stale, _PATH, (ROOT,))] == [doc]
    assert [r for _, r in _missing(stale, _PKG_PATH, (PKG,))] == [mod, sym]
    assert [r for _, r in _missing(stale, _ASSET_PATH, (PKG / "robot",))] == [asset]
    assert not _importable("g1d_deploy.gone.module")
    live = ("docs/overview.md, `runtime/timing.py::sleep_until`, "
            "`robot/assets/unitree_g1d/g1_d.urdf`, assets/unitree_g1d/g1_d.urdf, "
            "https://docs.astral.sh/uv/")
    for pattern, bases in ((_PATH, (ROOT,)), (_PKG_PATH, (PKG,)),
                           (_ASSET_PATH, (PKG / "robot",))):
        assert not list(_missing(live, pattern, bases))
