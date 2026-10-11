"""`[policy] entry` -> the policy class, and every other plug-in by name.

Used at config resolution, by P5, and by P6/P7 for their plug-ins (`load_attr`).
Design: `docs/overview.md`, "What you can customize" (the two spellings).

Two spellings, told apart by the part before the colon:

    "some.package.module:MyPolicy"     import the module, take the attribute
    "path/to/file.py:MyPolicy"         import the FILE, take the attribute

A path is anything ending in `.py` or containing a `/`. Relative paths are
resolved against `base_dir` — the deploy file's directory — so a deploy.toml
can name the policy beside it wherever the command is run from; `normalize`
turns them absolute once, at resolution, so every spawned process loads the
same file.

**A file is imported as a member of its own package when it has one**: walk up
while `__init__.py` keeps existing, put the first directory without one on
`sys.path`, import by dotted name. That is what lets a policy file use relative
imports (`from ..common import model`). A file with no package around it
loads standalone.

This is the ONE loader: P5 (`base.make_policy`), the deploy-file resolution
and the inspection's policy replay all come through `load_entry`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path

from g1d_deploy.config_loading.errors import ConfigError


def split(spec: str) -> tuple[str, str]:
    """`"target:Class"` -> `("target", "Class")`, refusing anything else."""
    target, sep, attr = str(spec).rpartition(":")
    if not sep or not target or not attr or not attr.isidentifier():
        raise ConfigError(
            f"policy entry {spec!r} must be \"pkg.module:Class\" or "
            f"\"path/to/file.py:Class\"")
    return target, attr


def is_path(target: str) -> bool:
    return target.endswith(".py") or "/" in target


def normalize(spec: str, *, base_dir: Path | None = None) -> str:
    """`spec` with a relative file path made absolute against `base_dir`.
    Module entries are returned unchanged."""
    target, attr = split(spec)
    if not is_path(target):
        return spec
    p = Path(target).expanduser()
    if not p.is_absolute():
        p = (base_dir or Path.cwd()) / p
    return f"{p.resolve()}:{attr}"


def _package_name(path: Path) -> tuple[str, str] | None:
    """`(import root, dotted name)` for `path`, or None if it is standalone."""
    parts = [path.stem]
    d = path.parent
    while (d / "__init__.py").is_file():
        parts.append(d.name)
        d = d.parent
    if len(parts) == 1:
        return None
    return str(d), ".".join(reversed(parts))


def _import_file(path: Path):
    if not path.is_file():
        raise ConfigError(f"policy entry names {path}, which does not exist")
    packaged = _package_name(path)
    if packaged is not None:
        root, dotted = packaged
        # The file's own tree goes FIRST: another checkout of the same package
        # earlier on sys.path (the working directory, say) would otherwise be
        # imported in its place, or fail to hold this module at all.
        if sys.path[:1] != [root]:
            if root in sys.path:
                sys.path.remove(root)
            sys.path.insert(0, root)
        try:
            module = importlib.import_module(dotted)
        except ModuleNotFoundError as exc:
            top = dotted.split(".")[0]
            other = getattr(sys.modules.get(top), "__file__", None)
            if exc.name and dotted.startswith(exc.name) and other \
                    and not str(Path(other).resolve()).startswith(root + os.sep):
                raise ConfigError(
                    f"policy entry {path}: the package {top!r} was already "
                    f"imported from {Path(other).parent}, a different copy than "
                    f"the one this file is in ({root}). Start the command from "
                    f"another directory, or take that copy off PYTHONPATH.") from exc
            raise
        return module
    # Standalone: a name unique to the PATH, so two policies that are both
    # called policy.py cannot resolve to one module.
    digest = hashlib.blake2b(str(path).encode(), digest_size=4).hexdigest()
    name = f"_g1d_policy_{path.stem}_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"{path} is not importable as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # so dataclasses/pickle can find it
    spec.loader.exec_module(module)
    return module


def load_entry(spec: str, *, base_dir: Path | None = None) -> type:
    """Import the policy class `spec` names. Raises `ConfigError` for a bad
    spelling, a missing file or module, or an attribute that is not a class;
    an exception raised by the policy's own module is left to propagate, with
    its traceback, because that is the policy author's to read."""
    target, attr = split(normalize(spec, base_dir=base_dir))
    if is_path(target):
        module = _import_file(Path(target))
    else:
        try:
            module = importlib.import_module(target)
        except ModuleNotFoundError as exc:
            if exc.name and target.startswith(exc.name):
                raise ConfigError(f"policy entry {spec!r}: no module {exc.name!r} "
                                  f"on sys.path") from exc
            raise
    try:
        cls = getattr(module, attr)
    except AttributeError:
        raise ConfigError(f"policy entry {spec!r}: {target} defines no "
                          f"{attr!r}") from None
    if not isinstance(cls, type):
        raise ConfigError(f"policy entry {spec!r} is a {type(cls).__name__}, "
                          f"not a class")
    return cls


def load_attr(spec: str, *, base_dir: Path | None = None):
    """Any attribute a `"pkg.module:attr"` or `"path/file.py:attr"` names —
    the loader every plug-in slot shares (an IK factory, a blend function, a
    servo smoother or feedforward), with the same two spellings and the same
    package-aware file import as a policy entry. `ImportError` /
    `AttributeError` for what cannot be found; the module's own exceptions
    propagate."""
    target, sep, attr = str(spec).rpartition(":")
    if not sep or not target or not attr:
        raise ImportError(f"{spec!r} is not \"pkg.module:attr\" or "
                          f"\"path/to/file.py:attr\"")
    if is_path(target):
        p = Path(target).expanduser()
        if not p.is_absolute():
            p = (base_dir or Path.cwd()) / p
        try:
            module = _import_file(p.resolve())
        except ConfigError as exc:
            raise ImportError(str(exc)) from None
    else:
        module = importlib.import_module(target)
    return getattr(module, attr)


__all__ = ["is_path", "load_attr", "load_entry", "normalize", "split"]
