"""A policy class's declarations, asked in the POLICY's interpreter.

    python -m elrobot_deploy.policy_contract.describe --entry SPEC [--out FILE]
        [--resolve-checkpoint RUN STEP]

The runtime reads a policy class before anything is spawned — its
`action_mode`, `anchor_camera`, `history`, `trained_*`, `end_effector`,
`max_prompt_tokens`, and its `resolve_checkpoint` hook
(`config_loading/deploy_file.py::resolve`, the inspection's replay picker). Importing the class means importing the
policy's module, and with `[policy] python` set
(`docs/deploying_a_policy/policy_contract.md`, "The adapter's own Python") that
module's dependencies — torch, jax, a numpy
2 — live in ANOTHER venv that the runtime cannot import from. So the runtime
asks that interpreter: this module, run there, imports the class and writes
its declarations as JSON; `describe` reads them back into a `PolicyInfo`.

A `PolicyInfo` answers the same attribute names the class does
(`info.action_mode`, `info.history`, ...), so the code that reads a class —
`check_against`, `check_end_effector`, `checkpoints.resolve` — reads one
unchanged. It is a description, never constructed and never run.

When the policy's interpreter IS this one (no `[policy] python`, the default),
the class is imported in-process, as it always was: no subprocess.

numpy + stdlib only: this runs in the policy's venv, which holds only the
minimal core of elrobot_deploy.
"""

from __future__ import annotations

import dataclasses

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: What `describe` hides from the interpreter it asks: reading class
#: attributes needs no GPU, and a describe that took one could starve a live
#: deployment on the same box.
CPU_ONLY_ENV = {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}

#: How long the policy's module may take to import.
TIMEOUT_S = 300


class DescribeError(RuntimeError):
    """The policy's interpreter could not describe the class. `kind` is the
    exception type the child raised (`ConfigError`, `PolicyError`, ...)."""

    def __init__(self, message: str, kind: str = ""):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class PolicyInfo:
    """A policy class, as its own interpreter described it."""

    entry: str
    #: `cls.__name__`, `cls.__module__`, `cls.__qualname__`, and the module's
    #: file (None for a module without one).
    name: str
    module: str
    qualname: str
    file: str | None
    #: Is it a subclass of `elrobot_deploy.policy_contract.Policy` at all?
    is_policy: bool
    #: `validate_policy_class`'s refusal, verbatim, or None when it passes.
    invalid: str | None
    action_mode: str | None = None
    end_effector: str | None = None
    anchor_camera: str | None = None
    uses_cameras: tuple[str, ...] | None = None
    history: dict[str, float] | None = None
    max_prompt_tokens: int | None = None
    deadline_shift: bool = True
    deadline_shift_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    trained_control_hz: float | None = None
    trained_action_horizon: int | None = None
    #: Does the class override `Policy.resolve_checkpoint`?
    resolves_checkpoints: bool = False
    #: The interpreter that answered, and what it runs.
    python: str = ""
    numpy_version: str = ""
    #: `shared_memory.schema.VERSION` in that interpreter's elrobot_deploy. P5 attaches
    #: to the arena with it, so it must equal the runtime's.
    schema_version: int = 0
    #: `resolve_checkpoint(run, step)`'s answer, when it was asked for.
    resolved_checkpoint: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def class_path(self) -> str:
        return f"{self.module}.{self.qualname}"

    def resolve_checkpoint(self, run, step) -> str | None:
        """The class's own `resolve_checkpoint(run, step)`, run where the
        class lives (`checkpoints.resolve` calls this exactly as it would the
        classmethod). None when the class does not override it."""
        if not self.resolves_checkpoints:
            return None
        info = describe(self.entry, python=self.python or None,
                        checkpoint=(str(run), str(step)))
        return info.resolved_checkpoint

    def as_json(self) -> dict:
        out = asdict(self)
        out["class"] = self.class_path
        return out

    @classmethod
    def from_json(cls, raw: dict) -> "PolicyInfo":
        raw = dict(raw)
        raw.pop("class", None)
        if raw.get("uses_cameras") is not None:
            raw["uses_cameras"] = tuple(raw["uses_cameras"])
        raw["deadline_shift_point"] = tuple(raw.get("deadline_shift_point")
                                            or (0.0, 0.0, 0.0))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


# --- the policy's side ---------------------------------------------------------


def _overrides_resolve_checkpoint(cls) -> bool:
    from elrobot_deploy.policy_contract.base import Policy
    own = getattr(cls, "resolve_checkpoint", None)
    base = Policy.__dict__["resolve_checkpoint"]
    return getattr(own, "__func__", own) is not getattr(base, "__func__", base)


def describe_class(cls, entry: str = "", *, checkpoint=None) -> PolicyInfo:
    """`cls`, imported in THIS interpreter, as a `PolicyInfo`."""
    import numpy as np

    from elrobot_deploy.runtime.shared_memory import schema
    from elrobot_deploy.policy_contract.base import (Policy, PolicyError,
                                                 validate_policy_class)

    is_policy = isinstance(cls, type) and issubclass(cls, Policy)
    try:
        validate_policy_class(cls)
        invalid = None
    except PolicyError as exc:
        invalid = str(exc)
    mod = sys.modules.get(getattr(cls, "__module__", ""), None)

    def attr(name, conv=None):
        v = getattr(cls, name, None)
        return None if v is None or conv is None else conv(v)

    uses = getattr(cls, "uses_cameras", None)
    hist = getattr(cls, "history", None)
    resolves = is_policy and _overrides_resolve_checkpoint(cls)
    resolved = None
    if checkpoint is not None and resolves:
        out = cls.resolve_checkpoint(Path(checkpoint[0]), str(checkpoint[1]))
        resolved = None if out is None else str(out)
    point = getattr(cls, "deadline_shift_point", (0.0, 0.0, 0.0))
    return PolicyInfo(
        entry=entry, name=getattr(cls, "__name__", repr(cls)),
        module=getattr(cls, "__module__", ""),
        qualname=getattr(cls, "__qualname__", ""),
        file=getattr(mod, "__file__", None), is_policy=is_policy, invalid=invalid,
        action_mode=attr("action_mode", str),
        end_effector=attr("end_effector", str),
        anchor_camera=attr("anchor_camera", str),
        uses_cameras=None if uses is None or isinstance(uses, str)
        else tuple(str(c) for c in uses),
        # Values kept as they are (not coerced): `check_against` and
        # `deploy_file` judge them, and must see what the class declared.
        history=dict(hist) if isinstance(hist, dict) else hist,
        max_prompt_tokens=getattr(cls, "max_prompt_tokens", None),
        deadline_shift=bool(getattr(cls, "deadline_shift", True)),
        deadline_shift_point=tuple(float(x) for x in np.ravel(point)),
        trained_control_hz=getattr(cls, "trained_control_hz", None),
        trained_action_horizon=getattr(cls, "trained_action_horizon", None),
        resolves_checkpoints=bool(resolves), python=sys.executable,
        numpy_version=np.__version__, schema_version=int(schema.VERSION),
        resolved_checkpoint=resolved)


def main(argv=None) -> int:
    import argparse
    import traceback

    p = argparse.ArgumentParser(prog="python -m elrobot_deploy.policy_contract.describe")
    p.add_argument("--entry", required=True)
    p.add_argument("--out", default=None,
                   help="write the JSON here (default: the last stdout line)")
    p.add_argument("--resolve-checkpoint", nargs=2, metavar=("RUN", "STEP"),
                   default=None)
    args = p.parse_args(argv)
    try:
        from elrobot_deploy.policy_contract.loading import load_entry
        cls = load_entry(args.entry)
        out = describe_class(cls, args.entry,
                             checkpoint=args.resolve_checkpoint).as_json()
        code = 0
    except Exception as exc:                                  # noqa: BLE001
        traceback.print_exc()
        out = {"error": str(exc) or type(exc).__name__,
               "kind": type(exc).__name__}
        code = 1
    text = json.dumps(out)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, flush=True)
    return code


# --- the runtime's side ----------------------------------------------------------


def same_interpreter(python: str | None) -> bool:
    """Is `python` this interpreter? Compared as paths, NOT resolved through
    symlinks: a venv's `bin/python` is a symlink to its base interpreter, and
    two venvs over one base Python are two environments."""
    if not python:
        return True
    return os.path.abspath(os.path.expanduser(python)) == os.path.abspath(sys.executable)


def describe(entry: str, *, python: str | None = None,
             checkpoint: tuple[str, str] | None = None,
             timeout: float = TIMEOUT_S) -> PolicyInfo:
    """Describe the class `entry` names, in `python` (None: this interpreter).

    Raises `DescribeError` with the child's own message when the class cannot
    be imported there — a missing module, a missing elrobot_deploy core in the
    policy venv (the message says how to install it), a broken import."""
    if same_interpreter(python):
        from elrobot_deploy.policy_contract.loading import load_entry
        return describe_class(load_entry(entry), entry, checkpoint=checkpoint)

    with tempfile.TemporaryDirectory(prefix="elr-describe-") as tmp:
        out = Path(tmp) / "info.json"
        cmd = [python, "-m", "elrobot_deploy.policy_contract.describe", "--entry", entry,
               "--out", str(out)]
        if checkpoint is not None:
            cmd += ["--resolve-checkpoint", str(checkpoint[0]), str(checkpoint[1])]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout,
                                  env={**os.environ, **CPU_ONLY_ENV})
        except FileNotFoundError:
            raise DescribeError(f"[policy] python = {python!r} does not exist") from None
        except subprocess.TimeoutExpired:
            raise DescribeError(f"{entry}: {python} did not import the class in "
                                f"{timeout:.0f} s") from None
        raw = json.loads(out.read_text()) if out.is_file() else None
    if raw is None or "error" in raw:
        stderr = (proc.stderr or "").strip()
        if raw is None and "No module named 'elrobot_deploy" in stderr:
            raise DescribeError(
                f"the policy interpreter {python} has no elrobot_deploy: install "
                f"the minimal core into its venv (`{python} -m pip install "
                f"<path to elrobot_deploy>` — no extras; docs/deploying_a_policy/policy_contract.md, \"The "
                f"adapter's own Python\")", "ModuleNotFoundError")
        if raw is None:
            tail = " | ".join(stderr.splitlines()[-3:])
            raise DescribeError(f"{entry}: {python} failed to describe the class "
                                f"(exit {proc.returncode}): {tail}")
        raise DescribeError(raw["error"], raw.get("kind", ""))
    # The CONFIGURED interpreter, not the child's `sys.executable`: a wrapper
    # (a script that sets PYTHONPATH, then execs a venv's python) reports the
    # bare python, and a later `resolve_checkpoint` run there cannot import
    # the adapter.
    return dataclasses.replace(PolicyInfo.from_json(raw), python=python)


__all__ = ["CPU_ONLY_ENV", "DescribeError", "PolicyInfo", "describe",
           "describe_class", "main", "same_interpreter"]


if __name__ == "__main__":
    raise SystemExit(main())
