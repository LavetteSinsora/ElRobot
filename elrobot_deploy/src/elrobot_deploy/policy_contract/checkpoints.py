"""`[checkpoint]` -> one directory.

Called at config resolution (`config_loading/deploy_file.py::resolve`), before
any process starts. Design: `docs/deploying_a_policy/deploy_toml.md`,
"`[checkpoint]`: which weights". The table is optional: a policy that loads
nothing from disk names no checkpoint and receives `checkpoint = None`.

Two ways to name it, never both:

    [checkpoint]
    path = "/runs/exp/best/9999"          # this step directory, as-is

    [checkpoint]
    run = "/runs/exp/my_run"              # a run directory ...
    step = 9999                           # ... and a step in it: a number,
                                          # "latest", or a policy's own alias

elrobot_deploy knows exactly one layout: numeric step subdirectories of the run,
`latest` meaning the highest. Anything else — a layout that stores steps under
`best/`, aliases like "crowned", an EMA subdirectory — is the POLICY's, through
an optional classmethod, which is consulted first whenever it exists:

    @classmethod
    def resolve_checkpoint(cls, run: Path, step: str) -> Path: ...

`step` reaches it as a string (`"latest"`, `"9999"`, `"crowned"`), exactly as
written, so the policy decides what its own words mean.
"""

from __future__ import annotations

from pathlib import Path

from elrobot_deploy.config_loading.errors import ConfigError


def numeric_steps(run: Path) -> list[int]:
    """The numeric subdirectories of `run`, ascending."""
    return sorted(int(p.name) for p in run.iterdir()
                  if p.is_dir() and p.name.isdigit())


def resolve(cls=None, *, path=None, run=None, step=None,
            base_dir: Path | None = None) -> str:
    """The checkpoint directory as a string, "" when none is named.

    Relative paths are resolved against `base_dir` (the deploy file's
    directory). Existence of a `path` is left to `config_loading/validate.py`, which
    refuses it before anything spawns; a `run` must exist here, since its
    listing is the answer.
    """
    def _abs(p) -> Path:
        q = Path(str(p)).expanduser()
        if not q.is_absolute() and base_dir is not None:
            q = base_dir / q
        return q

    if path not in (None, "") and run not in (None, ""):
        raise ConfigError("[checkpoint] names both `path` and `run`; a path is "
                          "the step directory itself, a run needs a `step` — "
                          "pick one")
    if run in (None, ""):
        if step not in (None, ""):
            raise ConfigError(f"[checkpoint] step = {step!r} without a `run` to "
                              f"find it in")
        return str(_abs(path)) if path not in (None, "") else ""

    run_dir = _abs(run)
    step_s = "latest" if step in (None, "") else str(step)

    # The class's own layout first; None (the base class's answer) means
    # "use the numbered-subdirectory rule below".
    hook = getattr(cls, "resolve_checkpoint", None) if cls is not None else None
    if callable(hook):
        out = hook(run_dir, step_s)
        if out is not None:
            return str(out)

    if not run_dir.is_dir():
        raise ConfigError(f"[checkpoint] run {str(run_dir)!r} is not a directory")
    if step_s == "latest":
        steps = numeric_steps(run_dir)
        if not steps:
            raise ConfigError(
                f"[checkpoint] step = \"latest\" but {run_dir} has no numeric "
                f"step directories. Name the step directory with `path`, or give "
                f"the policy a `resolve_checkpoint` classmethod for its layout.")
        return str(run_dir / str(steps[-1]))
    if step_s.isdigit():
        out = run_dir / str(int(step_s))
        if not out.is_dir():
            have = numeric_steps(run_dir)
            raise ConfigError(f"[checkpoint] step {step_s} is not in {run_dir} "
                              f"(numeric steps there: {have[-5:] or 'none'})")
        return str(out)
    raise ConfigError(
        f"[checkpoint] step = {step_s!r} is not a number or \"latest\", and the "
        f"policy class defines no `resolve_checkpoint(run, step)` to say what "
        f"its own aliases mean.")


__all__ = ["numeric_steps", "resolve"]
