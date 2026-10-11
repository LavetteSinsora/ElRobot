"""The IK solvers P6 can run: `ik = NAME` (`DeployConfig.ik`, `--ik NAME`).

The registry and the interface; built and called by P6
(`runtime/processes/p6_target.py`) and by the inspection dashboard's IK mode.
Design, and how to add a solver: `docs/infra_logic/ik_solvers.md`.

`ik` names a SOLVER, and the solver's MODE — which joints it may move — is a
property of the solver, not a second setting that could disagree with it:

    name            mode    what
    arm_planner     arm     the streamed branch planner (`arm_planner.py`).
                            **The default** (`ik = "arm"` means this).
    arm_qp          arm     `arm_qp.DualArmSolver`, the per-tick mink QP.
    waist_planner   waist   the arm planner on a slewed, settled, lag-aware
                            trunk plan (`waist_planner.py`). `ik = "waist"`
                            means this.
    waist_qp        waist   `waist_qp.WholeBodySolver`'s waist tier, the
                            whole-body QP.

`arm` mode reads the trunk and never commands it; `waist` mode also plans and
commands the waist's yaw and bow, and is sync only (`validate._check_ik`). The
shorthands `"arm"` / `"waist"` name each mode's default (`DEFAULT`), and are
normalized to it when a `DeployConfig` is built, so a record always names the
solver that ran.

A research solver is plugged in by `"module:factory"` (how an IK bench is
run): the factory must carry a `.mode` of "arm" or "waist", which is
the mode that run is then held to. It is fixed per launch — the dashboard can
switch among the four names between rollouts but cannot name a module.

P6 asks its solver for two things and nothing else (`runtime/processes/p6_target.py`):

    solve(pose_2x7, waist_3, seed_q14, budget) -> (q_arm14, err_m, ok)
    plan_chunk(poses_Kx2x7, waist_meas_3, seed_q14, *, budget) -> (K, 3) | None

`plan_chunk` is optional; P6 calls it when a chunk lands if the solver has one
(`plans_chunks` True) or the run is waist mode in sync (`DeployConfig.wbik`),
and a None result means "run this chunk on the measured trunk" — the arm
planner uses the hook to plan its branch and commands no trunk. A factory is
`factory(cfg, *, urdf=None, qdot_max=…)` (the P6 arm clamp speed, so the solver
can hold it inside the QP).

Nothing heavy is imported here: `mode_of` answers for the four names from the
table below, and a factory module is imported only when a solver is built (or
a `module:factory` is asked its mode).
"""

from __future__ import annotations

#: What this package ships: name -> (`module:factory`, mode). The order is the
#: dashboard's (`schema.IK_OPTIONS`), and must not change without a schema
#: bump: the settings record stores an index into it.
SOLVERS = {
    "arm_planner": ("elrobot_deploy.runtime.ik.arm_planner:arm_planner", "arm"),
    "arm_qp": ("elrobot_deploy.runtime.ik.arm_qp:arm_qp", "arm"),
    "waist_planner": ("elrobot_deploy.runtime.ik.waist_planner:waist_planner", "waist"),
    "waist_qp": ("elrobot_deploy.runtime.ik.waist_qp:waist_qp", "waist"),
}

#: The solver each mode shorthand means. Changing an entry changes what every
#: deploy file that says `ik = "arm"` / `"waist"` (or nothing) runs.
DEFAULT = {"arm": "arm_planner", "waist": "waist_planner"}

#: The modes, in the order `resolved.IK_MODES` lists them.
MODES = ("arm", "waist")

#: The one solver `--intervene` runs: the takeover was built and tested against
#: the per-tick QP, and a planner holds a per-chunk plan (and an escape toward
#: it) that an operator's targets would be steered by (`validate.check_intervene_ik`).
INTERVENE = "arm_qp"


def normalize(ik: str) -> str:
    """A mode shorthand -> its default solver; anything else unchanged."""
    return DEFAULT.get(ik, ik)


def is_custom(ik: str) -> bool:
    """Is `ik` a `module:factory` rather than a shipped name?"""
    return normalize(ik) not in SOLVERS


def resolve(name: str):
    """-> the factory for `name` (a `SOLVERS` key, a mode shorthand, or
    `module:factory`). Raises `ValueError` naming the known solvers."""
    name = normalize(name)
    spec = SOLVERS[name][0] if name in SOLVERS else name
    if ":" not in spec:
        raise ValueError(f"unknown ik solver {name!r}; known: {list(SOLVERS)} "
                         f"(or give module:factory)")
    from elrobot_deploy.policy_contract.loading import load_attr
    try:
        factory = load_attr(spec)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"ik solver {name!r} -> {spec}: {exc}") from exc
    return factory


def mode_of(name: str) -> str:
    """"arm" or "waist" for solver `name`. Imports a `module:factory` to read
    its `.mode`; a factory without one is refused, since the mode decides what
    the run may command and what it is refused for."""
    name = normalize(name)
    if name in SOLVERS:
        return SOLVERS[name][1]
    mode = getattr(resolve(name), "mode", None)
    if mode not in MODES:
        raise ValueError(f"ik solver {name!r} has mode {mode!r}; a factory must "
                         f"carry .mode = \"arm\" or \"waist\"")
    return mode


def check(cfg) -> None:
    """Startup refusal: the solver resolves and names a mode (`mode_of`)."""
    resolve(cfg.ik)
    mode_of(cfg.ik)


def tcp_point(tcp):
    """The robot file's `[end_effector] tcp` (7,) — or None — as the
    TCP's position in the FLANGE frame, (3,) metres. The residuals the planners
    and `wbik` score with measure this point; None (an end effector with no
    TCP, e.g. BrainCo today) scores the flange origin itself."""
    import numpy as np
    if tcp is None:
        return np.zeros(3)
    return np.asarray(tcp, dtype=np.float64)[:3].copy()


def build(cfg, hw, name: str | None = None):
    """The solver P6 runs for this config — or, with `name`, that solver built
    exactly as P6 would build it (the IK bench, which must solve what flew).

    The four shipped factories take the robot file's TCP (`hw.tcp`); a research
    `module:factory` is called without it unless its signature accepts `tcp`,
    so a factory that takes no `tcp` works."""
    import inspect

    import numpy as np
    qdot = float(np.min(np.asarray(hw.qdot_max_body)[:14]))
    factory = resolve(cfg.ik if name is None else name)
    kw = {"urdf": hw.urdf, "qdot_max": qdot}
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):             # pragma: no cover - builtins
        params = {}
    if "tcp" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
        kw["tcp"] = getattr(hw, "tcp", None)
    return factory(cfg, **kw)


__all__ = ["DEFAULT", "INTERVENE", "MODES", "SOLVERS", "build", "check",
           "is_custom", "mode_of", "normalize", "resolve", "tcp_point"]
