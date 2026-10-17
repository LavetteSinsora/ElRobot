"""The IK P6 runs: a TCP pose in, 7 arm joints out, once per target tick.

P6 knows nothing about how a pose is solved. It holds one object with this interface, built from
the profile's `[ik]` table (`solver = NAME`, every other key passed to the factory):

    solve(pose7, seed_q7, budget) -> (q7, err_m, ok)
        `pose7` is the TCP's `[x y z qw qx qy qz]` in the base frame. `seed_q7` is the
        previous PUBLISHED target (never the measurement). `ok` False means one thing only:
        the wall-clock `budget` ran out, and P6 writes nothing for the tick. An unreachable pose
        is not a failure: return the best configuration and its residual `err_m` (metres).
    fk(q7) -> pose7
        Where the TCP is at `q7`: P6 anchors the operator's motion on it at each clutch press.
    set_nominal(q7)                                   (optional)
        Called at each clutch press with the configuration there, for a solver that leans its
        redundant DOF toward a nominal pose.

The solver is also expected to respect the joint limits and, if it can, the per-tick speed cap
it was built with; P6 clamps the speed again afterwards and the bus process clamps limits and
speed once more, so a solver that ignores them is safe, merely worse.

A factory is `factory(cfg, **options)`, where `cfg` is the run's `config_loading.Config` (the
URDF is `cfg.urdf`, the tick `cfg.dt`, the speed cap `cfg.qdot_max[:7]`). `solver` names a
shipped factory below, or any `module:factory` (`package.module:name` or `path/file.py:name`),
which is how a new IK strategy is tried without touching P6:

    elrobot_deploy run --set ik.solver=my_ik.py:build --set ik.damping=0.01

The MuJoCo/mink stack is imported only by a factory, so nothing but P6 (and a test) loads it.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Protocol

import numpy as np

#: What this package ships: name -> `module:factory`.
SOLVERS = {
    "tcp_qp": "elrobot_deploy.runtime.ik.tcp_qp:factory",
}


class IKSolver(Protocol):
    def solve(self, pose7, seed_q7, budget: float | None = None
              ) -> tuple[np.ndarray, float, bool]: ...

    def fk(self, q7) -> np.ndarray: ...


def load_attr(spec: str):
    """`package.module:name` or `path/to/file.py:name` -> the object."""
    module, sep, name = spec.rpartition(":")
    if not sep or not module or not name:
        raise ValueError(f"{spec!r} is not module:name")
    if module.endswith(".py"):
        path = Path(module).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module)
    return getattr(mod, name)


def resolve(name: str):
    """A shipped solver's name, or `module:factory` -> the factory."""
    spec = SOLVERS.get(name, name)
    if ":" not in spec:
        raise ValueError(f"unknown ik solver {name!r}; shipped: {list(SOLVERS)} "
                         f"(or give module:factory)")
    try:
        factory = load_attr(spec)
    except (ImportError, AttributeError, OSError) as exc:
        raise ValueError(f"ik solver {name!r} -> {spec}: {exc}") from exc
    if not callable(factory):
        raise ValueError(f"ik solver {name!r} -> {spec} is not callable")
    return factory


def build(cfg) -> IKSolver:
    """The solver the profile names, built as P6 builds it."""
    solver = resolve(cfg.ik.solver)(cfg, **cfg.ik.options)
    for method in ("solve", "fk"):
        if not callable(getattr(solver, method, None)):
            raise ValueError(f"ik solver {cfg.ik.solver!r} has no {method}() "
                             f"(runtime/ik/__init__.py says what P6 calls)")
    return solver


__all__ = ["SOLVERS", "IKSolver", "build", "load_attr", "resolve"]
