"""`runtime/ik/__init__.py::build(cfg, hw, name=)`: the IK bench builds a solver exactly as P6 does
— with the robot file's TCP when the factory takes one — so a bench solves the
problem the rollout solved (it used to build without the TCP: flange-only)."""

from __future__ import annotations

import numpy as np

from g1d_deploy.runtime import ik

SEEN = {}


def recording_factory(cfg, *, urdf=None, qdot_max=None, tcp=None):
    SEEN.update(urdf=urdf, qdot_max=qdot_max, tcp=tcp)
    return object()


recording_factory.mode = "arm"


def test_a_named_solver_gets_the_tcp_p6_would_give_it(cfg, hw):
    ik.build(cfg, hw, name=f"{__name__}:recording_factory")
    assert SEEN["urdf"] == hw.urdf and SEEN["qdot_max"] > 0
    assert np.allclose(SEEN["tcp"], hw.tcp)
