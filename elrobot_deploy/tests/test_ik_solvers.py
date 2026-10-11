"""`ik = NAME`: the solver object P6 runs, by name (`g1d_deploy.runtime.ik`).

What must hold: exactly four shipped names, none reaching outside g1d_deploy,
with `arm_planner` / `waist_planner` the defaults of their modes; an unknown
name, or a `module:factory` without a mode, is refused at startup and never
reaches a process; the mode comes from the solver (so a waist solver is held to
sync); a plug-in that plans per chunk is asked once per chunk in arm mode and a
None plan leaves the trunk measured; and a solver that does not plan leaves the
hook to `cfg.wbik`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from g1d_deploy.config_loading import resolved as cfgmod
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading.validate import validate_rollout
from g1d_deploy.runtime import ik as ik_solvers
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.processes.p6_target import Target

from .test_p6_target import FakeClock, FakeSolver, publish

ARM, BODY = schema.ARM_DOF, schema.BODY_DOF


class ArmPlanner(FakeSolver):
    """An arm-mode plug-in: plans its own thing per chunk, commands no trunk."""

    mode = "arm"
    plans_chunks = True

    def __init__(self, **kw):
        super().__init__(**kw)
        self.plans = []

    def plan_chunk(self, poses, waist_meas, seed_q14, *, budget=None):
        self.plans.append((len(poses), np.array(waist_meas), np.array(seed_q14)))
        return None


def factory_arm(cfg, *, urdf=None, qdot_max=None):
    return ArmPlanner()


factory_arm.mode = "arm"


def factory_waist(cfg, *, urdf=None, qdot_max=None):
    return FakeSolver()


factory_waist.mode = "waist"


@pytest.fixture
def clk():
    return FakeClock()


def modeless(cfg, *, urdf=None, qdot_max=None):
    return FakeSolver()


def test_an_unknown_name_is_refused(cfg, hw):
    with pytest.raises(ValueError, match="unknown ik solver"):
        ik_solvers.check(dataclasses.replace(cfg, ik="no_such_solver"))
    with pytest.raises(ConfigError, match="no_such_solver"):
        validate_rollout(dataclasses.replace(cfg, ik="no_such_solver"), hw)


def test_a_module_solvers_mode_is_its_own_and_must_exist(cfg, hw):
    """The mode is read off the factory, so a waist plug-in is a waist run —
    held to sync like the shipped waist solvers — and a factory that does not
    say which joints it moves is refused rather than guessed."""
    arm = dataclasses.replace(cfg, ik=f"{__name__}:factory_arm")
    waist = dataclasses.replace(cfg, ik=f"{__name__}:factory_waist")
    ik_solvers.check(arm)
    assert (arm.ik_mode, waist.ik_mode) == ("arm", "waist")
    assert waist.wbik and not arm.wbik
    with pytest.raises(ConfigError, match="sync = false"):
        validate_rollout(dataclasses.replace(waist, sync=False), hw)
    with pytest.raises(ValueError, match="mode"):
        ik_solvers.check(dataclasses.replace(cfg, ik=f"{__name__}:modeless"))


def test_every_registered_name_resolves_to_a_factory_with_its_mode():
    pytest.importorskip("mujoco")
    for name, (_, mode) in ik_solvers.SOLVERS.items():
        f = ik_solvers.resolve(name)
        assert callable(f) and f.mode == mode, name


def test_an_arm_plug_in_is_asked_once_per_chunk_and_the_trunk_stays_measured(
        rings, hw, cfg, clk):
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    rings["control"].k_exec = 10
    publish(rings, 1, k_first=11, n=8)
    solver = ArmPlanner()
    loop = Target(rings, cfg, hw, solver, now=clk)
    assert loop.plans_chunks and not loop.wbik
    for _ in range(4):
        loop.step()
    assert len(solver.plans) == 1 and solver.plans[0][0] == 7     # k_exec+2 .. k_first+n-1
    assert loop._plan is None
    assert rings["targets"].get(12).payload[ARM:][1] == pytest.approx(0.0)
    assert len(solver.calls) >= 1


def test_a_solver_that_does_not_plan_leaves_the_hook_to_the_wbik_flag(
        rings, hw, cfg, clk):
    rings["targets"].write(0, clk.t, np.zeros(BODY), intervene=0)
    loop = Target(rings, cfg, hw, FakeSolver(), now=clk)
    assert loop.plans_chunks == loop.wbik == False   # noqa: E712


# --- The four solvers this package ships ------------------------------------


def test_exactly_four_shipped_solvers_all_inside_g1d_deploy():
    """The set is the user's decision (2026-09-26): two per mode, a planner
    and the original QP, in the dashboard's order. No registry of research
    names — those come in as `module:factory`."""
    assert list(ik_solvers.SOLVERS) == ["arm_planner", "arm_qp",
                                        "waist_planner", "waist_qp"]
    assert tuple(ik_solvers.SOLVERS) == schema.IK_OPTIONS == cfgmod.IK_SOLVERS
    for name, (spec, mode) in ik_solvers.SOLVERS.items():
        assert spec.startswith("g1d_deploy.runtime.ik"), spec
        assert ik_solvers.mode_of(name) == mode
    assert not hasattr(ik_solvers, "BENCH")


def test_the_defaults_are_the_planners_and_the_shorthands_name_them(cfg):
    """`ik = "arm"` (and no `ik` at all) runs `arm_planner`, not the per-tick
    QP (`arm_qp`), and `ik = "waist"` runs `waist_planner`, not the whole-body
    QP (`waist_qp`)."""
    assert cfgmod.DeployConfig.ik == "arm_planner"
    assert cfg.ik == "arm_planner" and cfg.ik_mode == "arm" and not cfg.wbik
    assert dataclasses.replace(cfg, ik="arm").ik == "arm_planner"
    assert dataclasses.replace(cfg, ik="waist").ik == "waist_planner"
    assert ik_solvers.DEFAULT == {"arm": "arm_planner", "waist": "waist_planner"}
    assert dataclasses.replace(cfg, ik="waist_qp").wbik
    assert not dataclasses.replace(cfg, ik="waist_qp", sync=False).wbik


def test_the_qp_names_build_the_original_solvers(cfg, hw):
    """`arm_qp` / `waist_qp` are the pre-solver-set P6 paths, unchanged:
    `DualArmSolver(cfg, urdf)` and `WholeBodySolver(..., TIERS["waist"])`."""
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from g1d_deploy.runtime.ik.arm_qp import DualArmSolver
    from g1d_deploy.runtime.ik.waist_qp import TIERS, WholeBodySolver

    arm = ik_solvers.build(dataclasses.replace(cfg, ik="arm_qp"), hw)
    assert type(arm) is DualArmSolver
    assert arm.rel is None and arm.max_iters == cfg.ik_max_iters
    waist = ik_solvers.build(dataclasses.replace(cfg, ik="waist_qp"), hw)
    assert type(waist) is WholeBodySolver
    assert set(waist.groups) == set(TIERS["waist"])


def test_the_deployed_solver_is_unchanged_by_default():
    """The convergence stop the planners use is opt-in: a `DualArmSolver` built
    the way P6 builds it has it OFF and keeps `cfg.ik_max_iters`."""
    from g1d_deploy.runtime.ik import arm_qp as ik

    class Cfg:
        dt = 1 / 30.0
        ik_early_stop = 1e-4
        ik_max_iters = 17
        ik_posture_cost = 1e-3
        ik_collision_min_dist = 0.005

    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    s = ik.DualArmSolver(Cfg())
    assert s.rel is None and s.max_iters == 17
    assert ik.DualArmSolver(Cfg(), max_iters=3, rel=0.05).max_iters == 3


def test_the_tick_anchor_caps_only_inside_a_tick():
    from g1d_deploy.runtime.ik.velocity_caps import TickAnchor

    a = TickAnchor(1 / 30.0)
    assert not a.active
    with a.ticking(np.zeros(14)):
        assert a.active and a.q.shape == (14,)
        with a.suspended():
            assert not a.active, "plan-time solves inside a tick are uncapped"
        assert a.active
        with a.transiting(3.0):
            assert a.transit and a.transit_speed == 3.0
        assert not a.transit and a.transit_speed is None
    assert not a.active


def test_the_trunk_servo_model_moves_a_bounded_fraction_per_tick():
    from g1d_deploy.runtime.ik.waist_planner import TRUNK_GAIN, TRUNK_VMAX, trunk_servo

    dt = 1 / 30.0
    cmd = np.zeros((4, 3))
    cmd[:, 1] = 0.01                                   # a small bow, held
    out = trunk_servo(cmd, np.arange(4), np.zeros(3), dt)
    assert out[0, 1] == 0.0
    assert np.isclose(out[1, 1], TRUNK_GAIN * 0.01)   # 0.7 of the way
    cmd[:, 1] = 1.0                                    # a bow the servo cannot make in a tick
    out = trunk_servo(cmd, np.arange(4), np.zeros(3), dt)
    assert np.isclose(out[1, 1], TRUNK_VMAX * dt)      # rate-limited


def test_the_slewed_trunk_holds_the_last_ticks_of_a_chunk():
    from g1d_deploy.runtime.ik.waist_planner import SlewTrunk

    slew = SlewTrunk(bend=0.2, unbend=0.1, yaw=0.3, settle=6)
    traj = np.zeros((25, 3))
    traj[:, 1] = np.linspace(0.0, 0.5, 25)             # asks to bow 0.5 rad in a chunk
    out = slew(traj, np.zeros(3), 1 / 30.0)
    assert np.all(np.diff(out[:19, 1]) <= 0.2 / 30 + 1e-12), "bend rate-limited"
    assert np.allclose(out[19:, 1], out[18, 1]), "the last 6 ticks hold"
    assert np.allclose(slew.last, out[-1])


# --- The robot file's TCP reaches the solvers ----------------------------------


def test_build_hands_the_robot_files_tcp_to_the_planner(cfg, hw):
    """`hw.tcp` (the robot file's `[end_effector] tcp`) is the point the
    planner's residuals measure — not the hard-coded Dex-1 constant."""
    tcp = np.array([0.25, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0])
    solver = ik_solvers.build(cfgmod.replace(cfg, ik="arm_planner"),
                              dataclasses.replace(hw, tcp=tcp))
    assert np.allclose(solver.inner.residuals.tcp, tcp[:3])


def test_no_tcp_scores_the_flange_origin():
    assert np.array_equal(ik_solvers.tcp_point(None), np.zeros(3))


def test_every_shipped_solver_drives_the_robot_files_tcp(cfg, hw):
    """With a `tcp`, every QP's task frame is the model's TCP site; without,
    the flange."""
    tcp = np.array([0.2, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
    with_tcp = dataclasses.replace(hw, tcp=tcp)
    qp = ik_solvers.build(cfgmod.replace(cfg, ik="arm_qp"), with_tcp)
    assert np.allclose(qp.tcp, tcp)
    assert qp.ik.tasks["left"].frame_name == "left_tcp_site"
    wq = ik_solvers.build(cfgmod.replace(cfg, ik="waist_qp"), with_tcp)
    assert np.allclose(wq.tcp, tcp)
    assert wq.ik.tasks["right"].frame_name == "right_tcp_site"
    pl = ik_solvers.build(cfgmod.replace(cfg, ik="arm_planner"), with_tcp)
    assert all(s.tcp is not None for s in pl.inner.solvers)
    bare = ik_solvers.build(cfgmod.replace(cfg, ik="arm_qp"),
                            dataclasses.replace(hw, tcp=None))
    assert bare.tcp is None
    assert bare.ik.tasks["left"].frame_name == "left_wrist_yaw_link"


def _reachable_pose(solver, q14):
    """The flange poses (base frame) of a known arm configuration."""
    from g1d_deploy.robot import kinematics as kin
    solver.place(q14, np.zeros(3))
    inv = np.linalg.inv(solver.base)
    return np.stack([kin.mat_to_pose(inv @ solver.backend.flange_pose(s))
                     for s in ("left", "right")])


@pytest.mark.parametrize("tcp", [None, [0.184759, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
def test_a_solve_lands_the_driven_point(cfg, tcp):
    """A perfect solve is the same either way: the flange lands on its target,
    and with a TCP the TCP lands on the flange target composed with it. The
    returned error is at the driven point."""
    from g1d_deploy.robot import kinematics as kin
    from g1d_deploy.runtime.ik.arm_qp import DualArmSolver

    c = cfgmod.replace(cfg, ik_max_iters=200)
    s = DualArmSolver(c, tcp=tcp, max_iters=200)
    q_true = np.array([0.2, 0.3, 0.1, 0.9, 0.1, 0.1, 0.0,
                       0.2, -0.3, -0.1, 0.9, -0.1, 0.1, 0.0])
    pose = _reachable_pose(s, q_true)
    seed = q_true + 0.05
    _q, err, ok = s.solve(pose, np.zeros(3), seed, budget=5.0)
    assert ok and err < 2e-3
    for h, side in enumerate(("left", "right")):
        want = s.base @ kin.pose_to_mat(pose[h])
        if tcp is None:
            got = s.backend.flange_pose(side)
        else:
            want = want @ kin.pose_to_mat(np.asarray(tcp))
            got = s.backend.ik_pose(side)
            # the site IS the flange composed with the tcp
            assert np.allclose(got, s.backend.flange_pose(side)
                               @ kin.pose_to_mat(np.asarray(tcp)), atol=1e-9)
        assert np.linalg.norm(got[:3, 3] - want[:3, 3]) < 2e-3


def test_a_miss_is_traded_at_the_tcp_not_the_flange(cfg):
    """Out of reach, the two rules miss differently: the TCP-driven solve
    leaves the TCP nearer its target than the flange-driven one does."""
    from g1d_deploy.robot import kinematics as kin
    from g1d_deploy.runtime.ik.arm_qp import DualArmSolver

    tcp = np.array([0.184759, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    X = kin.pose_to_mat(tcp)
    c = cfgmod.replace(cfg, ik_max_iters=100)
    flange = DualArmSolver(c, max_iters=100)
    at_tcp = DualArmSolver(c, tcp=tcp, max_iters=100)
    q0 = np.array([0.2, 0.3, 0.1, 0.9, 0.1, 0.1, 0.0,
                   0.2, -0.3, -0.1, 0.9, -0.1, 0.1, 0.0])
    pose = _reachable_pose(flange, q0)
    pose[:, 0] += 0.6                        # far past reach, straight ahead
    tcp_miss = []
    for s in (flange, at_tcp):
        s.solve(pose, np.zeros(3), q0, budget=5.0)
        miss = 0.0
        for h, side in enumerate(("left", "right")):
            want = s.base @ kin.pose_to_mat(pose[h]) @ X
            got = s.backend.flange_pose(side) @ X
            miss = max(miss, float(np.linalg.norm(got[:3, 3] - want[:3, 3])))
        tcp_miss.append(miss)
    assert tcp_miss[1] < tcp_miss[0]


def test_a_research_factory_without_tcp_still_builds(cfg, hw, monkeypatch):
    seen = {}

    def research(cfg, *, urdf=None, qdot_max=None):
        seen["called"] = True
        return object()

    research.mode = "arm"
    monkeypatch.setattr(ik_solvers, "resolve", lambda name: research)
    ik_solvers.build(cfg, hw)
    assert seen == {"called": True}
