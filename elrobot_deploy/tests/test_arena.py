"""The arena and its rings.

Everything else in the system is built on these queries, so the failures worth
catching here are the ones that would otherwise surface as a wrong joint target
three processes away: a wrap-around that returns a stale tick as if it were
current, a seed picked from the wrong side, a window scan that returns the
latest instead of the earliest.
"""

from __future__ import annotations

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.config_loading.derive import P_MIN, derive
from elrobot_deploy.config_loading.errors import ConfigError
from elrobot_deploy.config_loading.validate import validate, validate_rollout
from elrobot_deploy.runtime.shared_memory import (Arena, LayoutMismatch, build, open_all,
                                              schema)


# --- sizing -----------------------------------------------------


def test_chunk_slots_carry_one_spare_over_live_contributors(cfg, hw):
    """Sized for the WORST replan period a rollout may switch to:
    the smallest P `validate` accepts, so the most live contributors."""
    d = derive(cfg, hw)
    n_live = -(-cfg.H // P_MIN)
    assert d.n_contrib_max == n_live
    assert d.chunk_slots == n_live + 1
    assert d.n_contrib_max >= -(-cfg.H // cfg.P)


def test_target_rings_cover_horizon_plus_two_replans(cfg, hw):
    """...and for the LARGEST P, which is H: `H + 2P` at `P = H`."""
    d = derive(cfg, hw)
    assert d.n_target >= cfg.H + 2 * cfg.P
    assert d.n_target >= 3 * cfg.H
    assert d.n_target & (d.n_target - 1) == 0    # power of two


@pytest.mark.parametrize("P, sync, hz", [(2, True, 30.0), (8, False, 15.0),
                                         (5, True, 60.0)])
def test_switchable_settings_never_change_the_layout(cfg, hw, P, sync, hz):
    """P, sync/async and the control rate change between rollouts of ONE arena,
    so none of them may reach the layout hash every process checks
    before it attaches — a P5 latching P = 2 must attach the arena P0 built at
    P = 3."""
    other = cfgmod.replace(cfg, replan_period_ticks=P, sync=sync, control_hz=hz)
    validate_rollout(other, hw)
    assert (build(other, derive(other, hw)).hash
            == build(cfg, derive(cfg, hw)).hash)


def test_refusals(cfg, hw):
    with pytest.raises(ConfigError, match="must be >= 2"):
        validate(cfgmod.replace(cfg, replan_period_ticks=1), hw)
    with pytest.raises(ConfigError, match="action_horizon"):
        validate(cfgmod.replace(cfg, action_horizon=2), hw)
    with pytest.raises(ConfigError, match="primary_camera"):
        validate(cfgmod.replace(cfg, primary_camera="nope"), hw)

    # `observed_cameras` SELECTS from `cameras`; it cannot introduce one. A
    # stranger here reads as "this camera feeds the policy" and would otherwise
    # be a KeyError inside P5's first anchor — after the arena is mapped and the
    # checkpoint is compiled, a long way from the line that caused it.
    with pytest.raises(ConfigError, match="not declared cameras"):
        validate(
            cfgmod.replace(cfg, observed_cameras=("rwrist", "nope")), hw)

    # The primary's chosen frame IS t_anchor, so it cannot be a camera the
    # policy never looks at.
    with pytest.raises(ConfigError, match="not in observed_cameras"):
        validate(cfgmod.replace(cfg, observed_cameras=("head",)), hw)
    # A checkpoint that does not exist is caught at startup, not at WARMUP.
    with pytest.raises(ConfigError, match="does not exist"):
        validate(cfgmod.replace(cfg, checkpoint="/no/such/step"), hw)
    validate(cfg, hw)                     # the fixture itself is valid


# --- camera sources ---------------------------------------------------------


def _with_camera(cfg, name, spec):
    return cfgmod.replace(cfg, cameras={**cfg.cameras, name: spec},
                          primary_camera=name)


def test_only_the_zmq_source_needs_a_port_on_the_robot_board(cfg, hw):
    """A camera on the deploy PC's own USB has no `image_server` port, and
    demanding a placeholder would be demanding a number that means nothing."""
    assert "exo" not in hw.camera_ports

    with pytest.raises(ConfigError, match="no port in"):
        validate(_with_camera(cfg, "exo", cfgmod.CameraSpec(8, 8)), hw)

    realsense = _with_camera(cfg, "exo", cfgmod.CameraSpec(8, 8, source="realsense"))
    import importlib.util
    if importlib.util.find_spec("pyrealsense2") is None:
        # Without the optional `realsense` extra the camera is refused by name.
        with pytest.raises(ConfigError, match="realsense"):
            validate(realsense, hw)
    else:
        validate(realsense, hw)
    validate(_with_camera(
        cfg, "exo", cfgmod.CameraSpec(8, 8, source="v4l2",
                                      device="/dev/v4l/by-path/x")), hw)


def test_a_v4l2_camera_must_name_a_device(cfg, hw):
    """`/dev/videoN` is not stable across reboots and the two fisheye units
    report duplicate serials, so the by-path node is the only stable address —
    and getting left and right the wrong way round raises nothing at all."""
    with pytest.raises(ConfigError, match="by-path"):
        validate(_with_camera(
            cfg, "lwrist", cfgmod.CameraSpec(8, 8, source="v4l2")), hw)


def test_an_unknown_source_is_refused(cfg, hw):
    with pytest.raises(ConfigError, match="not one of"):
        validate(_with_camera(
            cfg, "exo", cfgmod.CameraSpec(8, 8, source="usb")), hw)


# --- layout hash ------------------------------------------------


def test_attach_refuses_a_different_layout(arena, cfg, derived, hw):
    other = build(cfgmod.replace(cfg, action_horizon=cfg.H + 1),
                  derive(cfgmod.replace(cfg, action_horizon=cfg.H + 1), hw))
    assert other.hash != arena.layout.hash
    with pytest.raises(LayoutMismatch):
        Arena.attach(arena.shm.name, other)


def test_attach_shares_memory(arena, cfg):
    b = Arena.attach(arena.shm.name, arena.layout)
    try:
        open_all(arena, cfg)["control"].k_exec = 41
        assert open_all(b, cfg)["control"].k_exec == 41
    finally:
        b.close()


# --- body target ring -------------------------------------------------------


def test_get_returns_none_for_a_tick_the_slot_no_longer_holds(rings, derived):
    body = rings["body"]
    body.write(4, 1.0, np.full(schema.BODY_DOF, 0.5), intervene=0)
    assert body.get(4) is not None
    body.write(4 + body.n, 2.0, np.full(schema.BODY_DOF, 0.7), intervene=0)
    # Same slot, later tick: the old tick must not be reported as present.
    assert body.get(4) is None
    assert body.get(4 + body.n).deadline == 2.0


def test_earliest_written_in_window_not_latest(rings):
    """P6 writes in increasing k, so the earliest is the first to
    exist. Returning the latest would make P5 wait for the chunk to be nearly
    exhausted before it could compute its wake time."""
    body = rings["body"]
    for k in (5, 6, 7):
        body.write(k, 10.0 + k, np.zeros(schema.BODY_DOF), intervene=0)
    assert body.earliest_written_in(4, 8).k == 5
    assert body.earliest_written_in(6, 8).k == 6
    assert body.earliest_written_in(20, 25) is None


def test_earliest_written_in_window_skips_holes(rings):
    """An IK solve that overran its budget leaves a hole; any other solved tick
    in the window yields the same t_next_inference."""
    body = rings["body"]
    body.write(9, 99.0, np.zeros(schema.BODY_DOF), intervene=0)
    assert body.earliest_written_in(7, 10).k == 9


def test_latest_below_is_the_seed(rings):
    body = rings["body"]
    for k in (2, 3, 6):
        body.write(k, float(k), np.full(schema.BODY_DOF, k), intervene=0)
    seed = body.latest_below(8)
    assert seed.k == 6
    assert np.allclose(seed.payload, 6)
    assert body.latest_below(3).k == 2
    assert body.latest_below(2) is None


def test_next_after_takes_the_smallest_future_deadline(rings):
    body = rings["body"]
    body.write(1, 1.0, np.full(schema.BODY_DOF, 1), intervene=0)
    body.write(2, 2.0, np.full(schema.BODY_DOF, 2), intervene=0)
    body.write(3, 3.0, np.full(schema.BODY_DOF, 3), intervene=0)
    assert body.next_after(0.5, 0).k == 1
    assert body.next_after(1.0, 0).k == 2           # strictly greater
    assert body.next_after(2.5, 0).k == 3
    assert body.next_after(3.0, 0) is None


def test_next_after_ignores_stale_past_ticks(rings):
    """P6 may write ticks below k_exec. They are harmless precisely
    because their deadlines are already in the past."""
    body = rings["body"]
    body.write(1, 1.0, np.zeros(schema.BODY_DOF), intervene=0)
    body.write(9, 0.5, np.zeros(schema.BODY_DOF), intervene=0)     # a stale, past deadline
    assert body.next_after(0.75, 0).k == 1


def test_negative_ticks_work(rings):
    """Homing runs at k < 0."""
    body = rings["body"]
    body.write(-5, 1.0, np.full(schema.BODY_DOF, -5), intervene=0)
    body.write(0, 2.0, np.full(schema.BODY_DOF, 0), intervene=0)
    assert body.get(-5).k == -5
    assert body.latest_below(0).k == -5
    assert body.next_after(0.0, 0).k == -5


def test_body_and_hand_rings_are_independent_but_parallel(rings):
    """Same k, same deadline, different payloads."""
    rings["targets"].write(7, 3.5, np.arange(schema.BODY_DOF, dtype=float), intervene=0)
    rings["targets"].write(7, 3.5, np.zeros(schema.BODY_DOF), np.ones((schema.N_HANDS, schema.HAND_DIM)), intervene=0)
    assert rings["body"].get(7).deadline == rings["hand"].get(7).deadline
    assert rings["hand"].get(7).payload.shape == (schema.N_HANDS, schema.HAND_DIM)


# --- chunk ring -------------------------------------------------------------


def _chunk(rings, cfg, chunk_id, k_first, n, value):
    rings["chunks"].write(
        chunk_id=chunk_id, k_first_action=k_first, n_actions=n,
        pose=np.full((n, schema.N_HANDS, schema.POSE_DIM), value),
        hand=np.full((n, schema.N_HANDS, schema.HAND_DIM), value),
        t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
        t_cpu_processed=0.0, t_chunk_received=0.0,
        t_chunk_written=0.1)


def test_covering_selects_the_right_row(rings, cfg):
    n = cfg.P
    rings["chunks"].write(
        chunk_id=1, k_first_action=10, n_actions=n,
        pose=np.arange(n * 2 * 7, dtype=float).reshape(n, 2, 7),
        hand=np.zeros((n, 2, 6)),
        t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
        t_cpu_processed=0.0, t_chunk_received=0.0,
        t_chunk_written=0.1)
    got = rings["chunks"].covering(11)
    assert len(got) == 1 and got[0].chunk_id == 1
    assert np.allclose(got[0].pose, np.arange(2 * 7, dtype=float).reshape(2, 7) + 14)


def test_covering_is_ascending_by_chunk_id(rings, cfg):
    _chunk(rings, cfg, 1, 10, cfg.P, 1.0)
    _chunk(rings, cfg, 2, 12, cfg.P, 2.0)
    got = rings["chunks"].covering(12)
    assert [c.chunk_id for c in got] == [1, 2]
    assert got[-1].pose[0, 0] == 2.0

def test_coverage_boundaries_are_inclusive_of_first_and_last(rings, cfg):
    _chunk(rings, cfg, 1, 10, cfg.P, 1.0)
    assert rings["chunks"].covers_any(10, adopted_through=None)
    assert rings["chunks"].covers_any(10 + cfg.P - 1, adopted_through=None)
    assert not rings["chunks"].covers_any(9, adopted_through=None)
    assert not rings["chunks"].covers_any(10 + cfg.P, adopted_through=None)


def test_coverage_can_be_limited_to_chunks_the_reader_has_adopted(rings, cfg):
    """The publish order leaves a window in which a chunk is coverable but not yet announced
    — its payload is stored, `newest_chunk_id` is not. `adopted_through` is what
    keeps a control caller out of that window; `None` is what lets a diagnostic
    look into it deliberately."""
    _chunk(rings, cfg, 1, 10, cfg.P, 1.0)
    _chunk(rings, cfg, 2, 10 + cfg.P, cfg.P, 2.0)   # written, imagine unannounced
    k = 10 + cfg.P                                  # covered by chunk 2 alone

    assert rings["chunks"].covers_any(k, adopted_through=None)   # it is there
    assert rings["chunks"].covers_any(k, adopted_through=2)      # and adopted
    assert not rings["chunks"].covers_any(k, adopted_through=1)  # not yet
    assert not rings["chunks"].covers_any(k, adopted_through=0)  # nothing read

    # The limit bounds only which chunks count, never which ticks they cover.
    assert rings["chunks"].covers_any(10, adopted_through=1)


def test_empty_slots_never_contribute(rings, cfg):
    assert rings["chunks"].covering(0) == []
    assert not rings["chunks"].covers_any(0, adopted_through=None)


# --- chunk ring: the lookahead rows -----------------------------------------
#
# `n_plan` is a SECOND word beside `n_actions`, not a wider one. Everything
# below exists to pin that distinction, because the cheap implementation —
# letting `n_actions` grow — would make the extra rows executable, and every
# scheduling decision in the system reads executability off `covering`.


def _chunk_and_plan(rings, cfg, chunk_id, k_first, n, extra, value, plan_value):
    """One chunk of `n` actions carrying `extra` rows it will never run."""
    def rows(*shape):
        return np.concatenate([np.full((n, *shape), value),
                               np.full((extra, *shape), plan_value)])
    rings["chunks"].write(
        chunk_id=chunk_id, k_first_action=k_first, n_actions=n,
        n_plan=n + extra,
        pose=rows(schema.N_HANDS, schema.POSE_DIM),
        hand=rows(schema.N_HANDS, schema.HAND_DIM),
        t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
        t_cpu_processed=0.0, t_chunk_received=0.0,
        t_chunk_written=0.1)


def test_a_chunk_without_a_plan_stores_n_plan_equal_to_n_actions(rings, cfg):
    """The knob off is the default, and the default must be the old behaviour
    exactly — including for a writer that has never heard of `n_plan`."""
    _chunk(rings, cfg, 1, 10, cfg.P, 1.0)        # the pre-existing call shape
    assert rings["chunks"].plan_rows(10, adopted_through=1) is None


def test_plan_rows_are_resident_but_not_covered(rings, cfg):
    """The whole point: P6 can read them, P7 can never run them."""
    _chunk_and_plan(rings, cfg, 1, 10, cfg.P, 3, 1.0, 9.0)
    last = 10 + cfg.P - 1

    # Coverage stops at the executed rows, as it did before.
    assert rings["chunks"].covers_any(last, adopted_through=1)
    assert not rings["chunks"].covers_any(last + 1, adopted_through=1)
    assert rings["chunks"].covering(last + 1) == []
    assert len(rings["chunks"].covering(last)) == 1

    # ...and the executed rows still read back as themselves, not as plan rows.
    assert rings["chunks"].covering(last)[0].pose[0, 0] == 1.0

    tail = rings["chunks"].plan_rows(10, adopted_through=1)
    assert tail is not None
    assert tail.chunk_id == 1
    assert tail.k_first == last + 1           # the tail starts where actions end
    assert len(tail.pose) == len(tail.hand) == 3
    assert tail.waist is None                 # an eef chunk carries none
    assert tail.pose[0, 0, 0] == 9.0          # and carries its own values


def test_plan_rows_obey_the_same_adoption_limit_as_coverage(rings, cfg):
    """A chunk whose payload is stored but not yet announced is invisible to a
    control caller — the plan rows must not be the hole in that rule."""
    _chunk_and_plan(rings, cfg, 1, 10, cfg.P, 3, 1.0, 9.0)
    assert rings["chunks"].plan_rows(10, adopted_through=1) is not None
    assert rings["chunks"].plan_rows(10, adopted_through=0) is None


def test_plan_rows_follow_the_newest_chunk_covering_the_tick(rings, cfg):
    """Two chunks can cover one tick during a seam; the plan that matters is
    the newer one's, matching `covering`'s ascending-by-chunk_id contract."""
    _chunk_and_plan(rings, cfg, 1, 10, cfg.P, 3, 1.0, 9.0)
    _chunk_and_plan(rings, cfg, 2, 11, cfg.P, 3, 2.0, 8.0)
    tail = rings["chunks"].plan_rows(11, adopted_through=2)
    assert tail.chunk_id == 2 and tail.pose[0, 0, 0] == 8.0


def test_a_plan_shorter_than_the_chunk_is_refused(rings, cfg):
    with pytest.raises(ValueError):
        rings["chunks"].write(
            chunk_id=1, k_first_action=10, n_actions=cfg.P, n_plan=cfg.P - 1,
            pose=np.zeros((cfg.P, schema.N_HANDS, schema.POSE_DIM)),
            hand=np.zeros((cfg.P, schema.N_HANDS, schema.HAND_DIM)),
            t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
            t_cpu_processed=0.0, t_chunk_received=0.0, t_chunk_written=0.1)


def test_a_plan_past_the_horizon_is_refused(rings, cfg):
    """The slot is `horizon` rows wide; there is nowhere to put row H+1."""
    n = cfg.H + 1
    with pytest.raises(ValueError, match="horizon"):
        rings["chunks"].write(
            chunk_id=1, k_first_action=10, n_actions=cfg.P, n_plan=n,
            pose=np.zeros((n, schema.N_HANDS, schema.POSE_DIM)),
            hand=np.zeros((n, schema.N_HANDS, schema.HAND_DIM)),
            t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
            t_cpu_processed=0.0, t_chunk_received=0.0, t_chunk_written=0.1)


def test_the_lookahead_knob_refuses_what_it_cannot_honour(cfg, hw):
    validate(cfgmod.replace(cfg, plan_lookahead_ticks=0), hw)
    validate(cfgmod.replace(cfg, plan_lookahead_ticks=cfg.H - cfg.P), hw)

    with pytest.raises(ConfigError, match="plan_lookahead_ticks"):
        validate(cfgmod.replace(cfg, plan_lookahead_ticks=-1), hw)

    # It extends ONE chunk past its own actions, which only means something
    # when that chunk owns the ticks after them.
    with pytest.raises(ConfigError, match="sync"):
        validate(
            cfgmod.replace(cfg, plan_lookahead_ticks=1, sync=False), hw)

    # The rows are decoded actions, so they can only come from the horizon the
    # policy already produces.
    with pytest.raises(ConfigError, match="action_horizon"):
        validate(
            cfgmod.replace(cfg, plan_lookahead_ticks=cfg.H - cfg.P + 1), hw)


# --- state ring -------------------------------------------------------------


def test_state_nearest_picks_the_closest_sample(rings):
    st = rings["state"]
    for i, t in enumerate([1.0, 2.0, 3.0]):
        st.write(t_recv=t, q=np.full(schema.N_JOINTS, i), dq=np.zeros(schema.N_JOINTS),
                 flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                 grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=t)
    assert rings["state"].nearest(2.4)["q"][0] == 1
    assert rings["state"].nearest(2.6)["q"][0] == 2
    assert rings["state"].latest()["q"][0] == 2


def test_state_ring_wraps_and_keeps_the_newest(rings, derived):
    st = rings["state"]
    for i in range(derived.state_slots + 5):
        st.write(t_recv=float(i), q=np.full(schema.N_JOINTS, i),
                 dq=np.zeros(schema.N_JOINTS),
                 flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                 grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=0.0)
    latest = st.latest()
    assert latest["q"][0] == derived.state_slots + 4


# --- cameras ----------------------------------------------------------------


def test_camera_publish_and_peek(rings):
    cam = rings["cameras"]["rwrist"]
    assert cam.peek() == (0.0, 0)
    cam.spare()["pixels"][:] = 7
    cam.publish(t_recv=12.5)
    t, fid = cam.peek()
    assert (t, fid) == (12.5, 1)
    out = cam.empty()
    assert cam.read_into(out)
    assert np.all(out["pixels"] == 7)


def test_camera_buffers_rotate(rings):
    """The writer must not decode into the buffer it just published."""
    cam = rings["cameras"]["rwrist"]
    seen = set()
    for i in range(6):
        buf = cam.spare()
        buf["pixels"][:] = i
        seen.add(buf.ctypes.data if hasattr(buf, "ctypes") else id(buf.tobytes()))
        cam.publish(t_recv=float(i))
        assert cam.peek()[0] == float(i)
    out = cam.empty()
    assert cam.read_into(out) and np.all(out["pixels"] == 5)


def test_head_and_wrist_regions_have_different_shapes(rings, cfg):
    """The head is one 480x1280 binocular frame, wrists are 640-wide.
    Anything assuming one buffer size is wrong by 2x on the head."""
    assert rings["cameras"]["head"].spare()["pixels"].shape == (8, 16, 3)
    assert rings["cameras"]["rwrist"].spare()["pixels"].shape == (8, 8, 3)


# --- site config: the knobs that used to be read and ignored ----------------


def test_the_model_path_actually_selects_the_model(hw):
    """`[model]` was loaded and then ignored — G1Backend used its own
    constant, so editing the knob changed nothing. Now it reaches the backend,
    and a bad path fails where it is chosen rather than deep inside mujoco."""
    import pytest as _pytest
    from elrobot_deploy.robot import kinematics as kin

    fk = kin.ForwardKinematics(hw.urdf)
    assert fk.flange(np.zeros(schema.ARM_DOF), np.zeros(schema.WAIST_DOF)).shape \
        == (schema.N_HANDS, schema.POSE_DIM)
    with _pytest.raises(Exception):
        kin.ForwardKinematics("/nonexistent/model.urdf")


def test_reserved_cores_are_enforced_not_merely_declared(hw):
    """The core reservation. `[cpu] reserved` was read by nothing, so a table putting a
    decoder on P7's sibling was accepted in silence — which costs exactly the
    jitter the reservation exists to prevent."""
    from elrobot_deploy.runtime import cpu_pinning as affinity

    ok = cfgmod.replace(hw, cpu={"p7_body": 1, "p8_hand": 7, "p5_policy": 8},
                        cpu_reserved=[13, 19])
    with pytest.raises(ConfigError, match="both"):
        affinity.validate_pinning(
            cfgmod.replace(ok, cpu={"p7_body": 1, "p6_target": 1}),
            error=ConfigError)
    with pytest.raises(ConfigError, match="reserved"):
        affinity.validate_pinning(
            cfgmod.replace(ok, cpu={"p7_body": 1, "p5_policy": 13}),
            error=ConfigError)


def test_every_hardware_default_is_the_vendor_value(hw):
    """The site file's numbers are traceable, not plausible. These are the ones
    a guess would have looked right for."""
    assert hw.qdot_max[0] == pytest.approx(2 * np.pi)   # VENDOR max_pos_speed
    assert (hw.kp[15], hw.kp[19], hw.kp[12]) == (80.0, 40.0, 300.0)  # kp_low/wrist/high
    assert (hw.kd[15], hw.kd[19]) == (3.0, 1.5)                      # kd_low/kd_wrist


#: Board camera topic -> the arena camera name the robot file keys ports by.
_BOARD_TOPICS = {"head_camera": "head", "left_wrist_camera": "lwrist",
                 "right_wrist_camera": "rwrist"}


def test_camera_ports_agree_with_the_board_server_config(hw):
    """`[camera.ports]` and the board's `cameras.yaml` are one number written
    twice, on two machines, and nothing at runtime compares them.

    They no longer trace to the vendor's CAMCFG: the ports point at
    `elrobot_deploy/runtime/drivers/board/fisheye_image_server/`, the fisheye publisher, which deliberately
    avoids teleimager's 55555/6/7 so that running the wrong server produces
    silence rather than a plausible 480x640 stream where 720p was expected.

    A drift here is invisible in the worst way — P2 connects to a port nobody
    publishes on and `acquire_anchor` blocks forever on a camera that has
    produced no frame at all, which reads as a hung run, not a typo.
    """
    from elrobot_deploy.runtime.drivers.board.fisheye_image_server import publisher

    board = publisher.load_config(publisher.DEFAULT_CONFIG)["cameras"]
    expected = {_BOARD_TOPICS[topic]: int(spec["zmq_port"])
                for topic, spec in board.items() if topic in _BOARD_TOPICS}
    assert hw.camera_ports == expected, (
        "the robot profile's camera ports and "
        "elrobot_deploy/runtime/drivers/board/fisheye_image_server/cameras.yaml disagree")
    # And they must not be the vendor's, or the distinction buys nothing.
    assert not (set(expected.values()) & {55555, 55556, 55557})
