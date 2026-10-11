"""The rollout viewer — `g1d_deploy.dashboards.inspect`.

The claims worth testing are the ones whose failure would be *plausible on
screen*: a timeline that silently shows the end of the run at its start, a
positional qpos assignment that puts every right-arm angle on the wrong axis, a
record that loads against the wrong model because it was synced from another
machine. All three produce a picture, and all three produce the wrong one.

Every test builds its own synthetic record in `tmp_path`. Nothing here reads the
real rollouts on disk — those are data, they get deleted and
re-recorded, and a test suite that depends on them fails for reasons that have
nothing to do with the code.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from g1d_deploy.rollout_record_utils import reader as record_read
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.dashboards.inspect import discover, events, loader, scene, timeline
from g1d_deploy.dashboards.web import robot_scene

# A one-pixel JPEG. Content is irrelevant — the tests check offsets and slicing,
# never pixels — but it must be a real file so `Camera.jpeg` returns bytes.
_JPEG = bytes.fromhex(
    "ffd8ffdb004300ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc2000b08"
    "000100010101110000ffc40014000100000000000000000000000000000000ffda000801"
    "0100000001d2cf20ffd9")

_FRAME_IDX = np.dtype([("frame_id", "<u8"), ("t_recv", "<f8"),
                       ("t_observed", "<f8"), ("offset", "<u8"),
                       ("nbytes", "<u4")])

#: The pelvis pose these synthetic records are built around, in the G1-D's world.
#: An arbitrary but non-trivial rigid transform — a real translation and a real
#: rotation, so a bug that drops the rotation cannot pass. `world_from_base`
#: has to recover exactly this.
_PLANTED_YAW = 0.11
_PLANTED_T = np.array([
    [np.cos(_PLANTED_YAW), -np.sin(_PLANTED_YAW), 0.0, 0.055],
    [np.sin(_PLANTED_YAW), np.cos(_PLANTED_YAW), 0.0, 0.012],
    [0.0, 0.0, 1.0, 0.690],
    [0.0, 0.0, 0.0, 1.0]])


def _fill_flange(state) -> None:
    """Give the synthetic record a physically consistent `flange` column.

    Rather than importing `g1d_deploy.robot.kinematics` — which would tie this suite to
    whatever the live control stack currently is, and has already broken once
    mid-refactor — the flange is generated from the VIEWER's own G1-D model and
    then expressed in a planted pelvis frame:

        T_base_flange = inv(_PLANTED_T) @ T_world_flange

    That makes the record self-consistent by construction and turns
    `world_from_base` into a test with a known answer: it must return
    `_PLANTED_T`.
    """
    import mujoco

    model, data = robot_scene.build_model()
    jmap = robot_scene._JointMap(model, _FACTS)
    inv = np.linalg.inv(_PLANTED_T)
    for i in range(len(state)):
        q = state["q"][i]
        data.qpos[:] = 0.0
        data.qpos[jmap.arm_adr] = q[jmap.arm_cols]
        data.qpos[jmap.waist_adr] = q[jmap.waist_cols]
        mujoco.mj_forward(model, data)
        for h, bid in enumerate(jmap.wrist_body):
            T = np.eye(4)
            T[:3, :3] = data.xmat[bid].reshape(3, 3)
            T[:3, 3] = data.xpos[bid]
            T = inv @ T
            state["flange"][i, h, :3] = T[:3, 3]
            state["flange"][i, h, 3:] = _mat_to_quat(T[:3, :3])


def _mat_to_quat(R):
    """3x3 -> wxyz. Local, for the same decoupling reason as `_fill_flange`."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
             (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
             0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


class _FACTS:
    """The joint map the synthetic records use — the real lowstate layout."""

    joints_arm = list(range(15, 29))
    joints_waist = [12, 13, 14]


#: The two wrists free-run at very slightly different rates on the real robot
#: (29.99-30.00 Hz measured), with no shared trigger. Reproduced here because a
#: pair of cameras in exact lockstep is the one case where the timeline's
#: phase-disagreement handling is never exercised.
_CAM_HZ = {"lwrist": 30.0, "rwrist": 30.05}


#: The commanded flange pose the synthetic chunks carry, in PELVIS coordinates —
#: which is what `chunk["pose"]` holds (the contract's flange pose). Deliberately NOT where the arm actually
#: is: the whole point of the overlay is to show the gap between asked and got,
#: so a bug that draws the ghost on the measured arm has to fail.
_CMD_POSE = {
    0: np.array([0.31, 0.22, 0.07, 0.9689124, 0.0, 0.2474040, 0.0]),
    1: np.array([0.28, -0.19, 0.11, 0.9689124, 0.0, 0.0, 0.2474040]),
}
#: Commanded jaw radians, both distinct from the record's `grip_meas` (1.0/4.0)
#: so "the ghost follows the command" and "the ghost follows the measurement"
#: cannot both pass.
_CMD_JAW = (3.1, 0.7)

#: The joint angles the synthetic `body_targets` publish — `q`, the IK's OUTPUT,
#: in the body target's own order: 14 arm then 3 waist. Every entry that the
#: measured posture also moves is set to a DIFFERENT value, for the same reason
#: `_CMD_POSE` is nowhere near the measured arm: a solved overlay that quietly
#: posed the measured state would look completely plausible on screen.
#:
#: The waist pair is the real robot's own case rather than an arbitrary offset:
#: commanded to zero, measured at 0.03 yaw and 0.039 of gravity bow, which is
#: exactly the divergence `g1d_deploy.robot.kinematics`'s docstring warns about.
_CMD_Q = np.zeros(17)
_CMD_Q[0] = 0.42        # left shoulder pitch; measured runs 0.20 -> 0.22
_CMD_Q[3] = -0.15       # left elbow; measured is 0
_CMD_Q[10] = 0.95       # right elbow; measured runs 0.60 -> 0.66
_CMD_Q[14] = 0.02       # waist slot 0, the yaw; measured 0.03
_CMD_Q[15] = -0.05      # waist slot 1, the bow; measured 0.039 (and the sign
                        # differs too, so a swap of the two cannot pass)

#: The trunk P6 pinned the IK at for those ticks — `body_targets["waist_ik"]`,
#: the frame `_CMD_Q`'s fourteen arm angles were solved in. This is the record's
#: MEASURED waist (`state["q"][12, 13]` below), which is the robot's own case:
#: the trunk rests where gravity leaves it while the tick commands it elsewhere.
#:
#: Distinct from `_CMD_Q[14:17]` in both live slots AND opposite in sign on the
#: bow, so an overlay that posed the commanded waist, or that swapped the two
#: slots, cannot pass a test written against this.
_CMD_WAIST_IK = np.array([0.03, 0.039, 0.0])

#: The commanded ticks the synthetic record covers. `k` starts at 1 because
#: `k < 1` is P0's homing ramp, which has no Cartesian target behind it.
_CMD_K0, _CMD_NK = 1, 40

#: How long the synthetic inference takes, and how far before the ticks it
#: produced the replan was triggered. Both in the range the real stack measures
#: (75-94 ms of inference on the 2026-08-26 rollouts), so a test that asserts on
#: them is asserting about a plausible record.
_INFER_S = 0.090
_LEAD_S = 0.200


def _write_commands(run_dir, manifest, *, t0, duration, control_hz=30.0,
                    record_pose=True, record_waist_ik=True, n_chunks=1,
                    warmup_chunk=False, tick_hitch=0.0):
    """`chunks.bin` + the two target rings, covering `[_CMD_K0, +_CMD_NK)`.

    `record_pose` picks the record GENERATION. True writes what the deploy stack
    writes from 2026-08-27: `body_targets["pose"]` holds the array P6 handed the
    IK. False strips the column, producing a record from before it existed, which
    the viewer has to reconstruct from the chunk ring instead.

    `record_waist_ik` is the same kind of switch for the 2026-09-03 column, and
    it is separate because the two arrived apart: a record can hold the pose the
    IK was given without holding the trunk it was given it in, and that is
    exactly the generation the solved overlay's waist switch has to grey out on.

    `n_chunks` picks the MODE. One chunk covers each tick exactly once, which is
    sync (`n_actions == P`, the sync invariant). Two overlapping chunks is async, where the
    commanded pose is a blend of both and the chunk-ring reconstruction has no
    honest answer.

    `warmup_chunk` prepends a chunk from before the rollout: P5's warm-up, whose
    ticks are P0's homing and whose instants predate the ACTIVE edge.
    `tick_hitch` inserts a stall into the tick line — the real one, where `k`
    jumps by 2 and the wall clock by 149 ms at every sync chunk boundary.
    """
    from g1d_deploy.rollout_record_utils import reader as record_read
    from g1d_deploy.runtime.processes import p9_record as p9

    start = t0 + 0.25 * duration
    horizon = _CMD_NK
    chunk_rec = np.dtype(schema.chunk_dtype(horizon).descr + [("t_observed", "<f8")])
    chunks = np.zeros(n_chunks + bool(warmup_chunk), dtype=chunk_rec)
    for c in range(n_chunks):
        chunks["chunk_id"][c] = c + 1
        # Triggered before the ticks it produced, and written 90 ms later. The
        # stagger keeps two async chunks from sharing an instant, which would
        # make "the marks are per chunk" untestable.
        chunks["t_trigger"][c] = start - _LEAD_S + 0.05 * c
        chunks["t_chunk_written"][c] = chunks["t_trigger"][c] + _INFER_S
        # Overlapping when there is more than one: every chunk starts at the
        # same tick and runs the full horizon, so each tick has `n_chunks`
        # contributors — the defining property of async, not an edge case.
        chunks["k_first_action"][c] = _CMD_K0
        chunks["n_actions"][c] = _CMD_NK
        for h, pose in _CMD_POSE.items():
            chunks["pose"][c, :, h] = pose
        for h, jaw in enumerate(_CMD_JAW):
            chunks["hand"][c, :, h, 0] = jaw

    if warmup_chunk:
        # P5 warming: an inference whose action range is P0's homing ticks, and
        # which finished before the rollout began. Nothing it produced ever ran.
        chunks["chunk_id"][-1] = 99
        chunks["k_first_action"][-1] = -60
        chunks["n_actions"][-1] = _CMD_NK
        chunks["t_trigger"][-1] = t0 + 0.02 * duration
        chunks["t_chunk_written"][-1] = t0 + 0.02 * duration + _INFER_S

    # Deadlines spread across the middle of the run, so some display frames fall
    # before the first command and some after the last — both must read as
    # "nothing commanded" rather than holding the nearest one.
    deadlines = start + np.arange(_CMD_NK) / control_hz
    if tick_hitch:
        # The arm waiting out an inference: every tick from the midpoint on is
        # late by the hitch, and the tick line stops being `k / control_hz`.
        deadlines[_CMD_NK // 2:] += tick_hitch

    from g1d_deploy.runtime.processes.p9_record import TARGET_REC
    body_dtype = TARGET_REC
    drop = ({"pose"} if not record_pose else set()) | \
           ({"waist_ik"} if not record_waist_ik else set())
    if drop:
        body_dtype = np.dtype([f for f in body_dtype.descr if f[0] not in drop])
    body = np.zeros(_CMD_NK, dtype=body_dtype)
    body["k"] = np.arange(_CMD_K0, _CMD_K0 + _CMD_NK)
    body["deadline"] = deadlines
    body["t_observed"] = deadlines
    # The IK's output for those poses. Constant across the window, which is all
    # the overlay's assertions need and keeps them independent of the tick line.
    body["q"] = _CMD_Q
    body["err"] = 0.004
    if record_pose:
        # What P6 stores: the blend's own output, the same array it hands the
        # solver. Identical to the chunk's here because this record's chunks all
        # carry the same pose, which is what lets one set of assertions check
        # both generations.
        for h, pose in _CMD_POSE.items():
            body["pose"][:, h] = pose
    if record_waist_ik:
        # The third thing P6 stores: the trunk it pinned the model at. Not
        # `body["q"][14:17]`, which is what the tick COMMANDED — see
        # `_CMD_WAIST_IK`.
        body["waist_ik"] = _CMD_WAIST_IK
    for h, jaw in enumerate(_CMD_JAW):
        body["hand"][:, h, 0] = jaw
    body.tofile(run_dir / "targets.bin")

    chunks.tofile(run_dir / "chunks.bin")
    manifest["streams"].update({
        "chunks": {"file": "chunks.bin", "descr": chunk_rec.descr,
                   "records": len(chunks)},
        "targets": {"file": "targets.bin", "descr": body_dtype.descr,
                    "records": _CMD_NK},
    })


#: Where in the synthetic run P0 goes ACTIVE, as a fraction of its duration.
#: A quarter in, so there is real recording on both sides of the edge: the
#: setup that must be cut and the rollout that must survive.
_ACTIVE_FRAC = 0.25


def _write_session(run_dir, manifest, *, t0, duration, active=True):
    """`session.bin` — P0's phase edges, as P9 sweeps them.

    The phases before ACTIVE are what the viewer exists to skip, so they are
    spread across the front of the run rather than bunched at `t0`: a cut that
    only worked when the setup was instantaneous would pass on a degenerate
    record and fail on every real one.

    `active=False` is a run that never got there — aborted in WARMUP, or stopped
    at the READY prompt. It has a session log and no ACTIVE edge, which is a
    different case from having no log at all, and the two must not be conflated.
    """
    from g1d_deploy.runtime.processes import p9_record as p9

    phases = [schema.CHECK, schema.HOME, schema.WARMUP, schema.READY]
    if active:
        phases.append(schema.ACTIVE)
    rows = np.zeros(len(phases), dtype=p9.SESSION_REC)
    rows["session"] = phases
    # The pre-ACTIVE phases share the run-up; ACTIVE lands exactly on the
    # fraction above, which is what the assertions quote.
    edges = list(np.linspace(t0, t0 + _ACTIVE_FRAC * duration, len(phases)))
    rows["t_observed"] = edges
    rows["warm"] = [0, 0, 0, 1, 1][:len(phases)]
    rows.tofile(run_dir / "session.bin")
    manifest["streams"]["session"] = {
        "file": "session.bin", "descr": p9.SESSION_REC.descr,
        "records": len(rows)}
    return edges[-1] if active else None


def _write_record(run_dir, *, n_state=2000, state_hz=1000.0,
                  t0=1000.0, cam_offset=0.0, model_block=True, n_cam=None,
                  commands=False, record_pose=True, record_waist_ik=True,
                  n_chunks=1,
                  session=False, active=True, warmup_chunk=False,
                  tick_hitch=0.0, head=False,
                  checkpoint="/box/runs/my_run_1/29999"):
    """A minimal but structurally honest rollout on disk.

    `session` picks the record generation for the phase log, the same way
    `record_pose` does for the commanded pose: False is anything written before
    2026-08-27, which has no `session.bin` and therefore no ACTIVE edge to open
    on. Default False so that every test predating the phase log keeps asserting
    against the window it was written for.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    state = np.zeros(n_state, dtype=schema.STATE)
    state["index"] = np.arange(1, n_state + 1)
    state["t_recv"] = t0 + np.arange(n_state) / state_hz
    # A posture that is not the zero pose, so FK has something to disagree about.
    for i in range(n_state):
        state["q"][i, 15] = 0.20 + 0.001 * i        # left shoulder pitch
        state["q"][i, 25] = 0.60 + 0.002 * i        # right elbow
        state["q"][i, 12] = 0.03                    # waist yaw
        state["q"][i, 13] = 0.039                   # the bow
        state["grip_meas"][i, 0] = 1.0
        state["grip_meas"][i, 6] = 4.0
    _fill_flange(state)
    state.tofile(run_dir / "state.bin")

    duration = n_state / state_hz
    cameras = {}
    # `head=True` is a rig with a third camera that is
    # recorded and never observed. It is off by default so every test written
    # against the two-wrist record keeps asserting what it was written for.
    rates = dict(_CAM_HZ, head=30.0) if head else dict(_CAM_HZ)
    for name, hz in rates.items():
        count = n_cam if n_cam is not None else int((duration - cam_offset) * hz)
        idx = np.zeros(count, dtype=_FRAME_IDX)
        idx["frame_id"] = np.arange(1, count + 1)
        idx["t_recv"] = t0 + cam_offset + np.arange(count) / hz
        idx["offset"] = np.arange(count) * len(_JPEG)
        idx["nbytes"] = len(_JPEG)
        idx.tofile(run_dir / f"{name}.idx.bin")
        (run_dir / f"{name}.jpg.bin").write_bytes(_JPEG * count)
        cameras[name] = {"blob": f"{name}.jpg.bin", "index": f"{name}.idx.bin",
                         "index_descr": _FRAME_IDX.descr, "frames": count,
                         "dropped": 0}

    # `replan_period_ticks` / `action_horizon`, not `P` / `H`: those are
    # properties of `DeployConfig`, and what a real manifest carries is the
    # field names. A fixture that held the short ones let a summary reading
    # `"P"` look correct while returning `None` on every real record.
    deploy_config = {"prompt": "t", "checkpoint": checkpoint, "sync": True,
                     "replan_period_ticks": 25, "action_horizon": 50,
                     "control_hz": 30.0}
    if head:
        deploy_config.update(
            cameras={
                "lwrist": {"height": 720, "width": 1280, "binocular": False},
                "rwrist": {"height": 720, "width": 1280, "binocular": False},
                "head": {"height": 480, "width": 1280, "binocular": True},
            },
            camera_slots={"lwrist": "left_wrist_0_rgb",
                          "rwrist": "right_wrist_0_rgb"},
            observed_cameras=["lwrist", "rwrist"],
            primary_camera="rwrist")
    manifest = {
        "created": "2026-08-26T12:00:00",
        "streams": {"state": {"file": "state.bin", "descr": schema.STATE.descr,
                              "records": n_state}},
        "cameras": cameras,
        "deploy_config": deploy_config,
        "joints": {"arm": list(range(15, 29)), "waist": [12, 13, 14]},
    }
    if model_block:
        # The robot block P9 writes (`rollout_record_utils.manifest.robot_block`): the shipped
        # profile chain, verbatim. No path in it has to exist here.
        from g1d_deploy.rollout_record_utils import manifest as provenance
        from g1d_deploy.config_loading.robot_file import load_robot
        manifest["robot"] = provenance.robot_block(
            "g1d_fisheye_dex1", load_robot("g1d_fisheye_dex1"))
        manifest["model"] = {"urdf": "assets/unitree_g1d/g1_d.urdf",
                             "waist_joint_names": ["torso_Joint", "Yaw_Joint",
                                                   None]}
    if commands:
        _write_commands(run_dir, manifest, t0=t0, duration=duration,
                        record_pose=record_pose, record_waist_ik=record_waist_ik,
                        n_chunks=n_chunks,
                        warmup_chunk=warmup_chunk, tick_hitch=tick_hitch)
    if session:
        _write_session(run_dir, manifest, t0=t0, duration=duration,
                       active=active)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


@pytest.fixture
def record(tmp_path):
    return _write_record(tmp_path / "20260826-120000")


# --- discovery --------------------------------------------------------------
#
# The layout-free rules have their own suite (`test_inspect_discover.py`);
# these are the ones about what the listing says of a record.


def test_a_recording_is_listed_with_the_checkpoint_it_was_flown_on(tmp_path):
    """The chip beside Load reads the listing, so the listing has to carry it.

    Which checkpoint a rollout used is what decides whether it is worth
    loading, and the directory name cannot say.
    """
    run = tmp_path / "my_task_x" / "my_run_1" / "rollout_record" / "20260826-120000"
    _write_record(run, checkpoint="/box/runs/my_task/my_run_1/best/4500")

    rec = discover.scan(tmp_path)[0]
    assert rec.details["checkpoint_step"] == "4500"
    assert rec.details["checkpoint_run"] == "my_run_1/best"
    assert rec.details["checkpoint"] == "/box/runs/my_task/my_run_1/best/4500"
    listing = discover.as_json(discover.scan(tmp_path), tmp_path)
    assert listing["groups"][0]["records"][0]["details"]["checkpoint_step"] == "4500"


@pytest.mark.parametrize("checkpoint,run,step", [
    # The ordinary shape: a step directory named after its training step.
    ("/x/runs/my_task/my_run_6/29999", "my_run_6", "29999"),
    # `best/` has to survive into the run name, or two rollouts of different
    # networks read as the same run.
    ("/x/runs/my_task/my_run_6/best/4500", "my_run_6/best", "4500"),
    # Neither shape: degrade to the last component rather than invent a step.
    ("/x/checkpoints/my_weights", "my_weights", ""),
    ("", "", ""),
])
def test_a_checkpoint_path_is_split_into_a_run_and_a_step(checkpoint, run, step):
    assert discover._checkpoint_id(checkpoint) == (run, step)


def test_p_and_h_are_read_under_the_names_a_manifest_actually_uses(tmp_path):
    """`P` and `H` are properties of `DeployConfig`, not fields, so a manifest
    carries `replan_period_ticks` and `action_horizon`."""
    manifest = {"deploy_config": {"replan_period_ticks": 25, "action_horizon": 50,
                                  "control_hz": 30.0}}
    assert discover.specifics(manifest)["P"] == 25
    assert discover.specifics(manifest)["H"] == 50


def test_the_rollouts_own_settings_win_over_the_launch_config():
    """Settings can change between the rollouts of one launch, so the
    chip reads `rollouts[0].active_config` first and `deploy_config` second."""
    manifest = {
        "deploy_config": {"replan_period_ticks": 25, "action_horizon": 50,
                          "control_hz": 30.0, "sync": True, "sample_steps": 10,
                          "prompt": "launch prompt"},
        "rollouts": [{"index": 2, "generation": 3, "prompt": "second prompt",
                      "active_config": {"sync": False, "replan_period": 10,
                                        "control_hz": 30.0, "ik": "arm",
                                        "sample_steps": 5,
                                        "servo": {"filter_hz": 4.0}}}],
        "policy": {"entry": "/p/policy.py:Policy", "class": "pkg.policy.Policy",
                   "git": {"sha": "b" * 40, "dirty": True}},
        "infra": {"version": "0.1.0", "git": {"sha": "a" * 40, "dirty": False}},
        "robot": {"name": "g1d_fisheye_dex1",
                  "files": [{"path": "/r/g1d_base.toml", "text": ""}]},
        "checkpoint": {"path": "/runs/my_run/1000"},
    }
    d = discover.specifics(manifest)
    assert (d["sync"], d["P"], d["sample_steps"]) == (False, 10, 5)
    assert d["prompt"] == "second prompt" and d["generation"] == 3
    assert d["policy_entry"] == "/p/policy.py:Policy"
    assert d["policy_class"] == "pkg.policy.Policy"
    assert d["policy_sha"] == "b" * 12 and d["policy_dirty"]
    assert d["infra_sha"] == "a" * 12 and d["infra_version"] == "0.1.0"
    assert d["robot"] == "g1d_fisheye_dex1"
    assert d["checkpoint"] == "/runs/my_run/1000"
    assert d["servo"] == {"filter_hz": 4.0}


def test_a_manifest_that_will_not_parse_is_refused_at_the_listing(tmp_path):
    """`loader` rebuilds every dtype from the manifest, so a truncated one is a
    record that cannot be opened. Saying so in the list beats a Load that
    spins and fails."""
    run = tmp_path / "20260826-120000"
    _write_record(run)
    (run / "manifest.json").write_text('{"created": "2026-08-2')

    rec = discover.scan(tmp_path)[0]
    assert not rec.loadable
    assert "manifest.json could not be read" in rec.reason
    assert rec.details == {}


# --- loader -----------------------------------------------------------------


def test_a_record_without_a_robot_file_is_unsupported(tmp_path):
    """No robot file in the manifest: nothing to rebuild the robot from, so
    the record is refused rather than drawn as a guess."""
    from g1d_deploy.dashboards.inspect.record_robot import UnsupportedRecord

    with pytest.raises(UnsupportedRecord, match="unsupported"):
        loader.load(_write_record(tmp_path / "r", model_block=False))


def test_a_record_with_no_manifest_refuses_with_a_reason(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest"):
        loader.load(d)


# --- timeline ---------------------------------------------------------------


def test_the_window_is_the_intersection_not_the_union(tmp_path):
    """A union window opens on a black camera pane, which reads as a broken
    viewer rather than as a stream that had not started yet."""
    rollout = loader.load(_write_record(tmp_path / "r", cam_offset=0.5))
    try:
        tl = timeline.build(rollout)
        assert tl.t0 >= rollout.cameras["lwrist"].index["t_recv"][0]
        assert tl.t0 >= rollout.state["t_recv"][0]
        # Every stream has a record at every frame, by construction.
        for column in tl.index.values():
            assert int(column.min()) >= 0
    finally:
        rollout.close()


def test_lookup_is_zero_order_hold(record):
    """A setpoint IS held until the next one, and holding the last camera frame
    is honest where blending would invent an image the robot never saw."""
    rollout = loader.load(record)
    try:
        tl = timeline.build(rollout)
        for f in range(tl.n):
            i = tl.at("lwrist", f)
            t_frame = rollout.cameras["lwrist"].index["t_recv"][i]
            assert t_frame <= tl.t[f] + 1e-12
            nxt = i + 1
            if nxt < len(rollout.cameras["lwrist"].index):
                assert rollout.cameras["lwrist"].index["t_recv"][nxt] > tl.t[f]
    finally:
        rollout.close()


def test_the_hold_returns_nothing_rather_than_the_last_record(record):
    """`searchsorted(...) - 1` yields -1 below the first sample, and Python reads
    -1 as the LAST element — presenting the end of a run as its beginning. The
    sentinel is named so callers cannot use it by accident."""
    times = np.array([10.0, 11.0, 12.0])
    got = timeline._hold(times, np.array([9.0, 10.0, 11.5, 99.0]))
    assert got.tolist() == [timeline.NOTHING, 0, 1, 2]
    assert timeline.NOTHING == -1


def test_every_commanded_deadline_is_a_frame_and_that_frame_is_exact(tmp_path):
    """The reason the grid exists in this shape.

    Both the display grid and the tick line are nominally 30 Hz, and it is very
    easy to assume they are therefore the same grid. They are not — sync
    re-stamps the epoch once per chunk, so the tick line slips and the phase
    wanders. A uniform grid left the commanded overlay a mean 37 ms stale
    against a measured pose accurate to half a millisecond, which put half of
    the apparent commanded-vs-measured gap into the viewer rather than the
    robot.
    """
    rollout = loader.load(_write_record(tmp_path / "cmds", commands=True))
    try:
        tl = timeline.build(rollout)
        bt = rollout.body_targets
        deadlines = bt["deadline"][bt["k"] >= 1]
        inside = deadlines[(deadlines > tl.t0) & (deadlines < tl.t1)]
        assert len(inside)

        # Every deadline is a frame, to the float, and it is flagged as one.
        for d in inside:
            f = tl.frame_at(d)
            assert tl.t[f] == d
            assert tl.on_anchor[f]
        assert int(tl.on_anchor.sum()) == len(inside)

        # And on those frames the commanded pose is the exact one P6 published,
        # not one held over from an earlier tick.
        col = tl.index["command"]
        age = tl.t[tl.on_anchor] - bt["deadline"][col[tl.on_anchor]]
        assert np.all(age == 0.0)
    finally:
        rollout.close()


def test_filler_keeps_the_nominal_period_and_never_shadows_an_anchor(tmp_path):
    """Filler covers the head, the tail and the replan gaps so the cameras and
    the measured pose still move there. It is laid at exactly `1/fps` from the
    start of its gap and stops short of the next anchor, rather than being
    spread evenly to meet it: even spreading gives a command-less record a
    period of `(t1-t0)/n` instead of `1/fps`, which is enough drift to walk a
    30.0 Hz camera off its own samples."""
    # `tick_hitch` is the real replan gap: the arm waiting out an inference,
    # where `k` keeps stepping by one but the wall clock jumps. Without it every
    # deadline is exactly one period from the last and the filler never runs
    # between two anchors at all — the case this test exists for.
    rollout = loader.load(
        _write_record(tmp_path / "cmds", commands=True, tick_hitch=0.149))
    try:
        tl = timeline.build(rollout)
        dt = 1.0 / tl.fps
        gaps = np.diff(tl.t)

        # The widest possible interval is 1.5 periods: at exactly that width the
        # single filler frame would land within half a period of the next anchor
        # and is dropped rather than shadowing it. Anything wider was filled.
        assert gaps.max() <= (1.0 + timeline._MIN_SEPARATION) * dt + 1e-9
        # And nothing is a near-duplicate of its neighbour.
        assert gaps.min() >= timeline._MIN_SEPARATION * dt - 1e-9

        # The hitch really is present and really was filled.
        assert np.diff(np.sort(rollout.body_targets["deadline"])).max() > 4 * dt

        # Filler that follows filler sits at exactly the nominal period; only
        # the runt against the next anchor is allowed to be short.
        ordinary = gaps[~tl.on_anchor[1:]]
        assert np.allclose(ordinary[:-1], dt, atol=1e-12)
    finally:
        rollout.close()


def test_the_measured_pose_is_interpolated_and_the_commanded_one_is_held(tmp_path):
    """Two rules, one per kind of signal. A target and a camera frame are HELD —
    the target until the next arrives, the image until the next is captured.
    Joint state is continuously sampled at ~1040 Hz, so it is interpolated to the
    frame instant, which is what makes "both stated at the same instant" true by
    construction instead of true to within a rounding argument."""
    rollout = loader.load(_write_record(tmp_path / "cmds", commands=True))
    try:
        tl = timeline.build(rollout)
        t_recv, q = rollout.state["t_recv"], rollout.state["q"]

        # Interpolated: on a frame that falls between two samples the value is
        # between them too, and it is NOT either one.
        mid = tl.n // 2
        i = int(np.searchsorted(t_recv, tl.t[mid], side="right") - 1)
        if t_recv[i] < tl.t[mid] < t_recv[i + 1]:
            u = (tl.t[mid] - t_recv[i]) / (t_recv[i + 1] - t_recv[i])
            assert np.allclose(tl.q[mid], q[i] + u * (q[i + 1] - q[i]))

        # Clamped, never extrapolated, at both ends.
        assert np.allclose(tl.q[0], q[timeline._hold(t_recv, tl.t[:1])[0]])

        # Held: the commanded index is still a zero-order hold, so the overlay
        # steps and never ramps.
        col = tl.index["command"]
        shown = col >= 0
        assert np.all(rollout.body_targets["deadline"][col[shown]] <= tl.t[shown])
    finally:
        rollout.close()


# --- recorded vs. observed --------------------------------------------------


def test_a_camera_the_policy_never_read_is_loaded_and_labelled(tmp_path):
    """A camera can be recorded on every rollout and read by nothing.

    It has to appear — that is the whole reason it is recorded — and it must not
    appear as an input. A pane that looked like the wrists would put a view the
    policy never saw beside the two it acted on, which is the one thing a reader
    of a rollout must not get wrong.
    """
    roll = loader.load(_write_record(tmp_path / "20260903-120000", head=True))
    try:
        assert set(roll.cameras) == {"lwrist", "rwrist", "head"}

        head = roll.cameras["head"]
        assert head.used is False
        assert head.slot is None
        assert head.primary is False
        # One frame, two eyes. Recorded whole; split at display.
        assert head.binocular is True

        # The captured shape, so the viewer can reserve the right box instead
        # of cropping the frame to fill the pane. For a binocular camera the
        # box it reserves is HALF this width — the left eye is what is shown —
        # and that arithmetic is the page's, so what is pinned here is that the
        # numbers reach it at all.
        assert (head.height, head.width) == (480, 1280)

        assert roll.cameras["rwrist"].used is True
        assert roll.cameras["rwrist"].slot == "right_wrist_0_rgb"
        assert (roll.cameras["rwrist"].height,
                roll.cameras["rwrist"].width) == (720, 1280)
        # The primary's chosen frame IS t_anchor, so it is worth naming apart
        # from the other camera the policy also read.
        assert roll.cameras["rwrist"].primary is True
        assert roll.cameras["lwrist"].primary is False

        # Observed first, record-only last: what the actions were computed from
        # comes before what is merely context.
        assert list(roll.cameras) == ["lwrist", "rwrist", "head"]
    finally:
        roll.close()


def test_a_record_only_camera_does_not_retire_the_wrist_skew_check(tmp_path):
    """The skew test used to be "exactly two cameras", which a third would have
    silently switched off. It asks whether the two views the POLICY saw stepped
    together, so the observed pair is its subject and the head is not."""
    roll = loader.load(_write_record(tmp_path / "20260903-120001", head=True))
    try:
        d = timeline.describe(timeline.build(roll), roll)
        # 30.00 vs 30.05 Hz: the two wrists really do drift apart, and the
        # record says so. Zero here would mean the check had stopped running.
        assert d["camera_phase_disagreements"] > 0
    finally:
        roll.close()


def test_camera_roles_read_an_older_record_as_what_it_meant(tmp_path):
    """`observed_cameras` did not exist before 2026-09-03, and no record written
    before it has a record-only camera: every camera those runs declared was
    observed. So the fallbacks are what those records MEAN, not a guess."""
    # A record from the two-wrist era: `camera_slots`, no `observed_cameras`.
    older = {"deploy_config": {
        "cameras": {"lwrist": {}, "rwrist": {}},
        "camera_slots": {"lwrist": "left_wrist_0_rgb",
                         "rwrist": "right_wrist_0_rgb"},
        "primary_camera": "rwrist"}}
    roles = loader.camera_roles(older)
    assert all(r["used"] for r in roles.values())
    assert roles["rwrist"]["primary"] and not roles["lwrist"]["primary"]
    assert roles["lwrist"]["slot"] == "left_wrist_0_rgb"
    # Absent means False, never None: a record predating the flag has no
    # binocular camera in it.
    assert roles["rwrist"]["binocular"] is False
    # An older record dumps no camera shapes. None rather than a guess: the
    # page falls back to the frame's own aspect, which is right for everything
    # except a binocular camera, and those did not exist in such a record.
    assert roles["rwrist"]["height"] is None
    assert roles["rwrist"]["width"] is None

    # A record of the base DeployConfig has no `camera_slots`
    # at all. Every camera it declared fed the policy.
    ancient = {"deploy_config": {"cameras": {"lwrist": {}, "rwrist": {}},
                                 "primary_camera": "rwrist"}}
    assert all(r["used"] for r in loader.camera_roles(ancient).values())

    # And a manifest with no camera config whatsoever says nothing, rather than
    # claiming every camera was unused.
    assert loader.camera_roles({}) == {}


def test_the_two_cameras_are_allowed_to_disagree(tmp_path):
    """Both free-run at ~30 Hz with no shared trigger, so they drift in and out
    of phase. Those images really were captured milliseconds apart; forcing one
    index on both would be a lie about the record.

    Thirty seconds, not the default two. The rates here differ by 0.05 Hz, which
    is one frame of drift per 20 s — at exactly 20 s the count is still 0, so a
    shorter record would assert the mechanism works while proving nothing.
    Cranking the rate difference instead would make the test pass against a
    record no robot produces. `state_hz` carries the length, since the state rate
    is irrelevant to camera phase.
    """
    rollout = loader.load(
        _write_record(tmp_path / "long", n_state=3000, state_hz=100.0))
    try:
        tl = timeline.build(rollout)
        d = timeline.describe(tl, rollout)
        assert tl.duration > 29.0
        assert d["camera_phase_disagreements"] > 0
        # And they are still nearly in step — a big number here would mean the
        # window or the hold is wrong, not that the cameras drifted.
        assert d["camera_phase_disagreements"] < tl.n // 2
    finally:
        rollout.close()


# --- The ACTIVE window ------------------------------------------------------


def test_the_window_opens_when_the_policy_takes_the_arm(tmp_path):
    """Everything before ACTIVE is the machine getting ready, not the rollout.

    CHECK settling, P0's homing ramp, P5 warming, READY waiting on a human. On
    `20260826-174457` that is 60 of the 80 seconds recorded, so three quarters of
    the scrubber went on an arm holding still at the home pose.
    """
    run = _write_record(tmp_path / "r", session=True)
    rollout = loader.load(run)
    try:
        active = rollout.active_t
        assert active is not None
        tl = timeline.build(rollout)
        assert tl.window == "active"
        assert abs(tl.t0 - active) < 1e-9
        # A quarter cut off the front, and the rest kept intact.
        assert abs(tl.skipped - 0.25 * 2.0) < 0.05
        assert abs(tl.duration - 0.75 * 2.0) < 0.05
        # Nothing was deleted or rebased: frame timestamps are still absolute
        # monotonic instants that can be quoted back at the raw streams.
        i = tl.at("state", 0)
        assert abs(float(rollout.state["t_recv"][i]) - tl.t[0]) < 2e-3
    finally:
        rollout.close()


def test_the_skipped_setup_is_reported_not_silently_swallowed(tmp_path):
    """A 20 s scrubber over an 80 s recording is a fact the reader must be told.

    Without it the rollout simply looks shorter than it was, and two records
    trimmed by different amounts look directly comparable when they are not.
    """
    rollout = loader.load(_write_record(tmp_path / "r", session=True))
    try:
        d = timeline.describe(timeline.build(rollout), rollout)
        assert d["window"] == "active"
        assert d["skipped_s"] > 0.4
        assert round(d["skipped_s"] + d["duration_s"], 1) == 2.0
    finally:
        rollout.close()


def test_a_record_with_no_session_log_keeps_the_whole_window(tmp_path):
    """Every rollout written before 2026-08-27.

    There is no ACTIVE edge to open on, and inventing one — from the first
    commanded tick, say — would silently show a different span for records that
    are otherwise identical. The full window is the honest answer, and `window`
    says which rule produced it.
    """
    rollout = loader.load(_write_record(tmp_path / "old", session=False))
    try:
        assert rollout.active_t is None
        tl = timeline.build(rollout)
        assert tl.window == "streams"
        assert tl.skipped == 0.0
        assert tl.duration > 1.9
    finally:
        rollout.close()


def test_a_run_that_never_went_active_keeps_the_whole_window(tmp_path):
    """Aborted in WARMUP, or stopped at the READY prompt.

    It HAS a phase log and it has no ACTIVE edge, which is a different case from
    having no log at all — and the one where a cut would be most destructive,
    since the only thing such a record contains is the setup. An empty timeline
    is a worse answer than the whole recording.
    """
    rollout = loader.load(_write_record(tmp_path / "aborted", session=True,
                                        active=False))
    try:
        assert len(rollout.session), "the phase log is present"
        assert rollout.active_t is None
        tl = timeline.build(rollout)
        assert tl.window == "streams"
        assert tl.n > 1
    finally:
        rollout.close()


def test_an_active_edge_outside_the_stream_overlap_is_ignored(tmp_path):
    """P9 can attach late, or a run can end the instant it goes ACTIVE.

    Either way the edge falls outside `[t0, t1]` and cutting to it would produce
    a degenerate or empty grid. The guard is `t0 < active < t1` rather than a
    clamp, because clamping would report `window == "active"` for a window that
    was not actually cut to anything.
    """
    from g1d_deploy.runtime.processes import p9_record as p9

    for shift in (-500.0, +500.0):            # far before, and far after
        run = _write_record(tmp_path / f"edge{shift:+.0f}", session=True)
        rows = np.fromfile(run / "session.bin", dtype=p9.SESSION_REC)
        rows["t_observed"] += shift
        rows.tofile(run / "session.bin")

        rollout = loader.load(run)
        try:
            assert rollout.active_t is not None, "the edge exists"
            tl = timeline.build(rollout)
            assert tl.window == "streams"
            assert tl.skipped == 0.0
            assert tl.n > 1
        finally:
            rollout.close()


def test_the_cut_moves_every_pane_together(tmp_path):
    """Cameras, state and the render share one grid, so one `t0` moves them all.

    Written because the obvious wrong implementation — trimming the camera
    columns and leaving the state column alone — passes a duration check and
    puts the robot's pose out of step with the video by the whole setup.
    """
    run = _write_record(tmp_path / "r", session=True, commands=True)
    rollout = loader.load(run)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert sc.transforms.shape[0] == tl.n
        assert sc.ghost_valid.shape[0] == tl.n
        for name in ("state", "lwrist", "rwrist"):
            col = tl.index[name]
            assert len(col) == tl.n
            assert int(col.min()) >= 0, f"{name} has no record at the new t0"
        # Every pane's first record sits at the ACTIVE edge, not at the file's
        # first sample: the state index has skipped the setup too.
        assert int(tl.index["state"][0]) > 0
        assert int(tl.index["lwrist"][0]) > 0
    finally:
        rollout.close()


# --- scene ------------------------------------------------------------------


def test_qpos_addresses_are_nothing_like_the_lowstate_order(record):
    """The trap this whole module is built to avoid.

    The humanoid MJCF happens to be nq=29 in exactly the lowstate order, which
    made `qpos[:] = q` look correct. Nothing about that survives on the G1-D:
    its qpos opens with two wheels and two column stages, and the Dex-1's jaws
    move the arms again. A positional assignment would still run, still look
    like a robot, and put every joint on the wrong axis.
    """
    import mujoco

    model, _ = robot_scene.build_model()
    assert model.nq == 24, "2 wheels + 2 column + 2 waist + 14 arm + 4 jaws"

    adr = lambda n: int(model.jnt_qposadr[   # noqa: E731
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
    # wheels 0-1, column 2-3, bow 4, yaw 5, left arm 6-12, left jaws 13-14,
    # right arm 15-21, right jaws 22-23. Not one of them is where lowstate
    # would put it.
    assert adr("left_shoulder_pitch_joint") == 6 != 15
    assert adr("right_shoulder_pitch_joint") == 15 != 22
    assert adr("right_wrist_yaw_joint") == 21 != 28
    # And the waist names are the ones that lie about their own axes.
    assert adr("torso_Joint") == 5 and adr("Yaw_Joint") == 4


def test_the_waist_columns_drive_the_axes_the_hardware_actually_has(record):
    """Lowstate 12 is the yaw about the column, 13 is the bow, 14 has no motor.

    Mapped by physics rather than by the record's joint NAMES: a record names
    joints that exist in the humanoid MJCF (`waist_pitch_joint`) and not in this
    model. And the G1-D URDF's own names are backwards from their axes —
    `Yaw_Joint` has `axis="0 1 0"`, which is the bow.
    """
    import mujoco

    model, _ = robot_scene.build_model()
    axis = lambda n: model.jnt_axis[   # noqa: E731
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]

    assert robot_scene._WAIST_JOINTS == ("torso_Joint", "Yaw_Joint", None)
    assert np.allclose(axis("torso_Joint"), [0, 0, 1]), "slot 0 must be the yaw"
    assert np.allclose(axis("Yaw_Joint"), [0, 1, 0]), "slot 1 must be the bow"


def test_the_marker_is_the_virtual_tcp_not_the_flange(record):
    """What the sphere marks, and why it is 18.5 cm from where the arm ends.

    The Dex-1's TCP is the midpoint of the jaw tips (the robot file's
    `[end_effector] tcp`). Marking the flange would put the sphere behind the
    point the gripper acts at.
    """
    import mujoco

    rollout = loader.load(record)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        model, data = robot_scene.build_model()

        # The offset decomposes as the deploy stack documents it: the gripper's
        # own mount, plus the base-to-fingertip-midpoint distance.
        from g1d_deploy.robot import g1d_model as g1d

        assert robot_scene.FLANGE_TO_TCP_X == g1d.FLANGE_TO_TCP_X
        assert robot_scene.FLANGE_TO_TCP_X - g1d.DEX1_MOUNT_X == pytest.approx(
            0.143259, abs=1e-9)

        jmap = robot_scene._JointMap(model, rollout.model)
        for f in (0, tl.n // 2, tl.n - 1):
            # `tl.q`, not `state["q"][tl.at("state", f)]`. The scene is posed
            # from the state INTERPOLATED to the frame instant, so re-posing
            # from the held record here would compare two different postures
            # and read the millimetre between them as a broken TCP offset.
            q = tl.q[f]
            data.qpos[:] = 0.0
            data.qpos[jmap.arm_adr] = q[jmap.arm_cols]
            data.qpos[jmap.waist_adr] = q[jmap.waist_cols]
            mujoco.mj_forward(model, data)
            for hand, bid in enumerate(jmap.wrist_body):
                # Exactly FLANGE_TO_TCP_X ahead of the wrist origin, along that
                # frame's own +X — a rigid offset, not a world-axis one.
                offset = sc.tcp[f, hand] - data.xpos[bid]
                local = data.xmat[bid].reshape(3, 3).T @ offset
                assert local[0] == pytest.approx(robot_scene.FLANGE_TO_TCP_X, abs=1e-5)
                assert abs(local[1]) < 1e-5 and abs(local[2]) < 1e-5
    finally:
        rollout.close()


def test_world_from_base_recovers_the_frame_the_record_was_written_in(record):
    """The transform that makes the commanded overlay mean anything.

    The record stores poses in the pelvis frame; the viewer draws in the G1-D's
    world. `_PLANTED_T` is the relationship the synthetic record was generated
    under, so recovering it is a test with a known answer rather than a
    self-consistency check that would pass on a wrong constant.

    It also pins the derivation itself: the arm has to cancel between the two
    sides. If the joint map ever drifts, the recovered transform stops matching.
    """
    import mujoco

    rollout = loader.load(record)
    try:
        model, data = robot_scene.build_model()
        jmap = robot_scene._JointMap(model, rollout.model)
        got = scene.world_from_base(rollout, model, data, jmap)

        assert np.allclose(got[:3, 3], _PLANTED_T[:3, 3], atol=1e-9)
        assert np.allclose(got[:3, :3], _PLANTED_T[:3, :3], atol=1e-9)
        # And it is a proper rotation, not an elementwise average of several.
        assert np.allclose(got[:3, :3] @ got[:3, :3].T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(got[:3, :3])) == pytest.approx(1.0)
        _ = mujoco
    finally:
        rollout.close()


def test_the_robot_has_no_legs_and_does_have_wheels_and_a_column(record):
    """The whole reason for switching models. The humanoid MJCF drew twelve leg
    joints whose absent motors report a hard zero; this one has the base and the
    lift column the machine actually has."""
    rollout = loader.load(record)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        parts = {g["part"] for g in sc.geoms}
        assert "leg" not in parts
        assert {"body", "gripper", "wheel", "column"} <= parts
        assert sum(g["part"] == "wheel" for g in sc.geoms) == 2
        # 3 per hand: base plus two jaws. The base arrives merged onto the wrist
        # body by MuJoCo's fixed-joint handling, so it is caught by mesh name.
        assert sum(g["part"] == "gripper" for g in sc.geoms) == 6
        # The three-finger hand the G1-D ships with is stripped, not drawn.
        assert not any("_hand_" in g["body"] for g in sc.geoms)
    finally:
        rollout.close()


def test_the_jaws_follow_the_measured_gripper(tmp_path):
    """`grip_meas` dim 0 of each hand is the Dex-1's jaw radian. The map to
    metres is uncalibrated and linear (see the dex1 PROVENANCE.md), so what is
    pinned here is that it MOVES and stays inside the joint's travel."""
    closed = _write_record(tmp_path / "closed")
    rollout = loader.load(closed)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        # Left is at 1.0 rad and right at 4.0 of a 0..5.4 travel, so the two
        # hands must not land in the same place.
        left = [g for g in sc.geoms if "left_dex1_finger_link_1" in g["body"]]
        right = [g for g in sc.geoms if "right_dex1_finger_link_1" in g["body"]]
        assert left and right
        li, ri = sc.geoms.index(left[0]), sc.geoms.index(right[0])
        assert not np.allclose(sc.transforms[0, li, :3], sc.transforms[0, ri, :3])
    finally:
        rollout.close()


# --- The commanded ghost ----------------------------------------------------


@pytest.fixture
def commanded(tmp_path):
    return _write_record(tmp_path / "20260826-130000", commands=True)


def test_the_ghost_lands_on_the_pose_the_policy_actually_commanded(commanded):
    """The overlay's one substantive claim, against a planted answer.

    `chunk["pose"]` is the commanded FLANGE in pelvis coordinates. Pushed through
    `T_world_pelvis` and offset by the same rigid `FLANGE_TO_TCP_X`, it must be
    where the ghost's virtual TCP is drawn. `_CMD_POSE` is deliberately nowhere
    near the measured arm, so an overlay that quietly tracked the measured
    gripper — the easy bug, and the one that looks perfectly plausible on screen
    because the ghost would sit neatly on the robot — cannot pass this.
    """
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert sc.ghost_valid.any(), "the record carries commands; some frame must draw"

        local = np.array([robot_scene.FLANGE_TO_TCP_X, 0.0, 0.0])
        for f in np.flatnonzero(sc.ghost_valid):
            for h in (0, 1):
                T = _PLANTED_T @ scene._pose7_to_mat(_CMD_POSE[h])
                want = T[:3, 3] + T[:3, :3] @ local
                assert np.allclose(sc.ghost_tcp[f, h], want, atol=1e-5)
                # And it is genuinely elsewhere than the measured marker, so the
                # two spheres are reporting different things.
                assert np.linalg.norm(sc.ghost_tcp[f, h] - sc.tcp[f, h]) > 0.01
    finally:
        rollout.close()


def test_the_ghost_is_only_the_gripper(commanded):
    """"only the dex-1 and not the robot's body". Six geoms, three per hand, and
    every one of them a Dex-1 part — a ghost that dragged an arm link along would
    draw a second floating forearm in mid-air."""
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert len(sc.ghost_geoms) == 6
        assert sc.ghost.shape == (tl.n, 6, 12)
        assert all(sc.geoms[i]["part"] == "gripper" for i in sc.ghost_geoms)
        sides = [sc.geoms[i]["body"].split("_")[0] for i in sc.ghost_geoms]
        assert sides.count("left") == 3 and sides.count("right") == 3
        # `meta()` names them so the browser reuses the meshes it already has
        # rather than being sent the same triangles a second time.
        assert sc.meta()["ghost_geoms"] == sc.ghost_geoms
        assert sc.meta()["ghost_frames"] == int(sc.ghost_valid.sum())
    finally:
        rollout.close()


def test_the_ghost_hides_rather_than_holds_where_nothing_was_commanded(commanded):
    """Homing, warmup and the wait at READY have no Cartesian target behind them.
    Drawing a ghost there would invent a goal nobody commanded, and it would be
    indistinguishable on screen from one that was.

    After the LAST deadline the overlay does hold, and that is deliberate rather
    than an oversight: the firmware runs the last command it received forever,
    so the robot really is still being commanded to that pose. The
    zero-order hold is reporting a fact, not padding.
    """
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        valid = sc.ghost_valid.astype(bool)
        assert not valid[0], "nothing is commanded before the first deadline"
        assert valid[-1], "the last command stays in force"
        # Contiguous once it starts: this record has one chunk, so the covered
        # window is one run. A gap would mean a tick resolved to no chunk or two.
        f = np.flatnonzero(valid)
        assert np.array_equal(f, np.arange(f[0], len(valid)))
        assert f[0] > 0
        # Before it starts, zeros rather than stale geometry.
        assert not sc.ghost[:f[0]].any()
        assert not sc.ghost_tcp[:f[0]].any()
    finally:
        rollout.close()


def test_a_tick_from_P0s_ramp_is_recorded_as_undefined_and_is_not_drawn(tmp_path):
    """P0's homing and exit ramps are joint-space: no Cartesian target exists.

    The ring records that as NaN rather than as a zero pose, so the viewer READS
    "there was no target here" instead of inferring it from `k < 1`. The two are
    not the same claim — `k` says who is expected to have written a tick, NaN
    says what was actually written — and a zero-filled pose would have drawn a
    ghost at the base frame's origin, which looks like a real commanded pose.
    """
    run = _write_record(tmp_path / "ramp", commands=True)
    from g1d_deploy.runtime.processes import p9_record as p9

    body = np.fromfile(run / "targets.bin", dtype=p9.TARGET_REC)
    # `k` is left as P6 wrote it — 1..40, ordinary ticks — precisely so that `k`
    # cannot be what excludes them. Only the NaN can.
    body["err"] = np.nan
    body["pose"] = np.nan
    body.tofile(run / "targets.bin")

    rollout = loader.load(run)
    try:
        assert rollout.has_recorded_pose
        sc = scene.build(rollout, timeline.build(rollout))
        assert not sc.ghost_valid.any()
    finally:
        rollout.close()


def test_the_ghost_reads_the_pose_P6_recorded_rather_than_the_chunk_ring(tmp_path):
    """The point of recording `body_targets["pose"]`: it is the array P6 handed
    the IK, so nothing has to be reconstructed.

    Proved by making the two sources DISAGREE — the chunks are overwritten with a
    pose the policy never commanded. A viewer still reading the chunk ring draws
    the chunk's pose and passes every other test in this file; only this one
    fails. That is exactly the async failure mode, where the chunk ring cannot
    say what the blend produced.
    """
    run = _write_record(tmp_path / "src", commands=True)
    chunk_dtype = np.dtype([tuple(f) for f in json.loads(
        (run / "manifest.json").read_text())["streams"]["chunks"]["descr"]])
    chunks = np.fromfile(run / "chunks.bin", dtype=chunk_dtype)
    chunks["pose"][...] = 0.0
    chunks["pose"][..., 0] = 9.9              # nowhere the arm could be
    chunks.tofile(run / "chunks.bin")

    rollout = loader.load(run)
    try:
        assert rollout.has_recorded_pose
        assert rollout.summary()["command_source"] == "recorded"
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert sc.ghost_valid.any()

        local = np.array([robot_scene.FLANGE_TO_TCP_X, 0.0, 0.0])
        f = int(np.flatnonzero(sc.ghost_valid)[0])
        for h in (0, 1):
            T = _PLANTED_T @ scene._pose7_to_mat(_CMD_POSE[h])
            want = T[:3, 3] + T[:3, :3] @ local
            assert np.allclose(sc.ghost_tcp[f, h], want, atol=1e-5)
    finally:
        rollout.close()


def test_async_overlap_is_drawn_when_the_pose_was_recorded(tmp_path):
    """The whole reason for the change.

    Two overlapping chunks is async: every tick has two contributors and the
    commanded pose is `chunk_blend.apply(fn, both)`. With the pose recorded that is
    simply read, and the overlay works in async exactly as it does in sync — no
    blend replay, no ring-residency model, no mode distinction anywhere in this
    package.
    """
    run = _write_record(tmp_path / "async", commands=True, n_chunks=2)
    rollout = loader.load(run)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert sc.ghost_valid.any(), "async must draw, not decline"

        local = np.array([robot_scene.FLANGE_TO_TCP_X, 0.0, 0.0])
        f = int(np.flatnonzero(sc.ghost_valid)[0])
        for h in (0, 1):
            T = _PLANTED_T @ scene._pose7_to_mat(_CMD_POSE[h])
            want = T[:3, 3] + T[:3, :3] @ local
            assert np.allclose(sc.ghost_tcp[f, h], want, atol=1e-5)
    finally:
        rollout.close()


def test_a_legacy_record_reconstructs_in_sync_and_declines_in_async(tmp_path):
    """Records written before `pose` existed, both modes.

    Sync is exact: `n_actions == P`, so chunks partition the tick line and the
    single covering chunk is the only contributor there could have been. Async is
    not recoverable at all — which chunks were resident when P6 asked depends on
    an instant no record contains — so it declines rather than picking one of the
    two overlapping chunks and presenting the guess as the commanded pose.
    """
    sync = loader.load(_write_record(tmp_path / "old_sync", commands=True,
                                     record_pose=False))
    try:
        assert not sync.has_recorded_pose
        assert sync.summary()["command_source"] == "chunks"
        sc = scene.build(sync, timeline.build(sync))
        assert sc.ghost_valid.any(), "sync reconstructs exactly"
    finally:
        sync.close()

    dual = loader.load(_write_record(tmp_path / "old_async", commands=True,
                                     record_pose=False, n_chunks=2))
    try:
        assert not dual.has_recorded_pose
        sc = scene.build(dual, timeline.build(dual))
        assert not sc.ghost_valid.any(), (
            "two contributors and no recorded blend output: the honest answer "
            "is to draw nothing")
    finally:
        dual.close()


def test_the_ghost_jaws_follow_the_command_not_the_measurement(commanded):
    """The ghost is "commanded by the commanded gripper width". `_CMD_JAW` is
    (3.1, 0.7) against a measured (1.0, 4.0) — note the two hands are ORDERED
    oppositely, so an overlay reading `grip_meas` by mistake gives the ghost's
    left jaw the right hand's aperture and still looks like a moving gripper."""
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        f = int(np.flatnonzero(sc.ghost_valid)[0])

        def jaw_gap(arr, geoms, side):
            fingers = [i for i in geoms
                       if sc.geoms[i]["body"].startswith(side)
                       and "finger" in sc.geoms[i]["body"]]
            a, b = (geoms.index(i) for i in fingers)
            return float(np.linalg.norm(arr[f, a, :3] - arr[f, b, :3]))

        ghost_l = jaw_gap(sc.ghost, sc.ghost_geoms, "left")
        ghost_r = jaw_gap(sc.ghost, sc.ghost_geoms, "right")
        # Commanded left (3.1) is wider than commanded right (0.7)...
        assert ghost_l > ghost_r
        # ...while MEASURED left (1.0) is narrower than measured right (4.0).
        # So the ghost cannot be reading the measurement.
        all_geoms = list(range(len(sc.geoms)))
        assert (jaw_gap(sc.transforms, all_geoms, "left")
                < jaw_gap(sc.transforms, all_geoms, "right"))
    finally:
        rollout.close()


def test_a_record_with_no_command_streams_still_loads_and_draws_the_robot(record):
    """Every 2026-08-19 and -08-20 rollout predates these streams, and a viewer
    that refused them would be unable to open most of the archive. The robot
    draws; the overlay is simply empty."""
    rollout = loader.load(record)
    try:
        assert not rollout.has_commands
        sc = scene.build(rollout, timeline.build(rollout))
        assert not sc.ghost_valid.any()
        assert sc.transforms.any(), "the robot itself must still be posed"
    finally:
        rollout.close()


# --- The solved overlay -----------------------------------------------------
#
# The joint angles P6 published, posed as a robot. It is the middle term of the
# diagnosis the viewer exists for: the ghost is what the policy ASKED for, this
# is what the IK SOLVED, and the opaque arm is where the robot GOT TO. Confusing
# it with either neighbour is the failure these tests are for, and both
# confusions draw a picture that looks entirely reasonable.


def _solved_tcp(q_body, jaws=_CMD_JAW, waist=None):
    """Where a body target's `q` puts the two virtual TCPs, computed here.

    Independently of `scene.build`, and by the same route `_fill_flange` takes:
    the viewer's own model, posed by NAME, read at the wrist bodies. That makes
    the assertion a comparison against an answer this file derived rather than
    against the code under test restated.

    `waist` overrides the trunk the arm hangs off, which is the whole subject of
    the overlay's waist switch: the same fourteen arm angles on a different
    trunk are a different robot. Default is `q_body`'s own three, the waist the
    tick COMMANDED; pass `_CMD_WAIST_IK` for the one the IK was pinned at.
    """
    import mujoco

    from g1d_deploy.robot import g1d_model as g1d

    model, data = robot_scene.build_model()
    jmap = robot_scene._JointMap(model, _FACTS)
    data.qpos[:] = 0.0
    data.qpos[jmap.arm_adr] = q_body[:14]
    # The waist mapping SPELLED OUT rather than taken from `jmap.waist_slots`,
    # which is the thing under test: a body target's `q[14:17]` is ordered by
    # waist SLOT, and slots 0 and 1 are the two the G1-D has a motor for. Borrow
    # the production mapping here and a mix-up with `waist_cols` — which
    # addresses the 29-wide measured state, and would read two ARM columns —
    # would agree with itself and pass.
    w = q_body[14:17] if waist is None else np.asarray(waist, dtype=float)
    data.qpos[jmap.waist_adr] = [w[0], w[1]]
    span = 5.4
    for side, jaw in zip(("left", "right"), jaws):
        frac = min(1.0, max(0.0, jaw / span))
        data.qpos[jmap.jaw_adr[side]] = (
            g1d.JAW_CLOSED + frac * (g1d.JAW_OPEN - g1d.JAW_CLOSED))
    mujoco.mj_forward(model, data)
    local = np.array([robot_scene.FLANGE_TO_TCP_X, 0.0, 0.0])
    return np.array([data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ local
                     for bid in jmap.wrist_body])


def test_the_solved_overlay_is_posed_at_the_angles_P6_published(commanded):
    """The overlay's one substantive claim, against a planted answer.

    `_CMD_Q` moves three joints the measured posture also moves, to different
    values, so an overlay that quietly posed `state["q"]` — the easy bug, and
    the one that looks perfect on screen because the violet arm would sit
    exactly on the cyan one — cannot pass.
    """
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert sc.solved_valid.any()

        want = _solved_tcp(_CMD_Q)
        for f in np.flatnonzero(sc.solved_valid):
            assert np.allclose(sc.solved_tcp[f], want, atol=1e-5)
            # And genuinely elsewhere than the measured arm, both hands.
            for h in (0, 1):
                assert np.linalg.norm(sc.solved_tcp[f, h] - sc.tcp[f, h]) > 0.01
    finally:
        rollout.close()


def test_the_solved_overlay_is_posed_on_BOTH_trunks(commanded):
    """The waist switch, as the two answers it picks between.

    A body target says two things about the waist. `q[14:17]` is the one the
    tick commanded, which is what P7 drove; `waist_ik` is the one P6 pinned the
    model at to solve for `q[:14]`, which is the trunk the arm was really
    hanging off. The arm angles are identical in both, so the ONLY difference on
    screen is the trunk under them — and it moves the wrists, because everything
    above the waist rides on it.
    """
    rollout = loader.load(commanded)
    try:
        assert rollout.has_pinned_waist
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert sc.solved_valid.any()
        assert sc.solved_pinned.shape == sc.solved.shape
        assert sc.meta()["solved_pinned"] is True

        commanded_want = _solved_tcp(_CMD_Q)
        pinned_want = _solved_tcp(_CMD_Q, waist=_CMD_WAIST_IK)
        # The two trunks are far enough apart to be worth drawing: a 0.089 rad
        # bow difference at arm's length is centimetres at the wrist.
        assert np.linalg.norm(pinned_want - commanded_want) > 0.01

        for f in np.flatnonzero(sc.solved_valid):
            assert np.allclose(sc.solved_tcp[f], commanded_want, atol=1e-5)
            assert np.allclose(sc.solved_pinned_tcp[f], pinned_want, atol=1e-5)
            # And they really are two different arrays of transforms, not one
            # array handed out twice.
            assert not np.allclose(sc.solved_pinned[f], sc.solved[f])
    finally:
        rollout.close()


def test_a_record_without_the_pinned_trunk_offers_no_second_answer(tmp_path):
    """Records written before 2026-09-03 can only pose the commanded waist.

    The pinned arrays come back EMPTY rather than as a copy of the commanded
    ones. A switch that redraws an identical picture is worse than one that is
    visibly unavailable — and the emptiness is exactly what the browser greys
    the switch on, so there is no second flag to keep in step with it.
    """
    run = _write_record(tmp_path / "20260826-150000", commands=True,
                        record_waist_ik=False)
    rollout = loader.load(run)
    try:
        assert rollout.has_recorded_pose
        assert not rollout.has_pinned_waist
        assert rollout.summary()["has_pinned_waist"] is False
        sc = scene.build(rollout, timeline.build(rollout))
        assert sc.solved_valid.any(), "the joints were recorded regardless"
        assert sc.solved_pinned.size == 0
        assert sc.solved_pinned_tcp.size == 0
        assert sc.meta()["solved_pinned"] is False
        # The commanded-waist overlay is untouched by any of this.
        want = _solved_tcp(_CMD_Q)
        f = int(np.flatnonzero(sc.solved_valid)[0])
        assert np.allclose(sc.solved_tcp[f], want, atol=1e-5)
    finally:
        rollout.close()


def test_a_tick_with_no_ik_behind_it_poses_both_overlays_the_same(tmp_path):
    """P0's ramps write NaN into `waist_ik`, and that is not missing data.

    A ramp is joint-space: there is no solve and therefore no frame anything was
    pinned in, so the waist it commanded is the only waist in the picture and
    posing it is right. The two overlays then coincide by construction rather
    than by a `k < 1` test — which matters because the solved overlay is drawn
    through the ramps, where the ghost is not.
    """
    run = _write_record(tmp_path / "ramp-waist", commands=True)
    from g1d_deploy.runtime.processes import p9_record as p9

    body = np.fromfile(run / "targets.bin", dtype=p9.TARGET_REC)
    # Only the trunk is undefined here. `pose` is left intact so this is about
    # the waist alone and not about a tick the ghost would drop anyway.
    body["waist_ik"] = np.nan
    body.tofile(run / "targets.bin")

    rollout = loader.load(run)
    try:
        assert rollout.has_pinned_waist, "the COLUMN is there; its values are not"
        sc = scene.build(rollout, timeline.build(rollout))
        assert sc.solved_valid.any()
        want = _solved_tcp(_CMD_Q)
        for f in np.flatnonzero(sc.solved_valid):
            assert np.allclose(sc.solved_pinned_tcp[f], want, atol=1e-5)
            assert np.allclose(sc.solved_pinned[f], sc.solved[f])
    finally:
        rollout.close()


def test_the_solved_overlay_covers_the_parts_a_body_target_can_move(commanded):
    """Arms, torso and grippers; not the wheels and not the lift column.

    Nothing in a body target moves those two, so a second translucent copy of
    them would sit exactly on the first and add haze rather than information.
    """
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        parts = {sc.geoms[i]["part"] for i in sc.solved_geoms}
        assert parts == {"body", "gripper"}
        assert not {"wheel", "column"} & parts
        assert sc.solved.shape == (tl.n, len(sc.solved_geoms), 12)
        assert sc.meta()["solved_geoms"] == sc.solved_geoms
        assert sc.meta()["solved_frames"] == int(sc.solved_valid.sum())
        # It really is a subset of the same geom list, so the browser draws it
        # with meshes it already holds rather than being sent them twice.
        assert all(0 <= i < len(sc.geoms) for i in sc.solved_geoms)
    finally:
        rollout.close()


def test_the_solved_overlay_is_drawn_where_the_ghost_declines(tmp_path):
    """The decoupling, as a test: an async record written before `pose` existed.

    `_poses_from_chunks` refuses such a record — two chunks overlap and the
    blend is unrecoverable — so the ghost is empty for the whole rollout. The
    joint angles were never in doubt for a moment, and the overlay that shows
    them must not be dragged down with the one that is.
    """
    run = _write_record(tmp_path / "20260826-160000", commands=True,
                        record_pose=False, n_chunks=2)
    rollout = loader.load(run)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        assert not sc.ghost_valid.any(), "async reconstruction must decline"
        assert sc.solved_valid.any(), "the joints were recorded regardless"
        want = _solved_tcp(_CMD_Q)
        f = int(np.flatnonzero(sc.solved_valid)[0])
        assert np.allclose(sc.solved_tcp[f], want, atol=1e-5)
    finally:
        rollout.close()


def test_the_solved_overlay_hides_where_no_target_was_in_force(commanded):
    """Before the first body target there is nothing to pose. Zeros, not stale
    geometry — a held pose would draw the first command across a stretch of run
    that predates it."""
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        sc = scene.build(rollout, tl)
        f = np.flatnonzero(sc.solved_valid)
        assert f[0] > 0
        assert not sc.solved[:f[0]].any()
        assert not sc.solved_tcp[:f[0]].any()
    finally:
        rollout.close()


# --- The wire ---------------------------------------------------------------


def _loaded(run):
    """A `server.Loaded` over a record, without starting an HTTP server."""
    from g1d_deploy.dashboards.inspect import server

    rollout = loader.load(run)
    tl = timeline.build(rollout)
    return server.Loaded(rollout, tl, scene.build(rollout, tl),
                         ("e", "s", "t"), events.build(rollout, tl)), rollout


def test_every_binary_section_starts_where_the_layout_says_and_is_4_aligned(
        commanded):
    """The browser builds its views with `new Float32Array(buffer, byteOffset,
    length)`, which THROWS on a byteOffset that is not a multiple of 4.

    The validity columns are uint8 and a rollout's frame count is not generally
    a multiple of four, so a float section packed straight after one lands on a
    live grenade: it fails at load time, on some rollouts and not others, with
    the frame count as the only clue. Asserted rather than left to the ordering
    of the sections, which the next person to add one will not think about.
    """
    cur, rollout = _loaded(commanded)
    try:
        for endpoint, payload in (("scene_bin", cur.scene_bin()),
                                  ("transforms_bin", cur.transforms_bin())):
            spans = cur.describe()["layout"][endpoint]
            assert spans, endpoint
            for name, (off, nbytes) in spans.items():
                assert off % 4 == 0, f"{endpoint}.{name} starts at {off}"
                assert off + nbytes <= len(payload), f"{endpoint}.{name} overruns"
            end = max(off + nb for off, nb in spans.values())
            # The payload is exactly the sections plus their padding: a trailing
            # surplus would mean the packer wrote something the map omits.
            assert len(payload) - end < 4
    finally:
        rollout.close()


def test_the_layout_addresses_the_arrays_it_names(commanded):
    """Reading each span back must give the array the server holds. The failure
    this catches renders — every geom drawn with some other geom's transform is
    still a robot, just not this one's."""
    cur, rollout = _loaded(commanded)
    try:
        payload = cur.transforms_bin()
        spans = cur.describe()["layout"]["transforms_bin"]
        for name, want in (("solved", cur.scene.solved),
                           ("solved_tcp", cur.scene.solved_tcp),
                           ("solved_pinned", cur.scene.solved_pinned),
                           ("solved_pinned_tcp", cur.scene.solved_pinned_tcp),
                           ("ghost", cur.scene.ghost)):
            off, nbytes = spans[name]
            got = np.frombuffer(payload, want.dtype, nbytes // want.itemsize, off)
            assert np.array_equal(got, want.reshape(-1), equal_nan=True), name
        off, nbytes = spans["solved_valid"]
        assert np.array_equal(
            np.frombuffer(payload, np.uint8, nbytes, off), cur.scene.solved_valid)
    finally:
        rollout.close()


def test_a_record_with_no_pinned_trunk_still_names_the_empty_sections(tmp_path):
    """The absent overlay is a span of zero bytes, not an absent key.

    The browser builds every view unconditionally, so a missing name would be a
    `TypeError` at load rather than a greyed-out switch. It also still has to be
    4-aligned: `new Float32Array(buffer, off, 0)` throws on a bad offset even
    when it is asked for nothing.
    """
    run = _write_record(tmp_path / "20260826-140000", commands=True,
                        record_waist_ik=False)
    cur, rollout = _loaded(run)
    try:
        spans = cur.describe()["layout"]["transforms_bin"]
        for name in ("solved_pinned", "solved_pinned_tcp"):
            off, nbytes = spans[name]
            assert nbytes == 0, name
            assert off % 4 == 0, name
            assert off <= len(cur.transforms_bin()), name
    finally:
        rollout.close()


# --- The replan marks -------------------------------------------------------


def test_the_marks_are_the_trigger_and_the_ring_write(commanded):
    """One replan, two instants, and the frames a click on each would land on.

    The instants are the ones the timeline can be quoted against: `t_trigger`
    is when P5 woke, `t_chunk_written` is when the finished chunk reached the
    ring, and everything between them is the arm running on the previous chunk.
    """
    rollout = loader.load(commanded)
    try:
        tl = timeline.build(rollout)
        marks = events.build(rollout, tl)
        assert len(marks.items) == 1
        r = marks.items[0]
        assert r.latency == pytest.approx(_INFER_S, abs=1e-6)

        j = marks.as_json(tl)["replans"][0]
        assert j["latency_ms"] == pytest.approx(_INFER_S * 1000, abs=0.01)
        # Both are absolute instants re-expressed on the scrubber's axis, so
        # the frame of each is the frame the viewer lands on when it is clicked.
        assert j["trigger_frame"] == tl.frame_at(r.trigger)
        assert j["written_frame"] == tl.frame_at(r.written)
        assert j["written_frame"] > j["trigger_frame"], (
            "90 ms is nearly three frames at 30 fps; the two marks are not the "
            "same stop on the timeline")
    finally:
        rollout.close()


def test_a_frame_is_the_nearest_one_not_the_one_before(record):
    """`frame_at` rounds; `_hold` floors. Both are right for their own question.

    A mark is asking "where is this instant on the scrubber", and flooring
    there lands up to a whole frame early — reliably showing the frame before
    the one the mark refers to, which is exactly the frame in which the thing
    marked had not happened yet.
    """
    rollout = loader.load(record)
    try:
        tl = timeline.build(rollout)
        just_after = tl.t[10] + 0.9 / tl.fps
        assert tl.frame_at(just_after) == 11
        assert tl.frame_at(tl.t[10] + 0.1 / tl.fps) == 10
        # Clamped rather than raising or wrapping: an instant outside the window
        # is a real thing to ask about and -1 would read as the last frame.
        assert tl.frame_at(tl.t0 - 5.0) == 0
        assert tl.frame_at(tl.t1 + 5.0) == tl.n - 1
    finally:
        rollout.close()


def test_coverage_is_the_ticks_that_ran_not_k_over_the_control_rate(tmp_path):
    """The claim the whole module rests on, against a record that stalls.

    `k / control_hz` looks like the way to place a chunk's ticks on the
    timeline, and it is wrong on every real rollout: `k` jumps by 2 and the wall
    clock by ~149 ms at each sync boundary, and by 2 across a 59 s READY wait.
    Here a 150 ms hitch is planted mid-chunk, and coverage must follow the
    recorded deadlines through it rather than a straight line drawn over it.
    """
    hitch = 0.150
    run = _write_record(tmp_path / "hitch", commands=True, tick_hitch=hitch)
    rollout = loader.load(run)
    try:
        tl = timeline.build(rollout)
        r = events.build(rollout, tl).items[0]
        deadlines = rollout.body_targets["deadline"]
        assert r.cover[0] == pytest.approx(deadlines[0])
        # The chunk covers every recorded tick, so its span ends one nominal
        # period past the last one — and carries the hitch, which a fitted line
        # would have averaged away into a slightly wrong period everywhere.
        assert r.cover[1] == pytest.approx(deadlines[-1] + 1.0 / 30.0)
        naive = (r.k0 + r.n) / 30.0 - r.k0 / 30.0
        assert r.cover[1] - r.cover[0] == pytest.approx(naive + hitch, abs=1e-6)
    finally:
        rollout.close()


def test_sync_coverage_tiles_and_async_coverage_stacks(tmp_path):
    """Lanes exist for async and cost sync nothing.

    In sync consecutive chunks partition the tick line and share an endpoint,
    which must stay one row — a second lane there would be a claim of overlap
    that the sync invariant says cannot happen. In async the chunks genuinely overlap, and
    drawing them on one row would paint one over the other.
    """
    sync = loader.load(_write_record(tmp_path / "sync", commands=True))
    try:
        tl = timeline.build(sync)
        assert events.build(sync, tl).as_json(tl)["lanes"] == 1
    finally:
        sync.close()

    dual = loader.load(_write_record(tmp_path / "async", commands=True,
                                     n_chunks=2))
    try:
        tl = timeline.build(dual)
        marks = events.build(dual, tl)
        assert marks.as_json(tl)["lanes"] == 2
        assert {r.lane for r in marks.items} == {0, 1}
        a, b = marks.items
        assert a.cover == b.cover, "this record's async chunks do overlap"
    finally:
        dual.close()


def test_a_warmup_replan_is_dropped_rather_than_pinned_to_the_start(tmp_path):
    """P5's warm-up inferences are not the first replan of the rollout.

    They happen before ACTIVE, their action range is P0's homing ticks, and none
    of it ever ran. Clamped onto the window they would draw a mark at the very
    start of the bar, which reads as "the rollout began by replanning here" —
    the one thing that did not happen.
    """
    run = _write_record(tmp_path / "warm", commands=True, warmup_chunk=True,
                        session=True)
    rollout = loader.load(run)
    try:
        tl = timeline.build(rollout)
        assert tl.window == "active"
        marks = events.build(rollout, tl)
        assert [r.chunk_id for r in marks.items] == [1], (
            "chunk 99 is the warm-up: before the window, and covering ticks "
            "that never executed")
    finally:
        rollout.close()


def test_a_record_with_no_chunk_stream_has_no_marks(record):
    """An empty strip has to mean "nothing recorded", never "nothing happened".

    Neither is an error — the run is perfectly viewable — so this returns no
    marks and the UI says so, rather than refusing the record or showing a bar
    that looks like a rollout that never replanned.
    """
    rollout = loader.load(record)
    try:
        tl = timeline.build(rollout)
        j = events.build(rollout, tl).as_json(tl)
        assert j["n"] == 0 and j["replans"] == []
        assert j["latency_ms"] is None and j["lanes"] == 1
    finally:
        rollout.close()
