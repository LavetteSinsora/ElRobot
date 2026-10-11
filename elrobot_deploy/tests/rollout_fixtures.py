"""A tiny rollout record, written to a tmp dir.

Small enough to reason about (5 replans, two cameras, 4 s) and shaped exactly
like P9's output, so the anchor reconstruction and the record writer are tested
against the real grammar rather than against a mock of it.

The camera timings are planted, not incidental: one chunk is arranged so the
frame NEAREST the trigger had not been published yet, which is the case that
made "nearest to the trigger" wrong on the real recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.processes import p9_record

# The dtypes P9 actually writes, not the raw ring dtypes: the recorder appends
# `t_observed` to every stream and `seq` to the targets, so a fixture built from
# `schema` alone would be a record shape that has never existed on disk.
STATE_REC = schema.STATE
TARGET_REC = p9_record.TARGET_REC
SESSION_REC = p9_record.SESSION_REC
INTERVENE_REC = schema.INTERVENE
FRAME_IDX = p9_record.FRAME_IDX


def chunk_rec(horizon):
    return p9_record._observed(schema.chunk_dtype(horizon))

#: The publish latency the fixture is built with, seconds. `calibrate_lambda`
#: has to find this back.
LAMBDA = 0.005

HZ = 30.0
DT = 1.0 / HZ
P = 25
H = 50


def _descr(dt):
    out = []
    for name in dt.names:
        f = dt.fields[name][0]
        if f.subdtype is None:
            out.append([name, f.str])
        else:
            base, shape = f.subdtype
            out.append([name, base.str, list(shape)])
    return out


def _pose(x, y=0.0, z=0.9):
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def sample_jpeg() -> bytes:
    """A real JPEG, because `materialise` really decodes one.

    Four magic bytes would satisfy the blob/index arithmetic and then fail in
    `cv2.imdecode`, which is exactly the failure the solve path must not have.
    """
    try:
        import cv2
    except ImportError:                     # the record still loads without it
        return b"\xff\xd8\xff\xd9"
    img = np.zeros((16, 16, 3), np.uint8)
    img[:, :, 0] = 200                      # blue in BGR, so a flip is visible
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b"\xff\xd8\xff\xd9"


def write_record(root: Path, *, n_chunks: int = 5, cap_chunk: int = 2,
                 jpeg: bytes | None = None, swept: bool = False,
                 tick: bool = False, drift_ppm: float = 0.0,
                 stall: tuple[float, float] | None = None) -> Path:
    """Write a record; return its directory.

    `cap_chunk` is the replan whose nearer frame is hidden behind the publish
    latency — the emulation must pick the earlier frame there. `swept` drops the
    first chunks, which is what a ring that wrapped looks like.

    `tick` fills the state stream's device clock, which a record written before
    2026-09-16 does not have — off by default so that the fixture stays the
    record the rest of this suite was written against. With it, `drift_ppm` sets
    how fast the robot's counter runs against the box clock and `stall` is a
    `(start, seconds)` receive-side hole, in seconds from the window start: the
    samples produced inside it arrive in a burst at its end, exactly as a
    stalled DDS socket delivers them. The PRODUCTION instants stay uniform
    through it, which is the fact `clock.estimate` has to recover.
    """
    run = Path(root)
    run.mkdir(parents=True, exist_ok=True)
    t0 = 1000.0

    # --- state, 1 kHz ---------------------------------------------------
    n_state = int((n_chunks + 2) * P * DT * 1000)
    state = np.zeros(n_state, dtype=STATE_REC)
    state["index"] = np.arange(n_state)
    state["t_recv"] = t0 + np.arange(n_state) * 0.001
    if tick:
        # `t_recv` above is the arrival series and stays the fixture's uniform
        # 1 kHz. The device clock is derived BACKWARDS from it — production one
        # base latency earlier, on a counter running at its own rate — so a
        # record built with `tick=False` and one built with `tick=True` and no
        # stall put the state samples at very nearly the same instants, and a
        # test can compare the two clocks without the window moving under it.
        base_lat = 0.0008
        prod = state["t_recv"] - base_lat
        # The counter, in ms, running `drift_ppm` slow against the box clock:
        # this is what the robot reports, and the rate is what the fit recovers.
        state["tick"] = np.round((prod - prod[0]) * 1000.0
                                 / (1.0 + drift_ppm * 1e-6)).astype(np.uint32)
        if stall is not None:
            a, span = stall
            held = (prod - t0 >= a) & (prod - t0 < a + span)
            # Delivered in a burst at the end of the hole, 20 µs apart, which
            # keeps `t_recv` ascending while carrying no information about when
            # any of them was taken.
            state["t_recv"][held] = (t0 + a + span + base_lat
                                     + 2e-5 * np.arange(held.sum()))
    for i in range(n_state):
        state["flange"][i, 0] = _pose(0.30 + 0.0001 * i, +0.20)
        state["flange"][i, 1] = _pose(0.30 + 0.0001 * i, -0.20)
    state["grip_meas"][:, 0] = 2.0
    state["grip_meas"][:, 6] = 2.0

    # --- cameras, 30 Hz, the two wrists on slightly different phases -----
    payload = jpeg or sample_jpeg()
    cams = {}
    for name, phase, period in (("rwrist", 0.0, DT), ("lwrist", 0.004, DT)):
        n_f = int((n_chunks + 2) * P * DT / period)
        idx = np.zeros(n_f, dtype=FRAME_IDX)
        idx["frame_id"] = np.arange(1, n_f + 1)
        idx["t_recv"] = t0 + phase + np.arange(n_f) * period
        idx["t_observed"] = idx["t_recv"] + LAMBDA
        idx["offset"] = np.arange(n_f) * len(payload)
        idx["nbytes"] = len(payload)
        (run / f"{name}.jpg.bin").write_bytes(payload * n_f)
        idx.tofile(run / f"{name}.idx.bin")
        cams[name] = {
            "blob": f"{name}.jpg.bin", "index": f"{name}.idx.bin",
            "index_descr": _descr(idx.dtype), "frames": int(n_f), "dropped": 0,
        }

    # --- chunks -----------------------------------------------------------
    chunk_dt = chunk_rec(H)
    chunks = np.zeros(n_chunks, dtype=chunk_dt)
    r_idx = np.fromfile(run / "rwrist.idx.bin", dtype=FRAME_IDX)
    for c in range(n_chunks):
        k0 = 1 + c * P
        t_trigger = t0 + (k0 - 1) * DT + 0.002
        # The frame P5 could actually see. For `cap_chunk` the nearest frame is
        # still inside its decode latency, so the acquisition capped and took
        # the earlier one.
        visible = r_idx["t_recv"] + LAMBDA <= t_trigger + (0.0 if c == cap_chunk
                                                          else 0.020)
        rows = np.flatnonzero(visible)
        pick = rows[np.argmin(np.abs(r_idx["t_recv"][rows] - t_trigger))]
        chunks["chunk_id"][c] = c + 1
        chunks["k_first_action"][c] = k0
        chunks["n_actions"][c] = P
        chunks["t_trigger"][c] = t_trigger
        chunks["t_anchor"][c] = r_idx["t_recv"][pick]
        chunks["t_obs_received"][c] = max(t_trigger + 0.0005,
                                          r_idx["t_recv"][pick] + LAMBDA)
        chunks["t_cpu_processed"][c] = chunks["t_obs_received"][c] + 0.01
        chunks["t_chunk_received"][c] = chunks["t_obs_received"][c] + 0.07
        chunks["t_chunk_written"][c] = chunks["t_chunk_received"][c] + 0.001
        chunks["t_observed"][c] = chunks["t_chunk_written"][c]
        for j in range(P):
            chunks["pose"][c, j, 0] = _pose(0.30 + 0.001 * (k0 + j), +0.20)
            chunks["pose"][c, j, 1] = _pose(0.30 + 0.001 * (k0 + j), -0.20)
            chunks["hand"][c, j, :, 0] = 2.0
    if swept:
        chunks = chunks[2:]
    chunks.tofile(run / "chunks.bin")

    # --- targets, one per commanded tick ----------------------------------
    n_ticks = n_chunks * P
    targets = np.zeros(n_ticks, dtype=TARGET_REC)
    for i in range(n_ticks):
        k = 1 + i
        targets["k"][i] = k
        targets["deadline"][i] = t0 + k * DT
        targets["q"][i] = 0.0
        targets["pose"][i, 0] = _pose(0.30 + 0.001 * k, +0.20)
        targets["pose"][i, 1] = _pose(0.30 + 0.001 * k, -0.20)
        targets["hand"][i, :, 0] = 2.0
        targets["seq"][i] = i
        targets["t_observed"][i] = targets["deadline"][i]
    targets.tofile(run / "targets.bin")

    session = np.zeros(1, dtype=SESSION_REC)
    session["session"] = 5                                   # ACTIVE
    session["t_observed"] = t0
    session.tofile(run / "session.bin")
    np.zeros(0, dtype=INTERVENE_REC).tofile(run / "intervene.bin")
    state.tofile(run / "state.bin")

    manifest = {
        "created": "2026-09-16T00:00:00",
        "streams": {
            "state": {"file": "state.bin", "descr": _descr(STATE_REC),
                      "records": len(state)},
            "chunks": {"file": "chunks.bin", "descr": _descr(chunk_dt),
                       "records": len(chunks)},
            "targets": {"file": "targets.bin", "descr": _descr(TARGET_REC),
                        "records": len(targets)},
            "session": {"file": "session.bin", "descr": _descr(SESSION_REC),
                        "records": 1},
            "intervene": {"file": "intervene.bin",
                          "descr": _descr(INTERVENE_REC), "records": 0},
        },
        "cameras": cams,
        "deploy_config": {
            "prompt": "Build the box.", "checkpoint": "/runs/my_run/24000",
            "sync": True, "replan_period_ticks": P, "action_horizon": H,
            "control_hz": HZ, "dt": DT,
            "camera_slots": {"rwrist": "right_wrist_0_rgb",
                             "lwrist": "left_wrist_0_rgb"},
            "primary_camera": "rwrist", "bgr_cameras": ["rwrist", "lwrist"],
            "anchor_shift": True, "flange_to_tcp_x": 0.184759,
            "gripper_command_range": [0.0, 5.4],
            "observed_cameras": ["rwrist", "lwrist"],
            "cameras": {n: {"height": 720, "width": 1280, "source": "zmq",
                            "binocular": False, "hz": HZ, "port": None,
                            "device": None} for n in cams},
            "rig": {"wrists": {n: {"device": "x", "crop": {"x0": 0, "y0": 0,
                                                           "side": 100},
                                   "circle": [50.0, 50.0, 50.0], "after": {}}
                               for n in cams}},
            "run": {"exp_name": "my_run", "action_dim": 32},
        },
        "code": {"sha": "0" * 40, "dirty": False},
        "warm": True,
        "joints": {"arm": list(range(15, 29)), "waist": [12, 13, 14]},
        "model": {"urdf": "assets/unitree_g1d/g1_d.urdf",
                  "waist_joint_names": ["torso_Joint", "Yaw_Joint", None]},
    }
    from g1d_deploy.rollout_record_utils import manifest as provenance
    from g1d_deploy.config_loading.robot_file import load_robot
    manifest["robot"] = provenance.robot_block(
        "g1d_fisheye_dex1", load_robot("g1d_fisheye_dex1"))
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run
