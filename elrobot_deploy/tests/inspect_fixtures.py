"""A g1d_deploy rollout record, written the way P9 writes one.

The manifest is built from the real resolution path — `deploy_file.resolve` of
one of the example deploy files, `rollout_record_utils.manifest.build` for the robot file chain,
the policy entry and the revisions, `p9_record._jsonable` for the config dump —
so the inspection code is tested against the grammar the recorder actually
produces rather than a hand-kept copy of it. Only the streams are synthetic:
the robot stands at its home pose, the head camera runs at 30 Hz, and every
replan's anchor is a real head frame.

`intervene` marks tick ranges as an operator's (odd `intervene` counter, a
finite EEF pose even in a joint-mode rollout), which is what a PICO takeover
leaves in the record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.processes import p9_record as p9

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

#: Replans in the synthetic rollout, and when the first one fires.
N_CHUNKS = 4
T0 = 5000.0
LEAD_S = 0.5


def _jpeg(h: int, w: int) -> bytes:
    import cv2

    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2, 2] = 200               # the left eye red (BGR), so a flip shows
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _descr(dt):
    return p9._jsonable(dt.descr)


def write(run: Path, *, example: str = "sine_eef", intervene=(),
          n_chunks: int = N_CHUNKS, robot_edit=None, manifest_edit=None,
          command_offset: float | None = None) -> Path:
    """Write a record into `run`; return it.

    `command_offset` adds P7's `command` stream (500 Hz over the ticks): the
    home arm plus that many radians on every arm joint, so the servo tracking
    error against the measured (home) arm is exactly `command_offset`.

    `robot_edit(text) -> text` rewrites the LAST robot file in the recorded
    chain (the profile the deploy file named), which is how a test gives the
    record a robot that differs from the shipped one.
    """
    from g1d_deploy.config_loading import deploy_file as deployfile
    from g1d_deploy.rollout_record_utils import manifest as provenance
    from g1d_deploy.robot.kinematics import ForwardKinematics
    from g1d_deploy.config_loading.rollout_settings import RolloutSettings

    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    r = deployfile.resolve(EXAMPLES / "policies" / f"{example}.toml", record=False)
    cfg, hw = r.cfg, r.hw
    dt, P, H = cfg.dt, cfg.P, cfg.H
    joint = example == "sine_joint"

    # --- state: standing at home -------------------------------------------
    n_state = int((LEAD_S + (n_chunks + 2) * P * dt) * 1000)
    state = np.zeros(n_state, dtype=schema.STATE)
    state["index"] = np.arange(n_state)
    state["t_recv"] = T0 + np.arange(n_state) * 1e-3
    q = np.zeros(schema.N_JOINTS)
    q[hw.joints_arm] = cfg.home_q
    q[hw.joints_waist[1]] = 0.039            # the bow's resting droop
    state["q"] = q
    waist = q[hw.joints_waist]
    flange = ForwardKinematics(hw.urdf).flange(cfg.home_q, waist).copy()
    state["flange"] = flange
    state["grip_meas"][:, 0] = 2.0
    state["grip_meas"][:, 6] = 2.0
    state.tofile(run / "state.bin")
    t_end = float(state["t_recv"][-1])

    # --- cameras: every robot-file camera, head at 30 Hz --------------------
    cams = {}
    head_t = None
    for name, spec in sorted(cfg.cameras.items()):
        payload = _jpeg(spec.height, spec.width) if name == "head" else _jpeg(8, 8)
        n_f = int((t_end - T0) * spec.hz)
        idx = np.zeros(n_f, dtype=p9.FRAME_IDX)
        idx["frame_id"] = np.arange(1, n_f + 1)
        idx["t_recv"] = T0 + 0.003 + np.arange(n_f) / spec.hz
        idx["t_observed"] = idx["t_recv"] + 0.002
        idx["offset"] = np.arange(n_f) * len(payload)
        idx["nbytes"] = len(payload)
        idx.tofile(run / f"{name}.idx.bin")
        (run / f"{name}.jpg.bin").write_bytes(payload * n_f)
        cams[name] = {"blob": f"{name}.jpg.bin", "index": f"{name}.idx.bin",
                      "index_descr": _descr(p9.FRAME_IDX), "frames": n_f,
                      "dropped": 0}
        if name == "head":
            head_t = idx["t_recv"]

    # --- chunks: one per replan, anchored on a real head frame --------------
    chunk_dt = p9._observed(schema.chunk_dtype(H))
    chunks = np.zeros(n_chunks, dtype=chunk_dt)
    for c in range(n_chunks):
        k0 = 1 + c * P
        t_trigger = T0 + LEAD_S + (k0 - 1) * dt
        row = int(np.argmin(np.abs(head_t - t_trigger)))
        ch = chunks[c]
        ch["chunk_id"] = c + 1
        ch["k_first_action"] = k0
        ch["n_actions"] = P
        ch["n_plan"] = P
        ch["mode"] = 1 if joint else 0
        ch["t_trigger"] = t_trigger
        ch["t_anchor"] = head_t[row]
        ch["t_obs_received"] = max(t_trigger, head_t[row]) + 0.006
        ch["t_cpu_processed"] = ch["t_obs_received"] + 0.01
        ch["t_chunk_received"] = ch["t_obs_received"] + 0.02
        ch["t_chunk_written"] = ch["t_chunk_received"] + 0.001
        ch["t_observed"] = ch["t_chunk_written"]
        ch["pose"] = np.nan
        ch["arm_q"] = np.nan
        ch["waist"] = np.nan
        ch["hand"][:P, :, 0] = 2.0
        if joint:
            ch["arm_q"][:P] = cfg.home_q.reshape(2, 7)
        else:
            ch["pose"][:P] = flange
    chunks.tofile(run / "chunks.bin")

    # --- targets: one per tick ----------------------------------------------
    n_ticks = n_chunks * P
    targets = np.zeros(n_ticks, dtype=p9.TARGET_REC)
    for i in range(n_ticks):
        k = i + 1
        t = targets[i]
        t["k"] = k
        t["deadline"] = T0 + LEAD_S + k * dt
        t["q"][:14] = cfg.home_q
        t["hand"][:, 0] = 2.0
        t["waist_ik"] = waist
        t["pose"] = np.nan if joint else flange
        t["err"] = np.nan if joint else 0.001
        t["seq"] = i
        t["t_observed"] = t["deadline"]
        if any(a <= k < b for a, b in intervene):
            t["intervene"] = 1               # odd: an operator's tick
            t["pose"] = flange
    targets.tofile(run / "targets.bin")

    command = None
    if command_offset is not None:
        t_cmd = np.arange(T0 + LEAD_S, T0 + LEAD_S + (n_ticks + 1) * dt, 0.002)
        command = np.zeros(len(t_cmd), dtype=schema.COMMAND)
        command["index"] = np.arange(1, len(t_cmd) + 1)
        command["t"] = t_cmd
        command["k_exec"] = np.clip(((t_cmd - T0 - LEAD_S) / dt).astype(int),
                                    1, n_ticks)
        command["q"][:, :14] = cfg.home_q + float(command_offset)
        command["q_lin"] = command["q"]
        command.tofile(run / "command.bin")
    streams_extra = {}
    if command is not None:
        streams_extra["command"] = {"file": "command.bin",
                                    "descr": _descr(schema.COMMAND),
                                    "records": len(command)}

    session = np.zeros(1, dtype=p9.SESSION_REC)
    session["session"] = schema.ACTIVE
    session["t_observed"] = T0 + 0.1
    session.tofile(run / "session.bin")

    prov = provenance.build(r)
    if robot_edit is not None:
        files = prov["robot"]["files"]
        files[-1] = dict(files[-1], text=robot_edit(files[-1]["text"]))
    manifest = {
        "created": "2026-09-26T12:00:00",
        "streams": {
            "state": {"file": "state.bin", "descr": _descr(schema.STATE),
                      "records": n_state},
            "chunks": {"file": "chunks.bin", "descr": _descr(chunk_dt),
                       "records": n_chunks},
            "targets": {"file": "targets.bin", "descr": _descr(p9.TARGET_REC),
                        "records": n_ticks},
            "session": {"file": "session.bin", "descr": _descr(p9.SESSION_REC),
                        "records": 1},
            **streams_extra,
        },
        "cameras": cams,
        "camera_views": {
            v.name: {"source": v.source, "half": v.half, "x0": v.x0,
                     "height": v.height, "width": v.width, "slots": v.slots,
                     "history_s": v.history_s}
            for v in cfg.views.values()},
        "deploy_config": dict(p9._jsonable(cfg), P=cfg.P, H=cfg.H, dt=cfg.dt),
        "code": {"sha": "c" * 40, "dirty": False},
        "warm": True,
        "cpu": {k: (v if isinstance(v, list) else [v]) for k, v in hw.cpu.items()},
        "cpu_reserved": list(hw.cpu_reserved),
        "joints": {"arm": list(hw.joints_arm), "waist": list(hw.joints_waist)},
        "model": {"urdf": p9._model_ref(hw.urdf), "waist_joint_names": ["torso_Joint",
                                                              "Yaw_Joint", None]},
        **{k: p9._jsonable(v) for k, v in prov.items()},
        "rollouts": [{"index": 1, "dir": run.name, "prompt": cfg.prompt,
                      "generation": 0,
                      "active_config": RolloutSettings.of(cfg).as_dict(),
                      "started": "2026-09-26T12:00:00"}],
    }
    if manifest_edit is not None:
        manifest_edit(manifest)
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return run


__all__ = ["EXAMPLES", "N_CHUNKS", "write"]
