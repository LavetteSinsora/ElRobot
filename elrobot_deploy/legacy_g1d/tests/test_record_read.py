"""Reading a rollout record, and the ramp-vs-step verdict.

Records are synthesised here rather than captured, because what is under test is
the READER: given a body-target ramp and a measured trajectory, does it say the
right thing? Feeding it a real record would test the recorder too, and the two
failures would be indistinguishable.

The step case is the one that matters. It is the shape the operator reported on
the first run, and a diagnostic that cannot tell it from a ramp is worse than no
diagnostic — it would have confirmed the wrong thing.
"""

import json

import numpy as np
import pytest

from elrobot_deploy.rollout_record_utils import reader as record_read
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.processes.p9_record import TARGET_REC as BODY_REC

N_TICKS = 90
DT = 1.0 / 30.0
T0 = 1000.0


def _write_record(tmp_path, measured_q, *, joints=True, n_state=300):
    """A record with a 90-tick homing ramp and a given measured trajectory.

    `measured_q` is a callable: normalised time in [0, 1] -> fraction of the
    travel completed. That is the only thing the tests vary.
    """
    body = np.zeros(N_TICKS, dtype=BODY_REC)
    q_from = np.zeros(schema.BODY_DOF)
    q_to = np.zeros(schema.BODY_DOF)
    q_to[3] = 1.0                                  # one joint travels 1 rad
    for i in range(N_TICKS):
        u = (i + 1) / N_TICKS
        body["k"][i] = i - (N_TICKS - 1)           # -89 … 0
        body["deadline"][i] = T0 + i * DT
        body["q"][i] = q_from + (q_to - q_from) * 0.5 * (1 - np.cos(np.pi * u))
    (tmp_path / "targets.bin").write_bytes(body.tobytes())

    state = np.zeros(n_state, dtype=schema.STATE)
    t_end = T0 + (N_TICKS - 1) * DT
    arm = list(range(schema.ARM_DOF))
    waist = [schema.ARM_DOF, schema.ARM_DOF + 1, schema.ARM_DOF + 2]
    for i in range(n_state):
        frac = i / max(1, n_state - 1)
        state["t_recv"][i] = T0 + frac * (t_end - T0)
        state["q"][i][arm[3]] = measured_q(frac)   # same joint as above
    (tmp_path / "state.bin").write_bytes(state.tobytes())

    manifest = {
        "created": "2026-08-19T19:07:35",
        "streams": {
            "targets": {"file": "targets.bin",
                        "descr": BODY_REC.descr, "records": N_TICKS},
            "state": {"file": "state.bin",
                      "descr": schema.STATE.descr, "records": n_state},
        },
        "cameras": {},
        "deploy_config": {"prompt": "p", "checkpoint": "", "sync": True,
                          "P": 25, "H": 50, "control_hz": 30.0,
                          "primary_camera": "rwrist"},
        "cpu": {}, "cpu_reserved": [],
    }
    if joints:
        manifest["joints"] = {"arm": arm, "waist": waist}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_a_cosine_ramp_reads_as_a_ramp(tmp_path):
    """Half the travel at the midpoint is the defining property of the ease."""
    _write_record(tmp_path, lambda u: 0.5 * (1 - np.cos(np.pi * u)))
    report = record_read.homing_report(record_read.load(tmp_path))
    assert "RAMP" in report
    assert "90 ticks" in report


def test_a_step_reads_as_a_step(tmp_path):
    """The reported symptom: the arm arrives almost immediately and the rest of
    the window is it sitting still."""
    _write_record(tmp_path, lambda u: 1.0 if u > 0.02 else 0.0)
    report = record_read.homing_report(record_read.load(tmp_path))
    assert "STEP" in report
    assert "RAMP" not in report


def test_an_arm_that_never_moved_says_so(tmp_path):
    """Not a failure — it is what homing looks like when the arm is already at
    home_q, which is easy to arrange right after measuring one."""
    _write_record(tmp_path, lambda u: 0.0)
    report = record_read.homing_report(record_read.load(tmp_path))
    assert "already at home_q" in report


def test_a_lagging_arm_is_not_called_a_ramp(tmp_path):
    _write_record(tmp_path, lambda u: u ** 6)
    report = record_read.homing_report(record_read.load(tmp_path))
    assert "LATE" in report


def test_an_older_record_falls_back_to_hardware_toml(tmp_path):
    """Records written before the manifest carried `joints` must still read.

    The fallback is the CURRENT robot file, which is right whenever the
    record came from this machine — and the commanded half, which needs no map
    at all, reads either way.
    """
    _write_record(tmp_path, lambda u: u, joints=False)
    rec = record_read.load(tmp_path)
    assert "joints" not in rec["manifest"]
    report = record_read.homing_report(rec)
    assert "90 ticks" in report
    assert "commanded travel" in report


def test_without_any_joint_map_the_reader_says_so(tmp_path, monkeypatch):
    """And does not silently compare columns that do not correspond."""
    _write_record(tmp_path, lambda u: u, joints=False)
    monkeypatch.setattr(record_read, "body_joint_index", lambda m: None)
    assert "no joint map" in record_read.homing_report(record_read.load(tmp_path))


def test_summarize_flags_dropped_frames(tmp_path):
    _write_record(tmp_path, lambda u: u)
    rec = record_read.load(tmp_path)
    rec["manifest"]["cameras"] = {"rwrist": {"frames": 100, "dropped": 7}}
    assert "DROPPED" in record_read.summarize(rec)


def test_no_homing_ticks_is_explained_not_crashed(tmp_path):
    _write_record(tmp_path, lambda u: u)
    rec = record_read.load(tmp_path)
    # Targets exist, but all of them belong to the run rather than to homing —
    # what a record started after HOME looks like.
    body = rec["body_targets"].copy()
    body["k"] += 100
    rec["body_targets"] = body
    assert "not recorded" in record_read.homing_report(rec)


def test_a_one_tick_ramp_is_called_a_step(tmp_path):
    """`cosine_ramp` floors at one sample, so this is reachable — and a ramp of
    one tick is a step whatever the deadlines say."""
    rec = record_read.load(_write_record(tmp_path, lambda u: u))
    rec["body_targets"] = rec["body_targets"][rec["body_targets"]["k"] == 0]
    assert "step by construction" in record_read.homing_report(rec)


@pytest.mark.parametrize("stream", ["targets", "state"])
def test_a_missing_stream_does_not_crash_the_reader(tmp_path, stream):
    _write_record(tmp_path, lambda u: u)
    (tmp_path / f"{stream}.bin").unlink()
    rec = record_read.load(tmp_path)
    assert isinstance(record_read.homing_report(rec), str)
