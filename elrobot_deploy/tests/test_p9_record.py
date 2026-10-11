"""The recorder.

The claims worth testing are the ones that would otherwise be discovered by
finding a rollout unanalysable weeks later: that polling cannot miss a ring
write, that a re-solved tick is distinguishable from a kept one, that a dropped
frame is visible rather than silent, and that the files can be read back without
this code.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.processes import p9_record
from g1d_deploy.runtime.processes.p9_record import FRAME_IDX, Recorder


def _read(stream):
    stream.flush()
    return np.fromfile(stream.path, dtype=stream.dtype)


def test_polling_captures_every_target_write(rings, cfg, hw, tmp_path):
    """Completeness is geometry, not diligence: a slot lives 4.3 s at 128 slots
    and 30 Hz, so a sweep cannot miss a write unless it stalls for seconds."""
    rec = Recorder(tmp_path, cfg, rings)
    for k in range(1, 40):
        rings["targets"].write(
            k, 100.0 + k, np.full(schema.BODY_DOF, k),
            np.full((schema.N_HANDS, schema.HAND_DIM), k), intervene=0)
        if k % 7 == 0:
            rec.sweep()                       # sweep far slower than the writes
    rec.sweep()

    body = _read(rec.targets)
    assert sorted(body["k"].tolist()) == list(range(1, 40))
    assert np.allclose(body[body["k"] == 12]["q"], 12)
    assert _read(rec.targets).shape == body.shape
    rec.close(hw)


def test_a_resolved_tick_is_distinguishable_from_a_kept_one(rings, cfg, hw,
                                                            tmp_path):
    """In async P6 re-solves ticks after a new chunk lands. An abandoned
    re-solve leaves the PREVIOUS pass's value in the slot — valid-looking, not a
    gap — so `seq` is the only thing that separates the two."""
    rec = Recorder(tmp_path, cfg, rings)
    rings["targets"].write(5, 1.0, np.full(schema.BODY_DOF, 1.0), intervene=0)
    rec.sweep()
    rec.sweep()                               # unchanged: must not re-record
    assert _read(rec.targets).shape[0] == 1

    rings["targets"].write(5, 2.0, np.full(schema.BODY_DOF, 2.0), intervene=0)
    rec.sweep()
    rows = _read(rec.targets)
    assert rows.shape[0] == 2, "a re-solve of the same tick must be recorded"
    assert rows["seq"][1] > rows["seq"][0]
    assert rows["deadline"].tolist() == [1.0, 2.0]
    rec.close(hw)


def test_t_observed_is_separate_from_deadline(rings, cfg, hw, tmp_path):
    """`deadline` says when a target should EXECUTE; `t_observed` bounds when it
    became AVAILABLE. Without both, a target that arrived late is
    indistinguishable from one that was always going to be late."""
    rec = Recorder(tmp_path, cfg, rings)
    rings["targets"].write(3, 12345.0, np.zeros(schema.BODY_DOF), intervene=0)
    rec.sweep()
    row = _read(rec.targets)[0]
    assert row["deadline"] == 12345.0
    assert row["t_observed"] != row["deadline"]
    rec.close(hw)


def test_chunks_are_recorded_whole_with_their_metadata(rings, cfg, hw, tmp_path):
    rec = Recorder(tmp_path, cfg, rings)
    n = cfg.P
    pose = np.arange(n * 2 * 7, dtype=float).reshape(n, 2, 7)
    rings["chunks"].write(chunk_id=1, k_first_action=9, n_actions=n, pose=pose,
                          hand=np.zeros((n, 2, 6)),
                          t_trigger=5.0, t_anchor=5.05, t_obs_received=5.1,
                          t_cpu_processed=5.15, t_chunk_received=5.2,
                          t_chunk_written=5.25)
    rec.sweep()
    row = _read(rec.chunks)[0]
    assert row["chunk_id"] == 1 and row["k_first_action"] == 9
    assert row["n_actions"] == n
    assert (row["t_trigger"], row["t_chunk_written"]) == (5.0, 5.25)
    assert row["t_anchor"] == 5.05
    assert np.allclose(row["pose"][:n], pose)
    rec.close(hw)


def _write_chunk(rings, cfg, *, chunk_id, k_first_action, t_written):
    """One chunk into the ring, with only the fields these tests read."""
    n = cfg.P
    rings["chunks"].write(
        chunk_id=chunk_id, k_first_action=k_first_action, n_actions=n,
        pose=np.zeros((n, 2, 7)), hand=np.zeros((n, 2, 6)),
        t_trigger=t_written - 0.1, t_anchor=t_written - 0.09,
        t_obs_received=t_written - 0.08, t_cpu_processed=t_written - 0.07,
        t_chunk_received=t_written - 0.01, t_chunk_written=t_written)


def test_the_previous_rollouts_chunks_are_not_recorded_as_this_ones(
        rings, cfg, hw, tmp_path):
    """A session serves many rollouts and the chunk ring outlives each one.

    At the ACTIVE edge its slots still hold the last chunks of the rollout
    BEFORE this one, and `Recorder` is built fresh per rollout — so without a
    floor the first sweep writes them out as if they were ours. The 2026-09-03
    15:08 record did exactly that: 27 rows for 24 chunks, ids 7, 8 and 9 twice,
    and the viewer drew each of those ids at two different instants.
    """
    # The previous rollout: three chunks left in the ring. `chunk_slots` is 3
    # here, so these fill it exactly, which is the real situation.
    for cid, k in ((7, 151), (8, 176), (9, 201)):
        _write_chunk(rings, cfg, chunk_id=cid, k_first_action=k,
                     t_written=100.0 + cid)

    # This rollout begins. Everything already in the ring predates the floor.
    rec = Recorder(tmp_path, cfg, rings, t_floor=200.0)
    rec.sweep()
    assert _read(rec.chunks).size == 0, "leftovers were recorded as this rollout's"

    # And this rollout's own chunks are recorded normally.
    _write_chunk(rings, cfg, chunk_id=1, k_first_action=1, t_written=300.0)
    rec.sweep()
    rows = _read(rec.chunks)
    assert [int(r["chunk_id"]) for r in rows] == [1]
    rec.close(hw)


def test_a_reused_chunk_id_is_still_recorded_after_the_floor(
        rings, cfg, hw, tmp_path):
    """The floor is a TIME and not an id, and this is why.

    Chunk ids restart at 1 every rollout, so the previous rollout's chunk 7 and
    this one's land in the same slot with the same id. Keying the skip on
    `(slot, id)` would suppress the real chunk 7 — turning a duplicated row into
    a missing one, which is strictly worse. `t_chunk_written` comes off a
    monotonic clock that does not restart between rollouts.
    """
    _write_chunk(rings, cfg, chunk_id=7, k_first_action=151, t_written=100.0)

    rec = Recorder(tmp_path, cfg, rings, t_floor=200.0)
    rec.sweep()
    assert _read(rec.chunks).size == 0

    # Same id, same slot, this rollout. It must be recorded.
    _write_chunk(rings, cfg, chunk_id=7, k_first_action=151, t_written=300.0)
    rec.sweep()
    rows = _read(rec.chunks)
    assert [int(r["chunk_id"]) for r in rows] == [7]
    assert float(rows[0]["t_chunk_written"]) == 300.0
    rec.close(hw)


def test_a_skipped_slot_is_reconsidered_rather_than_suppressed(
        rings, cfg, hw, tmp_path):
    """The floor check runs BEFORE `_seen_chunk` is set, so a slot it rejects
    is looked at again next sweep. That is what makes a torn read of the one
    field it consults cost a 2 ms poll instead of a lost row."""
    rec = Recorder(tmp_path, cfg, rings, t_floor=200.0)
    _write_chunk(rings, cfg, chunk_id=4, k_first_action=1, t_written=150.0)
    rec.sweep()
    assert _read(rec.chunks).size == 0
    # The same slot, rewritten by this rollout.
    _write_chunk(rings, cfg, chunk_id=4, k_first_action=1, t_written=250.0)
    rec.sweep()
    assert [int(r["chunk_id"]) for r in _read(rec.chunks)] == [4]
    rec.close(hw)


def test_the_manifest_names_the_code_that_ran_not_the_tree_at_close(
        rings, cfg, hw, tmp_path):
    """Sampled at PROCESS START and carried, so a merge landing mid-session
    cannot make a record claim a commit it never imported.

    That is not hypothetical: on 2026-09-03 a branch merged at 15:07:28 between
    a run starting and its second rollout closing, and the record named the
    commit that declares a head camera while its own `cameras` had none.
    """
    stamped = {"sha": "deadbeef", "dirty": False, "dirty_files": [],
               "dirty_file_count": 0}
    rec = Recorder(tmp_path, cfg, rings, code=stamped)
    rec.close(hw)
    code = json.loads((tmp_path / "manifest.json").read_text())["code"]
    assert code == stamped


def test_the_manifest_carries_the_launchs_provenance(rings, cfg, hw, tmp_path):
    """The resolved config (above, as always), the robot file(s)
    verbatim, the policy entry, the checkpoint, both repos' revisions, the
    infra version, and a `rollouts` list ready for per-rollout settings."""
    from pathlib import Path

    from g1d_deploy.config_loading import deploy_file as deployfile
    from g1d_deploy.rollout_record_utils import manifest as provenance

    deploy = Path(__file__).resolve().parents[1] / "examples" / "deploy.toml"
    r = deployfile.resolve(deploy)
    prov = provenance.build(r, argv=["run", str(deploy)])
    prov["crops"] = {"lwrist": {"used": "refit", "crop": [1, 2, 3, 4]}}
    rec = Recorder(tmp_path / "rec", cfg, rings, provenance=prov,
                   rollout={"index": 2, "started": "t"})
    rec.close(hw)
    m = json.loads((tmp_path / "rec" / "manifest.json").read_text())

    assert m["infra"]["name"] == "g1d_deploy" and m["infra"]["version"]
    assert "sha" in m["infra"]["git"]
    assert [Path(f["path"]).stem for f in m["robot"]["files"]] == [
        "g1d_base", "g1d_fisheye_dex1"]
    assert "extends" in m["robot"]["files"][1]["text"], "the file itself"
    # The end effector's TCP travels with the robot file's text (the rig's).
    assert "tcp = [0.184759, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]" in \
        m["robot"]["files"][1]["text"]
    assert m["policy"]["entry"].endswith("policies/hold.py:Policy")
    assert m["policy"]["class"] == "examples.policies.hold.Policy"
    assert "sha" in m["policy"]["git"], "the policy's own repo"
    assert m["checkpoint"] == {"path": ""}
    assert m["deploy"]["text"] == deploy.read_text()
    assert m["resolution"]["sources"]["replan_period_ticks"] == "deploy"
    assert "ssh_password" not in json.dumps(m["machine"])
    assert m["crops"]["lwrist"]["used"] == "refit"
    # A Recorder handed no latched block still names the settings it ran
    # with: its own config's, which is the launch config here.
    from g1d_deploy.config_loading.rollout_settings import RolloutSettings
    assert m["rollouts"] == [{
        "index": 2, "dir": "rec", "prompt": cfg.prompt, "generation": None,
        "active_config": RolloutSettings.of(cfg).as_dict(), "started": "t"}]


def test_a_recorder_given_no_revision_still_records_one(rings, cfg, hw, tmp_path):
    """Every other test builds a `Recorder` without one, and a manifest with no
    provenance at all would be a worse record than a late-sampled one."""
    rec = Recorder(tmp_path, cfg, rings)
    rec.close(hw)
    code = json.loads((tmp_path / "manifest.json").read_text())["code"]
    assert "sha" in code


def test_frames_are_passthrough_and_gaps_are_counted(rings, cfg, hw, tmp_path):
    """No encoding: the bytes the robot sent, stored verbatim. And a frame the
    recorder was too slow to see must show up as a counted gap — the ring
    lapping rather than blocking is correct, silence about it is not.

    A late sweep CATCHES UP on every frame still in the ring (it keeps N; the
    triple buffer kept only the newest), so only a frame actually lapped is a
    gap."""
    rec = Recorder(tmp_path, cfg, rings)
    cam = rings["cameras"]["rwrist"]
    assert cam.n == 3

    cam.spare()["pixels"][:] = 1
    cam.publish(t_recv=10.0, jpeg=b"\xff\xd8FIRST")
    rec.sweep()

    cam.publish(t_recv=10.1, jpeg=b"\xff\xd8LATE")      # swept late: caught up
    cam.publish(t_recv=10.2, jpeg=b"\xff\xd8THIRD")
    rec.sweep()

    for i in range(4):                                  # 4 frames into 3 slots
        cam.publish(t_recv=11.0 + i, jpeg=b"\xff\xd8LAP%d" % i)
    rec.sweep()

    blob, index = rec.frames["rwrist"]
    blob.flush(); index.flush()
    idx = np.fromfile(index.path, dtype=FRAME_IDX)
    raw = index.path.with_suffix("").with_suffix(".jpg.bin").read_bytes()

    assert idx["frame_id"].tolist() == [1, 2, 3, 5, 6, 7]
    assert rec.dropped_frames["rwrist"] == 1            # frame 4, lapped
    for row in idx:
        assert raw[row["offset"]:row["offset"] + row["nbytes"]].startswith(b"\xff\xd8")
    first = idx[0]
    assert raw[first["offset"]:first["offset"] + first["nbytes"]] == b"\xff\xd8FIRST"
    rec.close(hw)


def test_an_uncaptured_frame_is_a_gap_not_a_crash(rings, cfg, hw, tmp_path):
    """`jpeg_nbytes == 0` — a frame over the slot capacity, or a source with no
    compressed bytes at all. The RECORD degrades; the run never does."""
    rec = Recorder(tmp_path, cfg, rings)
    cam = rings["cameras"]["rwrist"]
    cam.publish(t_recv=1.0, jpeg=None)
    rec.sweep()
    assert rec.frames["rwrist"][1].count == 0
    assert rec.dropped_frames["rwrist"] == 1
    rec.close(hw)


def test_state_is_recorded_once_and_in_order(rings, cfg, hw, tmp_path):
    rec = Recorder(tmp_path, cfg, rings)
    for i in range(20):
        rings["state"].write(
            t_recv=float(i), q=np.full(schema.N_JOINTS, i),
            dq=np.zeros(schema.N_JOINTS),
            flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
            grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=float(i))
        if i % 5 == 0:
            rec.sweep()
    rec.sweep()
    rows = _read(rec.state)
    assert rows["index"].tolist() == list(range(1, 21))
    rec.sweep()
    assert _read(rec.state).shape[0] == 20, "a second sweep must add nothing"
    rec.close(hw)


def test_the_manifest_makes_the_files_readable_without_this_code(rings, cfg, hw,
                                                                 tmp_path):
    """The dtype descriptor is the point: an old record stays readable by its
    own recorded schema rather than by whatever the constant says today."""
    rec = Recorder(tmp_path, cfg, rings)
    rings["targets"].write(1, 1.0, np.zeros(schema.BODY_DOF), intervene=0)
    rec.sweep()
    rec.close(hw)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    descr = [tuple(f) if len(f) == 2 else (f[0], f[1], tuple(f[2]))
             for f in (tuple(x) for x in manifest["streams"]["targets"]["descr"])]
    rows = np.fromfile(tmp_path / "targets.bin", dtype=np.dtype(descr))
    assert rows.shape[0] == 1 and rows[0]["k"] == 1

    assert manifest["streams"]["targets"]["records"] == 1
    assert manifest["cameras"]["rwrist"]["dropped"] == 0
    assert manifest["cpu"]["p5_policy"] == [4, 12, 14]


# --- The session log --------------------------------------------------------


def test_each_phase_is_logged_once_at_its_edge(rings, cfg, hw, tmp_path):
    """One row per transition, not one per sweep.

    The session word has no ring behind it, so unlike every other stream here
    this one is EDGE-triggered: it is the only thing that keeps a phase P0 sat
    in for forty seconds from becoming twenty thousand identical rows.
    """
    control = rings["control"]
    rec = Recorder(tmp_path, cfg, rings)

    for state in (schema.CHECK, schema.HOME, schema.WARMUP, schema.READY,
                  schema.ACTIVE, schema.EXITING, schema.STOPPED):
        control.session = state
        for _ in range(5):                    # many sweeps, one phase
            rec.sweep()

    rows = _read(rec.session)
    assert rows["session"].tolist() == [
        schema.CHECK, schema.HOME, schema.WARMUP, schema.READY,
        schema.ACTIVE, schema.EXITING, schema.STOPPED]
    assert np.all(np.diff(rows["t_observed"]) > 0), "edges must be ordered"
    rec.close(hw)


def test_the_first_phase_is_recorded_even_if_nothing_ever_changes(rings, cfg, hw,
                                                                  tmp_path):
    """A run that never leaves the phase it started in still says which one.

    Both halves matter, and the INIT half is the one with a trap in it. Dedupe
    is seeded with `None` rather than with INIT, because a fresh arena is zeroed
    and INIT *is* zero: seeding with the value would make "the phase word has
    never moved off its initial value" indistinguishable from "there is no phase
    log", and those are the two cases you most need to tell apart when a rollout
    produced nothing. P0 happens to spawn this process during CHECK today, so
    the INIT case is unreachable in production — which is precisely why it would
    otherwise go untested until the day the spawn order changes.
    """
    rec = Recorder(tmp_path, cfg, rings)
    rec.sweep()                               # a zeroed arena: still INIT
    rec.sweep()
    rows = _read(rec.session)
    assert rows.shape[0] == 1 and rows[0]["session"] == schema.INIT
    rec.close(hw)

    rings["control"].session = schema.HOME
    other = Recorder(tmp_path / "sub", cfg, rings)
    other.sweep()
    other.sweep()
    rows = _read(other.session)
    assert rows.shape[0] == 1 and rows[0]["session"] == schema.HOME
    other.close(hw)


def test_a_phase_edge_carries_the_rest_of_the_control_block(rings, cfg, hw,
                                                            tmp_path):
    """`k_exec`, `newest_chunk_id` and `warm` AS OF that edge.

    Three more reads off a cache line already being touched, so they are free,
    and they answer the questions that actually get asked of a phase log: how
    far the tick line had got, how many chunks P5 had published, and whether P5
    was warm *at that moment* — which the manifest's `warm` cannot say, being
    sampled once in `close()`.
    """
    control = rings["control"]
    rec = Recorder(tmp_path, cfg, rings)
    control.session = schema.WARMUP
    rec.sweep()

    control.k_exec = 91
    control.newest_chunk_id = 4
    control.warm = 1
    control.session = schema.ACTIVE
    rec.sweep()

    rows = _read(rec.session)
    assert rows["warm"].tolist() == [0, 1], "warm must be per-edge, not final"
    active = rows[rows["session"] == schema.ACTIVE][0]
    assert active["k_exec"] == 91 and active["newest_chunk_id"] == 4
    rec.close(hw)


def test_the_session_log_survives_a_recorder_that_never_closes(rings, cfg, hw,
                                                              tmp_path):
    """Flushed at every edge, so a `SIGKILL` costs the last phase and no more.

    This is the one stream whose value is highest in exactly the run that ends
    badly, and the default 1 MB buffer would hold ten 40-byte rows forever. The
    three 2026-08-19 records are what this is guarding against: killed before
    `close()`, and everything still in the buffer went with them.
    """
    control = rings["control"]
    rec = Recorder(tmp_path, cfg, rings)
    control.session = schema.HOME
    rec.sweep()
    control.session = schema.ACTIVE
    rec.sweep()

    # No close(), no flush() — read the file exactly as a killed process left it.
    rows = np.fromfile(tmp_path / "session.bin", dtype=rec.session.dtype)
    assert rows["session"].tolist() == [schema.HOME, schema.ACTIVE]


def test_the_session_log_reads_back_from_its_own_manifest_descriptor(
        rings, cfg, hw, tmp_path):
    """It is a stream like any other, which is the whole reason it is one.

    `rollout_record_utils.reader.load` walks `manifest["streams"]` and rebuilds each dtype from
    the record's own `descr`, so this file is readable by every existing reader
    without a line of reader code being written. A list of transitions in the
    manifest would have needed bespoke parsing in each of them.
    """
    from g1d_deploy.rollout_record_utils import reader as record_read

    control = rings["control"]
    rec = Recorder(tmp_path, cfg, rings)
    control.session = schema.HOME
    rec.sweep()
    control.session = schema.ACTIVE
    rec.sweep()
    rec.close(hw)

    got = record_read.load(tmp_path)
    rows = got["session"]
    assert rows.shape[0] == 2
    assert [schema.SESSION_NAMES[int(s)] for s in rows["session"]] == \
        ["HOME", "ACTIVE"]
    assert got["manifest"]["streams"]["session"]["records"] == 2


def test_recording_a_phase_edge_does_not_disturb_any_other_stream(rings, cfg, hw,
                                                                  tmp_path):
    """The session sweep is additive. Nothing else sees it.

    Written because the change that added it went into `sweep()`, which every
    other stream also runs through — and the whole premise of P9 is that a run
    behaves identically whether or not it exists.
    """
    control = rings["control"]
    rec = Recorder(tmp_path, cfg, rings)
    for k in range(1, 6):
        control.session = schema.HOME + (k % 3)     # thrashing the phase word
        rings["targets"].write(k, 100.0 + k, np.full(schema.BODY_DOF, k), intervene=0)
        rings["state"].write(
            t_recv=float(k), q=np.full(schema.N_JOINTS, k),
            dq=np.zeros(schema.N_JOINTS),
            flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
            grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=float(k))
        rec.sweep()

    assert sorted(_read(rec.targets)["k"].tolist()) == [1, 2, 3, 4, 5]
    assert _read(rec.state)["index"].tolist() == [1, 2, 3, 4, 5]
    rec.close(hw)


# --- run(): nothing before ACTIVE --------------------------------------------
#
# `Recorder` above is tested directly, mid-rollout, with `session` already
# wherever a test wants it — that is deliberate isolation and stays. These two
# exercise `run()` itself, the part that now decides whether a `Recorder` gets
# built at all. Threaded rather than subprocessed: `run()` blocks on the real
# arena, and the point under test is what it does to that arena and to
# `out_dir`, not process isolation, which `Recorder`'s own tests already don't
# need either.


def _run_in_background(arena, cfg, hw, out_dir) -> threading.Thread:
    t = threading.Thread(target=p9_record.run,
                         args=(arena.shm.name, cfg, hw),
                         kwargs={"out_dir": out_dir}, daemon=True)
    t.start()
    return t


def _records(base) -> list:
    """The stamped rollout directories under the base, oldest first.

    `out_dir` is the base — `rollout_record/` — and P9 stamps one directory
    inside it per ACTIVE period, so "what did this session record" is a
    listing rather than a single path.
    """
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() \
        else []


def _sessions(record) -> list:
    """`session.bin` read back through its own manifest descriptor."""
    manifest = json.loads((record / "manifest.json").read_text())
    descr = manifest["streams"]["session"]["descr"]
    rows = np.fromfile(record / "session.bin",
                       dtype=np.dtype([tuple(f) for f in descr]))
    return rows["session"].tolist()


def test_run_writes_nothing_before_active(arena, rings, cfg, hw, tmp_path):
    """CHECK/HOME/WARMUP/READY must leave no trace: not a directory, not a
    manifest, not a single record. That phase is setup, not the rollout, and it
    used to cost as much wrist-camera JPEG as the run itself for a window
    nobody read back."""
    control = rings["control"]
    control.session = schema.HOME
    out_dir = tmp_path / "rollout"
    t = _run_in_background(arena, cfg, hw, out_dir)
    try:
        for phase in (schema.WARMUP, schema.READY):
            time.sleep(0.05)
            assert not out_dir.exists(), (
                f"{schema.SESSION_NAMES[phase - 1]} wrote to disk before ACTIVE")
            control.session = phase
        time.sleep(0.05)
        assert not out_dir.exists(), "READY wrote to disk before ACTIVE"
    finally:
        control.session = schema.STOPPED
        t.join(timeout=2.0)
    assert not t.is_alive()


def test_run_records_nothing_for_a_rollout_abandoned_at_ready(arena, rings, cfg,
                                                               hw, tmp_path):
    """Ctrl-C at the READY prompt takes P0 straight to EXITING/STOPPED — ACTIVE
    is never set (`Launcher.ready`/`exit`). `run()` must recognise that as
    "never started" rather than as a one-tick rollout, and leave nothing
    behind: there is no reason to keep a record of an operator declining to
    run."""
    control = rings["control"]
    control.session = schema.READY
    out_dir = tmp_path / "rollout"
    t = _run_in_background(arena, cfg, hw, out_dir)
    time.sleep(0.05)

    control.session = schema.EXITING
    control.session = schema.STOPPED
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert not out_dir.exists(), \
        "a run that skipped ACTIVE entirely must record nothing"


def test_run_starts_recording_exactly_at_active(arena, rings, cfg, hw, tmp_path):
    """The first row `run()` ever writes to `session.bin` is ACTIVE, and
    everything the rollout needs — state, cameras, targets — starts from the
    same instant. Nothing from HOME leaks in just because the ring it lives in
    still holds it when the sweep starts."""
    control = rings["control"]
    control.session = schema.HOME
    out_dir = tmp_path / "rollout"
    t = _run_in_background(arena, cfg, hw, out_dir)
    time.sleep(0.05)
    assert not out_dir.exists()

    control.session = schema.ACTIVE
    time.sleep(0.05)
    control.session = schema.STOPPED
    t.join(timeout=2.0)
    assert not t.is_alive()

    records = _records(out_dir)
    assert len(records) == 1, "one ACTIVE period, one record"
    seen = _sessions(records[0])
    assert seen[0] == schema.ACTIVE, \
        f"first row must be ACTIVE, got {[schema.SESSION_NAMES[s] for s in seen]}"
    assert schema.HOME not in seen and schema.WARMUP not in seen \
        and schema.READY not in seen


# --- run(): one record per ACTIVE period -------------------------


def test_each_active_period_gets_its_own_record(arena, rings, cfg, hw, tmp_path):
    """The point of reset for the recorder. A session now serves many rollouts on
    one warmup, and one file spanning all of them would carry a tick line that
    runs 0…50000 and then starts again at −90 — which nothing that reads these
    records expects, and which no reader could split without guessing."""
    control = rings["control"]
    control.session = schema.READY
    out_dir = tmp_path / "rollout_record"
    t = _run_in_background(arena, cfg, hw, out_dir)
    try:
        for _ in range(3):
            control.session = schema.ACTIVE
            time.sleep(0.08)
            control.session = schema.HOME          # the operator pressed Reset
            time.sleep(0.08)
            control.session = schema.READY
            time.sleep(0.05)
    finally:
        control.session = schema.STOPPED
        t.join(timeout=2.0)

    records = _records(out_dir)
    assert len(records) == 3, \
        f"three rollouts, {len(records)} records: {[p.name for p in records]}"
    for record in records:
        # Each one closed by the process that owned it, so each one is readable.
        assert (record / "manifest.json").is_file(), record.name
        assert _sessions(record)[0] == schema.ACTIVE, record.name


def test_each_record_stores_the_settings_and_prompt_its_rollout_ran_with(
        arena, rings, cfg, hw, tmp_path):
    """Settings switch between rollouts, so a record that only
    carried the launch config would lie about every rollout but the first.
    Each manifest must name the ACTIVE block of its own rollout — and its
    `deploy_config` must be that block applied, because `record_read` reads
    the tick line with its `P` and `dt`."""
    import dataclasses

    from g1d_deploy.config_loading.rollout_settings import RolloutSettings

    control = rings["control"]
    control.session = schema.READY
    launch = RolloutSettings.of(cfg)
    plans = [(dataclasses.replace(launch, replan_period=2), "pick the cup"),
             (dataclasses.replace(launch, replan_period=5, sync=False,
                                  control_hz=15.0), "open the box")]
    out_dir = tmp_path / "rollout_record"
    t = _run_in_background(arena, cfg, hw, out_dir)
    try:
        for gen, (settings, prompt) in enumerate(plans, start=1):
            # What P0 does at an accepted START: the block, then the counter.
            control.activate(settings.to_record(), prompt)
            control.rollout_generation = gen
            control.session = schema.ACTIVE
            time.sleep(0.08)
            # A request made DURING the rollout must not reach its record.
            control.request_config(dataclasses.replace(
                settings, replan_period=7).to_record())
            time.sleep(0.03)
            control.session = schema.HOME
            time.sleep(0.05)
            control.session = schema.READY
            time.sleep(0.03)
    finally:
        control.session = schema.STOPPED
        t.join(timeout=2.0)

    records = _records(out_dir)
    assert len(records) == 2
    for record, (gen, (settings, prompt)) in zip(records,
                                                enumerate(plans, start=1)):
        m = json.loads((record / "manifest.json").read_text())
        (entry,) = m["rollouts"]
        assert entry["index"] == gen and entry["generation"] == gen
        assert entry["prompt"] == prompt
        assert entry["active_config"] == settings.as_dict()
        assert m["deploy_config"]["replan_period_ticks"] == settings.replan_period
        assert m["deploy_config"]["P"] == settings.replan_period
        assert m["deploy_config"]["sync"] == settings.sync
        assert m["deploy_config"]["dt"] == pytest.approx(1 / settings.control_hz)
        assert m["deploy_config"]["prompt"] == prompt


def test_a_record_ends_at_the_reset_rather_than_running_into_ready(
        arena, rings, cfg, hw, tmp_path):
    """A reset's homing ramp belongs to no rollout. It runs at `k < 0`, because
    reset puts the cursor back to zero, so recording it would put a
    descending tick line in the same file as the rollout's ticks. READY is worse
    still — it waits on a human, which is the window 6d72d3e removed."""
    control = rings["control"]
    control.session = schema.ACTIVE
    out_dir = tmp_path / "rollout_record"
    t = _run_in_background(arena, cfg, hw, out_dir)
    time.sleep(0.08)

    control.session = schema.HOME
    time.sleep(0.25)                    # long enough for READY to leak in
    control.session = schema.READY
    time.sleep(0.15)
    control.session = schema.STOPPED
    t.join(timeout=2.0)

    seen = _sessions(_records(out_dir)[0])
    assert schema.READY not in seen, \
        f"the record ran into READY: {[schema.SESSION_NAMES[s] for s in seen]}"


def test_the_exit_ramp_is_not_recorded_either(arena, rings, cfg, hw, tmp_path):
    """Exit ends a record exactly as Reset does. What P0 drives afterwards is
    the robot being put away — the same motion every time — not the rollout.

    `session.bin` still carries ONE row for the phase the rollout ended into, so
    a reader can tell an exit from a reset. What must not be there is the ramp
    itself: the targets and the frames of a minute of arm motion nobody asked to
    keep.
    """
    control = rings["control"]
    control.session = schema.ACTIVE
    out_dir = tmp_path / "rollout_record"
    t = _run_in_background(arena, cfg, hw, out_dir)
    time.sleep(0.08)

    control.session = schema.EXITING
    time.sleep(0.25)                    # a ramp's worth of time to leak in
    control.session = schema.STOPPED
    t.join(timeout=2.0)

    seen = _sessions(_records(out_dir)[0])
    # EXITING appears either way — `_sweep_session` records a phase once, at its
    # edge, and the final sweep catches it. STOPPED is what separates the two:
    # a recorder that swept THROUGH the ramp is still running when the session
    # ends and records its end, and one that stopped at the ACTIVE edge cannot
    # have seen it.
    assert seen == [schema.ACTIVE, schema.EXITING], \
        f"swept past the ACTIVE edge: {[schema.SESSION_NAMES[s] for s in seen]}"


def test_two_rollouts_never_share_a_directory(tmp_path):
    """A stamp has one-second resolution and a reset cannot produce two rollouts
    inside one second — the homing ramp alone is `home_min_s`. But two records
    sharing a directory would interleave into one unreadable set of files, which
    is worth making impossible rather than merely unlikely."""
    base = tmp_path / "rollout_record"
    first = p9_record.rollout_dir(base)
    first.mkdir(parents=True)
    second = p9_record.rollout_dir(base)

    assert second != first
    assert second.parent == base



# --- The chunk ring carries either action mode --------------------


def test_both_action_modes_are_recorded_and_read_back_by_mode(
        rings, cfg, hw, tmp_path):
    """The chunk stream is the ring's own dtype, so a joint chunk's `arm_q`,
    `mode` and `waist` reach disk with no recorder change; `record_read`
    then hands each chunk back in the half of the contract it spoke."""
    from g1d_deploy.rollout_record_utils import reader as record_read

    rec = Recorder(tmp_path, cfg, rings)
    n = cfg.P
    stamps = dict(t_trigger=0.0, t_anchor=0.0, t_obs_received=0.0,
                  t_cpu_processed=0.0, t_chunk_received=0.0)
    rings["chunks"].write(chunk_id=1, k_first_action=1, n_actions=n,
                          pose=np.zeros((n, 2, 7)), hand=np.zeros((n, 2, 6)),
                          t_chunk_written=5.0, **stamps)
    q = np.full((n, 2, 7), 0.3)
    rings["chunks"].write(chunk_id=2, k_first_action=1 + n, n_actions=n,
                          arm_q=q, waist=np.full((n, 3), 0.01),
                          hand=np.zeros((n, 2, 6)), t_chunk_written=6.0,
                          **stamps)
    rec.sweep()
    rec.close(hw)

    got = record_read.load(tmp_path)
    rows = got["chunks"][np.argsort(got["chunks"]["chunk_id"])]
    eef, joint = (record_read.chunk_actions(r) for r in rows)
    assert eef["mode"] == "eef" and eef["waist"] is None
    assert eef["pose"].shape == (n, 2, 7) and "arm_q" not in eef
    assert joint["mode"] == "joint" and np.allclose(joint["arm_q"], q)
    assert np.allclose(joint["waist"], 0.01) and "pose" not in joint
    assert record_read.chunk_modes(got["chunks"]) == {"eef": 1, "joint": 1}
    assert "eef (1 chunks), joint (1 chunks)" in record_read.summarize(got)


def test_a_record_from_before_the_mode_column_reads_as_eef_with_its_waist():
    """Arena schema < 9: no `mode`, and the `waist` column WAS the command."""
    from g1d_deploy.rollout_record_utils import reader as record_read

    old = np.dtype([(name, *rest) for name, *rest in schema.chunk_dtype(4).descr
                    if name not in ("mode", "arm_q")])
    row = np.zeros((), dtype=old)
    row["n_actions"] = 2
    row["waist"][:] = 0.2
    got = record_read.chunk_actions(row)
    assert got["mode"] == "eef" and np.allclose(got["waist"], 0.2)
    assert got["pose"].shape == (2, 2, 7)
    assert record_read.chunk_modes(np.zeros(3, dtype=old)) == {"eef": 3}
