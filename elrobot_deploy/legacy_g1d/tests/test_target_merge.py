"""The merged target ring, and the bridge that keeps old records readable.

Two rings became one, which is what turns the shared-deadline rule ("a body slot and a hand slot with
the same `k` carry the same deadline") and the both-or-neither rule ("P6 writes both rings for a tick,
or neither") from rules P6 follows into properties of the record. This file pins
both the new guarantee and the thing it must not have cost: 36 recordings were
written in the two-stream layout and every one of them still has to open.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from elrobot_deploy.rollout_record_utils import reader as record_read
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.rings import TargetRing

T0 = 1000.0


# --- The guarantee ----------------------------------------------------------


def test_one_row_carries_both_payloads(rings):
    """The point of the merge: a tick's joints and its hand command are one
    record, so neither can exist without the other."""
    q = np.full(schema.BODY_DOF, 0.3)
    cmd = np.full((schema.N_HANDS, schema.HAND_DIM), 1.7)
    rings["targets"].write(4, T0, q, cmd, intervene=0)

    body = rings["body"].get(4)
    hand = rings["hand"].get(4)
    assert body is not None and hand is not None
    assert np.allclose(body.payload, q)
    assert np.allclose(hand.payload, cmd)


def test_the_two_views_cannot_disagree_about_a_deadline(rings):
    """The shared-deadline rule, now unfalsifiable: there is one `deadline` column, so the body and
    hand views are reading the same eight bytes."""
    rings["targets"].write(9, 12.5, np.zeros(schema.BODY_DOF),
                           np.zeros((schema.N_HANDS, schema.HAND_DIM)), intervene=0)
    assert rings["body"].get(9).deadline == rings["hand"].get(9).deadline == 12.5


def test_the_views_share_one_region(rings):
    """Not two regions kept in step — one region, opened twice. If these ever
    became separate allocations the invariants would go back to being
    promises."""
    assert rings["body"]._rec.ctypes.data == rings["hand"]._rec.ctypes.data
    assert rings["body"]._seq.ctypes.data == rings["hand"]._seq.ctypes.data
    assert rings["targets"]._rec.ctypes.data == rings["body"]._rec.ctypes.data


def test_a_view_is_named_for_the_column_it_returns(rings):
    assert rings["body"].column == "q"
    assert rings["hand"].column == "hand"
    # A bad column is a programming error and is refused, whether the ring is
    # opened on it or a view is asked for it.
    with pytest.raises(ValueError, match="not a column of TARGET"):
        rings["targets"].view("elbow")
    with pytest.raises(ValueError, match="not a column of TARGET"):
        TargetRing(rings["targets"], "elbow")


def test_a_view_reads_the_writers_rows_without_a_second_lock(rings):
    """`view` shares the seqlock object itself, so a reader on the hand column
    and the writer are serialised by the same counters — not two locks that
    happen to guard the same bytes."""
    hand = rings["targets"].view("hand")
    assert hand._lock is rings["targets"]._lock
    cmd = np.full((schema.N_HANDS, schema.HAND_DIM), 0.7)
    rings["targets"].write(3, 1.0, np.zeros(schema.BODY_DOF), cmd, intervene=0)
    assert np.array_equal(hand.get(3).payload, cmd)
    assert np.array_equal(rings["hand"].get(3).payload, cmd)


def test_an_unwritten_column_is_zeroed_not_inherited(rings):
    """A wrapped slot must not hand back the previous tick's hand command.

    This is why `write` zero-fills a payload it is not given rather than leaving
    what was there: `get(k)` answering with another tick's data is exactly the
    "individually plausible, jointly a lie" failure the ring already guards
    against elsewhere.
    """
    n = rings["targets"].n
    rings["targets"].write(3, T0, np.zeros(schema.BODY_DOF),
                           np.full((schema.N_HANDS, schema.HAND_DIM), 9.0), intervene=0)
    rings["targets"].write(3 + n, T0 + 1.0, np.zeros(schema.BODY_DOF), intervene=0)
    assert np.allclose(rings["hand"].get(3 + n).payload, 0.0)


# --- The reader's views -----------------------------------------------------


def test_a_record_offers_body_and_hand_views_of_its_targets(tmp_path):
    """Readers that want one half read `body_targets` / `hand_targets`."""
    n = 5
    rows = np.zeros(n, dtype=np.dtype(schema.TARGET.descr
                                      + [("seq", "<u8"), ("t_observed", "<f8")]))
    for i in range(n):
        rows["k"][i] = i
        rows["deadline"][i] = T0 + i
        rows["q"][i] = i
        rows["hand"][i] = 3.0 + i
    rows.tofile(tmp_path / "targets.bin")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "created": "test",
        "streams": {"targets": {"file": "targets.bin",
                                "descr": rows.dtype.descr, "records": n}}}))

    rec = record_read.load(tmp_path)
    assert np.allclose(rec["body_targets"]["q"], rows["q"])
    assert np.allclose(rec["hand_targets"]["cmd"], rows["hand"])
    assert np.array_equal(rec["hand_targets"]["k"], rows["k"])
