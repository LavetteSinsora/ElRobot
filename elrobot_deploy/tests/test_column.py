"""The lift column: its wire, and P1 recording its height.

Without it a record cannot say how high the robot was, because the column is
not in `rt/lowstate`'s 29 slots.

Nothing here commands anything yet. What is built is the measurement path and
the wire adapter, both of which are testable with no robot; the controller that
turns a commanded height into a velocity is not built, and wants a gain
measured on hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.runtime.processes.p1_state import ColumnLatch
from g1d_deploy.runtime.drivers import column as col


# --- The wire ---------------------------------------------------------------


def test_the_command_is_metres_per_second_at_the_boundary():
    """The wire is normalised to [-1, 1]; nothing above it should have to know
    that. `publish` takes m/s and scales here, once."""
    c = col.FakeColumnCmd()
    c.connect()
    c.publish(col.FULL_SCALE_MPS)
    c.publish(col.FULL_SCALE_MPS / 2)
    assert c.sent[0] == pytest.approx(col.FULL_SCALE_MPS)
    assert c.sent[1] == pytest.approx(col.FULL_SCALE_MPS / 2)


def test_a_command_past_full_scale_is_clamped_not_refused():
    """The last point a command passes through before the robot, so it clamps —
    the same argument P8's range clamp is built on."""
    c = col.FakeColumnCmd()
    c.connect()
    c.publish(10.0)
    c.publish(-10.0)
    assert c.sent[0] == pytest.approx(col.FULL_SCALE_MPS)
    assert c.sent[1] == pytest.approx(-col.FULL_SCALE_MPS)


def test_publishing_before_connect_raises():
    with pytest.raises(RuntimeError, match="before connect"):
        col.FakeColumnCmd().publish(0.0)


def test_stop_is_repeated_because_the_topic_is_fire_and_forget():
    """An arm that loses its publisher holds position; a column that loses its
    publisher keeps moving. One dropped datagram must not be the difference."""
    c = col.FakeColumnCmd()
    c.connect()
    c.publish(0.05)
    c.stop()
    assert c.sent[-col.STOP_REPEATS:] == [0.0] * col.STOP_REPEATS
    assert col.STOP_REPEATS > 1


def test_close_stops_the_column():
    c = col.FakeColumnCmd()
    c.connect()
    c.publish(0.05)
    c.close()
    assert c.sent[-1] == 0.0


def test_stop_before_connect_is_harmless():
    """Called from a `finally` that may run before anything was set up."""
    col.DdsColumnCmd(hw=None).stop()


# --- The latch --------------------------------------------------------------


def test_the_latch_holds_the_last_height_and_its_stamp():
    latch = ColumnLatch(col.FakeColumnState([0.10, 0.15]))
    assert latch.latest == (0.0, 0.0)          # never received
    latch.start()
    latch._thread.join(timeout=2.0)
    height, t = latch.latest
    assert height == pytest.approx(0.15)       # the LAST sample, held
    assert t > 0.0


def test_no_source_is_a_latch_that_never_moves():
    """A run without the column subscriber records an honest "never received"
    rather than a robot standing at exactly zero."""
    latch = ColumnLatch(None)
    latch.start()
    assert latch.latest == (0.0, 0.0)
    assert latch._thread is None


def test_the_latch_stops_rather_than_spinning_at_eof():
    latch = ColumnLatch(col.FakeColumnState([0.2]))
    latch.start()
    latch._thread.join(timeout=2.0)
    assert not latch._thread.is_alive()
    assert latch.latest[0] == pytest.approx(0.2)


# --- The record -------------------------------------------------------------


def test_state_carries_the_height_and_its_own_stamp(rings):
    """Its own stamp, because the column arrives at ~30 Hz against the loop's
    1070 — most samples carry a HELD value and a reader has to be able to tell.
    """
    assert "column_h" in schema.STATE.names
    assert "t_column_recv" in schema.STATE.names
    rings["state"].write(t_recv=5.0, q=np.zeros(schema.N_JOINTS),
                         dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=4.0,
                         column_h=0.37, t_column_recv=4.9)
    snap = rings["state"].latest()
    assert float(snap["column_h"]) == pytest.approx(0.37)
    assert float(snap["t_column_recv"]) == pytest.approx(4.9)


def test_a_state_write_without_a_column_records_never_received(rings):
    """The default has to be writable: P1 must not refuse a state sample
    because nothing is subscribed to the column."""
    rings["state"].write(t_recv=1.0, q=np.zeros(schema.N_JOINTS),
                         dq=np.zeros(schema.N_JOINTS),
                         flange=np.zeros((schema.N_HANDS, schema.POSE_DIM)),
                         grip_meas=np.zeros(schema.GRIP_DIM), t_grip_recv=0.0)
    snap = rings["state"].latest()
    assert float(snap["t_column_recv"]) == 0.0
