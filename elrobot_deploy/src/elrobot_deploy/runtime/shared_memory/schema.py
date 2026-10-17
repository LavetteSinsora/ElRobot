"""Shapes, dtypes and record layouts: the vocabulary every process shares.

Nothing here allocates or reads anything. It is the one place where every shared record turns
into a numpy dtype, so a disagreement about what a slot contains is a diff in this file rather
than a torn read at 250 Hz.

The robot is ONE 7-DoF arm plus a gripper, all on one servo bus: 8 joints, in servo order,
base to gripper (`robot/elrobot_joints.py`). The gripper is joint 7 of the same row, not a
separate hand ring: it is the same bus, the same rate and the same deadline, so one row and one
segment walk carry all 8 (PORTING_PLAN.md workstream 1).

Every record's itemsize is a multiple of 8, so the 8-byte words the unlocked column scans read
are aligned in every slot (an unaligned word is not guaranteed to load atomically).

Counters are `uint64` in their own array, never a field inside a payload record: a float
counter stops incrementing past its mantissa, and every torn read afterwards goes undetected.

Changing a record here means bumping `VERSION` (THE RULE below). The g1d version of this file,
with its two arms, waist and policy chunks, is `legacy_g1d/src/elrobot_deploy/runtime/
shared_memory/schema.py`.
"""

from __future__ import annotations

import numpy as np

# --- the robot ------------------------------------------------------------------------------

ARM_DOF = 7           # rev_motor_01 .. rev_motor_07
N_JOINTS = 8          # + the gripper (servo 8)
GRIPPER = 7           # its index in every joint vector
POSE_DIM = 7          # x y z qw qx qy qz

#: `k` sentinel for "this slot has never been written". Homing runs at k < 0, so the sentinel
#: cannot be a small negative number.
K_UNSET = np.iinfo(np.int64).min

# --- session states -------------------------------------------------------------------------
#
# Written by P0 only. WARMUP is the policy's (loading and compiling before READY); a teleop
# launch passes straight through it.

INIT, CHECK, HOME, WARMUP, READY, ACTIVE, EXITING, STOPPED = range(8)

SESSION_NAMES = {
    INIT: "INIT", CHECK: "CHECK", HOME: "HOME", WARMUP: "WARMUP",
    READY: "READY", ACTIVE: "ACTIVE", EXITING: "EXITING", STOPPED: "STOPPED",
}

# --- operator requests ----------------------------------------------------------------------
#
# What somebody ASKED for, as opposed to `session`, which is where the run IS. Stored by a
# requester (the phone, a dashboard, the terminal), consumed and cleared by P0 (the one
# two-writer word: a lost request is a button that did nothing, never a state walked
# backwards). START begins a rollout from READY, RESET ends it and returns to READY, EXIT is
# the exit sequence, the same as the first Ctrl-C.
REQ_NONE, REQ_START, REQ_RESET, REQ_EXIT, REQ_HOME = range(5)

REQUEST_NAMES = {REQ_NONE: "NONE", REQ_START: "START", REQ_RESET: "RESET", REQ_EXIT: "EXIT",
                 REQ_HOME: "HOME"}

# --- what P0 is waiting for (`control.wait`) --------------------------------------------------
#
# Written by P0 so a dashboard can offer the one button that moves the session on: HOME (the
# arm is held where it was found, the home pose is previewed, nothing moves until confirmed)
# and START (homed, waiting for the rollout to begin). PHONE: CHECK is waiting for the phone to
# stream (no button: it ends when the phone does).
WAIT_NONE, WAIT_HOME, WAIT_START, WAIT_PHONE = range(4)

WAIT_NAMES = {WAIT_NONE: "NONE", WAIT_HOME: "HOME", WAIT_START: "START", WAIT_PHONE: "PHONE"}

# --- the bus process's own word -------------------------------------------------------------
#
# Written by the bus process only. P0 reads it: a HOLDING bus has stopped following the
# target ring (a guard fired; `fault` says which) and holds its last command, so the run must
# end. DONE is written in the bus process's `finally`, after torque went off (or failed to).
# RECOVERING: a guard fired (`fault` says which) but the loop still follows the ring, so P0 can
# ramp the arm home and to rest before torque goes off.
# DRY is RUNNING with torque off: the bus process reads the arm and computes every command, but
# writes none (`--dry-run`).
BUS_STARTING, BUS_RUNNING, BUS_HOLDING, BUS_DONE, BUS_DRY, BUS_RECOVERING = range(6)

BUS_NAMES = {BUS_STARTING: "STARTING", BUS_RUNNING: "RUNNING", BUS_HOLDING: "HOLDING",
             BUS_DONE: "DONE", BUS_DRY: "DRY", BUS_RECOVERING: "RECOVERING"}

# --- records --------------------------------------------------------------------------------

HEADER = np.dtype([
    ("magic", "<u8"),
    ("version", "<u8"),
    ("layout_hash", "<u8"),
])

MAGIC = 0x454C52000001  # "ELR" + 1

#: 1: the single-arm arena (one 8-joint row; bus state; the phone's relative motion).
#: 2: `CONTROL.wait` (what P0 waits for), the bus process's hardware report, REQ_HOME, BUS_DRY.
#: 3: camera rings (`camera_dtype`, `CAMERA_STATUS`), one per `[cameras.<name>]`.
#:
#: THE RULE: bump this whenever a record's fields, a region's name or a ring's read protocol
#: changes, anything a reader written against the old layout would misread rather than fail
#: on. The layout hash already refuses an attach across different SIZES; this refuses one
#: across different MEANINGS.
VERSION = 3

#: Bytes of UTF-8 one line of "why" may take (a bus fault, a refused START).
REASON_BYTES = 256
#: Bytes of the bus process's hardware report (JSON, `drivers.st3215_driver.check`).
REPORT_BYTES = 4096

# One record. The single words are independently atomic aligned 8-byte stores with no seqlock,
# because no reader needs two of them to agree; the fault text is multi-word and carries its
# own seqlock counter (`fault_seq`, odd while written).
CONTROL = np.dtype([
    ("session", "<u8"),          # P0
    ("k_exec", "<i8"),           # the bus process: the tick it is walking toward
    ("request", "<u8"),          # a requester stores, P0 clears
    # THE INTERVENTION COUNTER, written by the phone source. Even = nobody is driving (the arm
    # holds, or a policy drives), odd = the operator is. Every increment is also "every target
    # stamped with the previous value belongs to a world that no longer exists": the bus
    # process abandons its segment and P6 drops its cursor. A counter, not a flag, so a reader
    # cannot miss two transitions between its polls. Monotonic, never reset.
    ("intervene", "<u8"),
    ("bus", "<u8"),              # the bus process: BUS_*
    # P0: may the bus process turn torque off when the session reaches STOPPED? Set only once
    # the exit ramp has brought the arm to the exit pose (and the profile allows it). On any
    # other way out (a failure, a second Ctrl-C) it stays 0 and the arm stays HELD.
    ("release_torque", "<u8"),
    ("fault_seq", "<u8"),        # the bus process; seqlock over the two fields below
    ("fault_len", "<u8"),
    ("fault", "u1", (REASON_BYTES,)),
    ("refusal_seq", "<u8"),      # P0: why the last START was refused
    ("refusal_len", "<u8"),
    ("refusal", "u1", (REASON_BYTES,)),
    ("wait", "<u8"),             # P0: WAIT_*
    # The bus process: what the hardware check found, JSON, written once before torque on.
    ("report_seq", "<u8"),
    ("report_len", "<u8"),
    ("report", "u1", (REPORT_BYTES,)),
])

#: What the bus process measured, one row per position read (every tick, 250 Hz). The slow
#: fields are refreshed every few ticks and carry the time of their own read (`t_slow`, NaN
#: when this row's read did not include them), so a reader never mistakes an old load for a
#: new one.
STATE = np.dtype([
    ("index", "<u8"),                   # absolute row counter; 0 = never written
    ("t", "<f8"),                       # monotonic, middle of the position read
    ("q", "<f8", (N_JOINTS,)),          # measured, rad
    ("q_cmd", "<f8", (N_JOINTS,)),      # the goal written just before this read, rad
    ("t_slow", "<f8"),
    ("speed", "<f8", (N_JOINTS,)),      # rad/s
    ("load", "<f8", (N_JOINTS,)),       # fraction of stall torque, signed
    ("temp", "<f8", (N_JOINTS,)),       # deg C
    ("alarm", "u1", (N_JOINTS,)),       # each servo's status byte from its latest reply
])

#: ONE target: what the joints are to be at `deadline`. Written by P6 (and by P0 during
#: homing, reset and exit); read by the bus process, which interpolates between them.
TARGET = np.dtype([
    ("k", "<i8"),                       # tick; K_UNSET = never written
    ("deadline", "<f8"),                # monotonic, this machine
    # Which world this slot belongs to: `control.intervene` when it was written. Beside
    # `deadline` because the bus process's unlocked column scan reads both (one cache line).
    # A row from a previous world is never selected, so a takeover or a hand-back needs no
    # wipe.
    ("intervene", "<u8"),
    ("q", "<f8", (N_JOINTS,)),          # rad, servo order; [GRIPPER] is the jaw (0 = open)
    # The IK's residual for this tick: TCP position error, metres, at the configuration
    # published. NaN when no solve produced the row (P0's joint-space ramps).
    ("err", "<f8"),
    # The IK's INPUT: the TCP pose (base frame) `q` is the solution for. NaN without an IK.
    ("pose", "<f8", (POSE_DIM,)),
])

#: ONE operator sample, written by the phone source while the clutch is held, read by P6.
#:
#: It carries the operator's motion RELATIVE to the clutch press, already mapped into the
#: robot's base axes (`teleop/phone.py`), not an absolute pose: the phone source needs no model
#: of the arm, and P6, which owns the model, composes it with the TCP pose it froze at the same
#: counter change. One anchor, in the one process that can compute it.
INTERVENE = np.dtype([
    ("index", "<u8"),                   # this ring's own row counter; 0 = never written
    ("intervene", "<u8"),               # the counter it was written under (always odd)
    ("deadline", "<f8"),                # already mapped off the phone's clock (`Pace`)
    # The TCP's motion since the press: translation (m) and rotation (wxyz), both in the base
    # frame, rotation about the TCP. `[0 0 0 1 0 0 0]` = "where it was at the press".
    ("delta", "<f8", (POSE_DIM,)),
    ("grip", "<f8"),                    # gripper command, rad (0 = open), already bounded
    ("t_device", "<f8"),                # the phone's capture instant (its own clock)
    ("t_recv", "<f8"),                  # when this machine received it
    # The raw inputs, recorded rather than used downstream.
    ("engage", "u1"),                   # the clutch button
    ("grip_button", "u1"),              # the close button
], align=True)   # padded to a multiple of 8, so every slot's words stay 8-byte aligned

#: The phone source's header: what is true of the phone rather than of a row. Independent
#: words, like `CONTROL`. `samples` is bumped on EVERY packet, engaged or not: it is what tells
#: "nobody is holding the clutch" from "the phone is gone", which the ring cannot (it is
#: written only while engaged, so it is empty on a healthy idle rig).
INTERVENE_META = np.dtype([
    ("samples", "<u8"),                 # packets received
    ("newest", "<u8"),                  # published row index
    ("t_last_sample", "<f8"),           # when a packet last arrived
    ("fresh", "<u8"),                   # is the stream live right now
    ("engaged", "<u8"),                 # is the clutch engaged right now
    ("track", "<u8"),                   # the phone's tracking state (`teleop.phone.TRACK_*`)
    ("world_epoch", "<u8"),             # the phone's world-origin counter
    ("latency_s", "<f8"),               # newest one-way latency estimate, s (NaN = unknown)
    ("released", "<u8"),                # forced releases so far (a guard fired; P11 logs why)
])


# --- cameras --------------------------------------------------------------------------------
#
# One ring per `[cameras.<name>]`, written by that camera's P2 only (`rings.CameraRing`). The
# slot holds the WHOLE frame, as the camera produced it, plus the square a reader should cut
# from it (`lens.py`): the crop is metadata, so a refit mid-session (`lens.REFIT_RETRY_S`)
# changes the next frame's square and never the ring's shape, and P2 opens its camera once
# (the fit is its own, from its own first frames).

#: Who set this frame's exposure: the camera's own loop, or ours (`cameras/exposure.py`).
EXPOSURE_CAMERA, EXPOSURE_OURS = 0, 1

#: Bytes of a camera's status (JSON: device, mode, lens decision, exposure state), for the
#: dashboard and the terminal.
CAMERA_STATUS_BYTES = 4096


def camera_dtype(height: int, width: int) -> np.dtype:
    """One camera-ring slot: the frame and what is true of it.

    `exposure` / `gain` are the registers in effect when we own exposure (-1 while the
    camera's own loop does, unless it was read back), so a recording can find every step
    (PORTING_PLAN.md section 6, item 4).
    """
    return np.dtype([
        ("frame_id", "<u8"),                 # this camera's frames, from 1; 0 = empty
        ("t_capture", "<f8"),                # the frame's capture time (else its arrival)
        ("t_recv", "<f8"),                   # when P2 received it
        ("crop", "<i8", (4,)),               # x0, y0, x1, y1 (end exclusive; may overhang)
        ("exposure_owner", "<i8"),           # EXPOSURE_*
        ("exposure", "<i8"),                 # 100 us units, -1 = unknown
        ("gain", "<i8"),                     # -1 = unknown
        ("pixels", "u1", (height, width, 3)),   # BGR
    ], align=True)


CAMERA_STATUS = np.dtype([
    ("len", "<u8"),
    ("text", "u1", (CAMERA_STATUS_BYTES,)),
])


__all__ = [
    "ACTIVE",
    "ARM_DOF",
    "BUS_DONE",
    "BUS_HOLDING",
    "BUS_NAMES",
    "BUS_RECOVERING",
    "BUS_RUNNING",
    "BUS_STARTING",
    "CAMERA_STATUS",
    "CHECK",
    "CONTROL",
    "EXITING",
    "EXPOSURE_CAMERA",
    "EXPOSURE_OURS",
    "GRIPPER",
    "HEADER",
    "HOME",
    "INIT",
    "INTERVENE",
    "INTERVENE_META",
    "K_UNSET",
    "MAGIC",
    "N_JOINTS",
    "POSE_DIM",
    "READY",
    "REASON_BYTES",
    "REQUEST_NAMES",
    "REQ_EXIT",
    "REQ_NONE",
    "REQ_RESET",
    "REQ_START",
    "SESSION_NAMES",
    "STATE",
    "STOPPED",
    "TARGET",
    "VERSION",
    "WARMUP",
    "camera_dtype",
]
