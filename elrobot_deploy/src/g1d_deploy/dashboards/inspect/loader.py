"""One recording, in memory.

Thin on purpose. `g1d_deploy.rollout_record_utils.reader` already owns the contract that matters
— rebuild each stream's dtype from the manifest's own `descr`, never from
`schema.py`, which may have moved on since the record was written — so this
module reuses it and adds only what a viewer needs on top: the camera index
arrays (which the reader deliberately leaves on disk), the JPEG blobs kept as
open file handles, and the facts needed to pose a model.

**Nothing is decoded here.** A rollout is ~100 MB of JPEG per camera and the
viewer shows one frame at a time; the blobs stay on disk and are sliced by
`(offset, nbytes)` on demand. What is loaded is `state.bin` (~58 MB of float64)
and the two index arrays (~85 KB each), because the timeline needs all of their
timestamps at once.

**The robot comes from the record** (`record_robot`): the robot file chain P9
copied into the manifest verbatim gives the joints, the hand's travel, the
virtual TCP and the URDF, so a record is drawn against the robot it ran on and
not against whatever this machine's default profile says today. A record that
predates that copy falls back to the shipped default, and the summary says so.

Used by the inspection server (one `Rollout` per Load) and by the replay
producer. For a user: `docs/deploying_a_policy/rollout_inspection.md`, "What it opens".
"""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from g1d_deploy.rollout_record_utils import reader as record_read
from g1d_deploy.dashboards.inspect import discover
from g1d_deploy.dashboards.inspect import record_robot

#: Serialises seek-then-read on a blob with no file descriptor (`Camera.jpeg`).
_BLOB_LOCK = threading.Lock()

#: `schema.ACTIVE`, copied rather than imported. A record's integers mean what
#: they meant when it was written: `session.bin` stores the phase as a bare integer, so its
#: meaning is fixed at the moment the record was written. Importing
#: the live constant would mean that inserting a phase into the enum silently
#: re-reads every rollout on disk as some other phase; a literal here stays
#: correct for the records that already exist, which is the only thing this
#: package ever looks at.
_SESSION_ACTIVE = 5

def camera_roles(manifest: dict) -> dict[str, dict]:
    """Per camera: which policy input slot it filled (if the record says),
    whether the policy saw it, and whether one frame carries two eyes.

    **Read from the record's own `deploy_config`, never from a table here.**
    "Which cameras exist" and "which cameras the policy read" are different
    questions (a camera may be recorded and read by nothing), and only the
    record can answer either.

    The fallbacks are what records without the field mean, not guesses about
    them. A record with no `observed_cameras` declares no record-only camera —
    every camera it lists was observed — so an absent field resolves to
    `camera_slots` if the config had one and otherwise to every declared
    camera.
    """
    cfg = manifest.get("deploy_config") or {}
    declared = cfg.get("cameras") or {}
    slots = cfg.get("camera_slots") or {}
    observed = (tuple(cfg.get("observed_cameras") or ())
                or tuple(slots) or tuple(declared))
    primary = cfg.get("primary_camera")
    return {
        name: {
            "slot": slots.get(name),
            "used": name in observed,
            "primary": name == primary,
            "binocular": bool((spec or {}).get("binocular", False)),
            "source": (spec or {}).get("source", "zmq"),
            # The frame's shape, so the viewer can letterbox to the right box
            # instead of guessing. None on a record whose config predates this
            # dump, where the page falls back to plain `object-fit: contain` —
            # correct for every camera except a binocular one, whose displayed
            # half is not the shape of the frame it arrives in.
            "height": (spec or {}).get("height"),
            "width": (spec or {}).get("width"),
            # A fisheye's lens square and circle, drawn over its pane.
            "crop": (spec or {}).get("crop"),
            "lens_circle": (spec or {}).get("lens_circle"),
        }
        for name, spec in declared.items()
    }


def _camera_order(manifest: dict, roles: dict) -> list[str]:
    """Recorded cameras, the ones the policy read first.

    A record-only view is context, not evidence, and putting it after the
    cameras the actions were computed from keeps the panes in the order the
    question is usually asked in. Within each group, by name.
    """
    return sorted(manifest.get("cameras") or {},
                  key=lambda n: (not roles.get(n, {}).get("used", True), n))


#: What the scene reads to pose this record's robot: the recorded robot
#: (`record_robot.RecordRobot`). The scene, the live dashboard and the
#: overlays all take "facts" with `joints_arm` / `joints_waist`, so the name
#: stays.
ModelFacts = record_robot.RecordRobot


def _model_facts(manifest: dict) -> ModelFacts:
    return record_robot.from_manifest(manifest)


#: `chunk.mode`'s values (`schema.chunk_dtype`), copied for `_SESSION_ACTIVE`'s
#: reason: a record's integers mean what they meant when it was written.
_CHUNK_MODE = {0: "eef", 1: "joint"}


def action_mode(chunks: np.ndarray) -> str:
    """"eef", "joint" or "mixed", read off the chunks the policy published.

    Per chunk, because a chunk is self-describing (`schema.chunk_dtype`'s
    `mode`). A record written before the joint contract has no `mode` column,
    and every such record is eef; so is one with no chunks, where nothing says
    otherwise and the overlays simply have nothing to draw.
    """
    if not len(chunks) or "mode" not in (chunks.dtype.names or ()):
        return "eef"
    live = chunks[chunks["chunk_id"] > 0] if "chunk_id" in chunks.dtype.names \
        else chunks
    modes = {_CHUNK_MODE.get(int(m), f"#{int(m)}") for m in live["mode"]}
    if not modes:
        return "eef"
    return modes.pop() if len(modes) == 1 else "mixed"


@dataclass
class Camera:
    """One camera's index, plus the blob it points into."""

    name: str
    index: np.ndarray                   # FRAME_IDX records
    blob: BinaryIO
    dropped: int

    #: What this camera was FOR, straight off the record's `deploy_config`.
    #: `slot` is the policy input slot it filled (None when the record names
    #: none, or for a camera that was only
    #: recorded; `used` says whether the policy saw it at all; `primary` marks
    #: the one whose chosen frame became `t_anchor`, which is the instant every
    #: other stream in the record is matched to.
    slot: str | None = None
    used: bool = True
    primary: bool = False
    #: The captured frame's shape, straight off the run's config. The viewer
    #: needs it to reserve the right box for a letterboxed pane; None on an
    #: older record that did not dump its camera specs.
    height: int | None = None
    width: int | None = None
    #: One image, two eyes, left eye in columns 0-639. Recorded whole — the
    #: recorder stores the wire JPEG verbatim and there is no half a JPEG — so
    #: the split happens at DISPLAY, in the browser. See `static/style.css`.
    binocular: bool = False
    #: A fisheye's lens square `[x0, y0, x1, y1]` and circle `[cx, cy, r]` on
    #: the recorded frame, which the viewer outlines; None otherwise.
    crop: list | None = None
    lens_circle: list | None = None

    def jpeg(self, i: int) -> bytes:
        """The i-th recorded frame's compressed bytes, exactly as the robot sent
        them — P9 stores the wire JPEG verbatim, so this is not a re-encode."""
        row = self.index[i]
        off, n = int(row["offset"]), int(row["nbytes"])
        # Positional read: the server answers image requests on several threads
        # at once, and a shared seek-then-read let one request read from where
        # another had just seeked (29% of concurrent responses were another
        # frame's bytes). `pread` has no shared position.
        try:
            fd = self.blob.fileno()
        except (AttributeError, OSError, io.UnsupportedOperation):
            fd = None                   # an in-memory blob (tests): serialise
        if fd is not None:
            return os.pread(fd, n, off)
        with _BLOB_LOCK:
            self.blob.seek(off)
            return self.blob.read(n)

    def close(self) -> None:
        self.blob.close()


@dataclass
class Rollout:
    """A loaded recording. Hold one at a time; `close()` releases the blobs."""

    path: Path
    manifest: dict
    state: np.ndarray
    cameras: dict[str, Camera]
    model: ModelFacts
    #: What P6 wrote into the rings, and the chunks it blended to get there.
    #: Sorted by `deadline` so a zero-order hold over them is a `searchsorted`.
    #: Empty arrays when the record predates a stream or the run wrote none —
    #: the commanded overlay simply has nothing to draw, which is the honest
    #: answer for homing and for a rollout that never reached ACTIVE.
    body_targets: np.ndarray
    hand_targets: np.ndarray
    chunks: np.ndarray
    #: P0's phase transitions, one row per edge, oldest first. Empty for a
    #: record without the stream. See `active_t`.
    session: np.ndarray

    @property
    def active_t(self) -> float | None:
        """When P0 entered ACTIVE — the instant the policy takes the arm.

        This is what the viewer opens on, because everything before it is the
        machine getting ready rather than the rollout: P1/P2 settling in CHECK,
        P0's homing ramp, P5 warming (tens of seconds for a policy that
        compiles), and then READY, which waits on a human and can run
        for minutes. On one recorded rollout that is 60 of the 80 seconds recorded.

        Deliberately the ACTIVE edge and not "the first frame where the arm
        moves". The arm moves during HOME too, under P0's ramp, and a
        motion-threshold rule would have to pick a threshold — it would open the
        viewer somewhere slightly different on every record, and be wrong on a
        rollout where the policy's first action is to hold still. The phase word
        is what the system itself says, and P0 is the one process that knows.

        `None` when the record has no session log, or has one that never reached
        ACTIVE (a run aborted in WARMUP, or stopped at the READY prompt). Both
        mean the same thing to the caller: there is no edge to open on, so show
        everything.
        """
        if not len(self.session):
            return None
        rows = self.session[self.session["session"] == _SESSION_ACTIVE]
        if not len(rows):
            return None
        return float(rows["t_observed"][0])

    @property
    def has_commands(self) -> bool:
        """Is there anything for the commanded overlay to draw?

        `chunks` is required only when the pose has to be RECONSTRUCTED from
        them. A record that carries `body_targets["pose"]` already holds the
        array P6 handed the IK, so it needs no chunk ring at all — and asking
        for one would refuse a perfectly complete record whose chunks happened to
        be truncated.
        """
        if not len(self.body_targets):
            return False
        return self.has_recorded_pose or bool(len(self.chunks))

    @property
    def has_recorded_pose(self) -> bool:
        """Does `body_targets` carry the pose P6 handed the IK?

        Asked of the STREAM's own
        dtype rather than of a version number or a date in the manifest, which
        is the same contract the rest of this package runs on: a record says
        what it contains, and the reader rebuilds each dtype from the
        manifest's `descr`. A record written by a branch that has the field, or
        by a future one that renames something else, answers correctly without
        anything here being told about it.

        When False the overlay falls back to reconstructing from the chunk ring,
        which is exact in sync and declines in async — see `scene._commands`.
        """
        return bool(len(self.body_targets)) and \
            "pose" in (self.body_targets.dtype.names or ())

    @property
    def has_pinned_waist(self) -> bool:
        """Does `body_targets` carry the trunk each solve was pinned at?

        Asked of the stream's dtype
        for exactly the reasons `has_recorded_pose` gives — a record says what it
        contains — and kept a separate question from that one because a record
        can carry the pose without this.

        When False the solved overlay can only be posed at the COMMANDED waist.
        That is not wrong; it
        is a different claim, and the switch that picks between them is disabled
        rather than silently answering the other question. See `scene._commands`.
        """
        return bool(len(self.body_targets)) and \
            "waist_ik" in (self.body_targets.dtype.names or ())

    @property
    def action_mode(self) -> str:
        """Which half of the action contract the policy spoke: `action_mode`."""
        return action_mode(self.chunks)

    @property
    def intervened(self) -> np.ndarray:
        """`(len(body_targets),)` bool: the tick was an operator's (PICO).

        `TARGET.intervene` is the value of `control.intervene` when P6 wrote the
        row, and it is ODD exactly while an intervention runs. Those
        rows carry an operator's EEF pose whatever the policy's own mode, which
        is why a joint-mode rollout can still draw a commanded pose there.
        All False on a record without the column.
        """
        body = self.body_targets
        if not len(body) or "intervene" not in (body.dtype.names or ()):
            return np.zeros(len(body), dtype=bool)
        return (body["intervene"].astype(np.int64) % 2) == 1

    def close(self) -> None:
        for cam in self.cameras.values():
            cam.close()

    def summary(self) -> dict:
        """The handful of facts worth showing beside the dropdowns.

        The run-identifying half comes from `discover.specifics`, which is what
        the listing showed for this recording before it was loaded. One reader,
        so pressing Load cannot change what the dashboard says the policy
        was. What is added here is everything that needed the record OPEN: how
        many state samples it holds, which robot it was posed as, the action
        mode, and where the commanded overlay's poses came from.
        """
        m = self.model
        return {
            **discover.specifics(self.manifest),
            "state_samples": int(len(self.state)),
            "action_mode": self.action_mode,
            "intervened_ticks": int(self.intervened.sum()),
            "model": Path(m.urdf).name,
            "urdf": m.urdf,
            "urdf_source": m.urdf_source,
            "robot_source": m.source,
            "tcp": None if m.tcp is None else [round(float(v), 6) for v in m.tcp],
            "tcp_source": m.tcp_source,
            "robot_notes": list(m.notes),
            "waist_joint_names": m.waist_joint_names,
            "model_from_record": m.source == "record",
            # Where the ghost's poses came from. Worth showing: "recorded" is
            # the pose P6 actually solved, in either mode; "chunks" is a
            # reconstruction that is exact in sync and empty in async. The two
            # look identical on screen and are very different claims.
            "command_source": ("recorded" if self.has_recorded_pose
                               else "chunks" if self.has_commands else "none"),
            # Whether the solved overlay can be posed on the trunk the IK was
            # pinned to, or only on the trunk that was commanded. Same kind of
            # fact as `command_source` and shown for the same reason: the two
            # pictures differ by the whole gravity droop and look equally
            # plausible.
            "has_pinned_waist": self.has_pinned_waist,
        }


def load(run_dir: Path | str) -> Rollout:
    """Read a recording. Raises if it has no manifest — see `discover`."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{run_dir.name} has no manifest.json, so nothing records what its "
            f"streams contain. It cannot be read.")

    rec = record_read.load(run_dir)
    manifest = rec["manifest"]
    # The robot first: a record whose robot cannot be rebuilt here
    # (`record_robot.UnsupportedRecord`) is refused before any file is opened.
    model = _model_facts(manifest)

    state = rec.get("state")
    if state is None or not len(state):
        raise ValueError(f"{run_dir.name} has no state samples to replay.")

    roles = camera_roles(manifest)
    cameras: dict[str, Camera] = {}
    for name in _camera_order(manifest, roles):
        spec = (manifest.get("cameras") or {}).get(name)
        if spec is None:
            continue
        index_path = run_dir / spec["index"]
        blob_path = run_dir / spec["blob"]
        if not index_path.exists() or not blob_path.exists():
            continue
        dtype = np.dtype([tuple(f) for f in spec["index_descr"]])
        index = np.fromfile(index_path, dtype=dtype)
        # An EMPTY index is how the exo shows up, and it is not a fault: the
        # RealSense source publishes no JPEG (`runtime/drivers/cameras.py`), so P2 records
        # every exo frame as "not captured" and the blob holds nothing to seek
        # into. Skipping it here is what keeps the exo out of the panes without
        # this module holding a list of camera names again.
        if not len(index):
            continue
        role = roles.get(name, {})
        cameras[name] = Camera(name, index, open(blob_path, "rb"),
                               int(spec.get("dropped", 0)),
                               slot=role.get("slot"),
                               used=bool(role.get("used", True)),
                               primary=bool(role.get("primary", False)),
                               binocular=bool(role.get("binocular", False)),
                               height=role.get("height"),
                               width=role.get("width"),
                               crop=role.get("crop"),
                               lens_circle=role.get("lens_circle"))

    if not cameras:
        raise ValueError(
            f"{run_dir.name} has no readable camera index. The blobs may hold "
            f"JPEG bytes, but without an index nothing says where a frame "
            f"starts or when it arrived.")

    def stream(name: str) -> np.ndarray:
        got = rec.get(name)
        return got if got is not None else np.empty(0)

    body = stream("body_targets")
    hand = stream("hand_targets")
    if len(body):
        # By deadline, not by `k` and not as written: the overlay asks "which
        # target was in force at this instant", and only `deadline` answers
        # that. `k` is not a proxy — P6 re-anchors to `k_exec + 2` on every new
        # chunk, so ticks can be revisited, and the ring is swept in observation
        # order rather than execution order.
        body = body[np.argsort(body["deadline"], kind="stable")]
    if len(hand):
        hand = hand[np.argsort(hand["deadline"], kind="stable")]

    return Rollout(run_dir, manifest, state, cameras, model,
                   body_targets=body, hand_targets=hand,
                   chunks=stream("chunks"), session=stream("session"))
