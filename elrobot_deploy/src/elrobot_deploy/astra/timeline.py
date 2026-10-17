"""`timeline.sqlite`: one row per tick (the anchor camera's frame, 30 Hz), what Astra queries.

One column per number, the unit in its name, for everything Astra may filter or sort on; the
joint vectors as JSON arrays (`json_extract(meas_q_rad, '$[3]')`, names in table `joints`);
`column_docs` explains every column from inside the database. WAL, so the harness appends
(one transaction a fraction of a second) while Astra reads.

The commanded trajectory lives here and nowhere else: `cmd_*` is the pose the IK was asked
for at that tick (P6's input, the composed path), `meas_*` the forward kinematics of the
measured joints, both in Astra's base frame.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

POSE = (("x_m", "m"), ("y_m", "m"), ("z_m", "m"), ("roll_deg", "deg"), ("pitch_deg", "deg"),
        ("yaw_deg", "deg"), ("gripper_mm", "mm"))
PHASES = ("executing", "holding")


def columns(cameras) -> list[tuple[str, str, str, str]]:
    """(name, SQL type, unit, meaning) of every column, in order."""
    cols = [
        ("tick", "INTEGER PRIMARY KEY", "",
         ("the anchor camera's frame index, 30 Hz, from the session start, counting through "
          "holds")),
        ("t", "REAL NOT NULL", "s", "seconds since the session start (tick 0)"),
        ("action_id", "INTEGER", "",
         ("the action whose commands drive this tick; NULL = none (holding, also while you "
          "think)")),
        ("phase", "TEXT NOT NULL CHECK (phase IN ('executing','holding'))", "",
         "executing while an action runs, holding otherwise"),
    ]
    for cam in cameras:
        cols.append((f"{cam}_path", "TEXT", "",
                     (f"the {cam} camera's frame at this tick, relative to the session "
                      f"folder (NULL = none)")))
    for prefix, what in (("meas", "measured: forward kinematics of the measured joints"),
                         ("cmd", "commanded: the pose the IK was asked for (the planned path)")):
        for suffix, unit in POSE:
            q = "gripper opening" if suffix == "gripper_mm" else f"virtual TCP {suffix[:-2]}"
            if suffix.endswith("_deg"):
                q = f"virtual TCP {suffix[:-4]} (R = Rz(yaw) Ry(pitch) Rx(roll))"
            cols.append((f"{prefix}_{suffix}", "REAL", unit, f"{q}, base frame; {what}"))
    cols += [
        ("ik_residual_mm", "REAL", "mm",
         "how far the IK's joint solution leaves the TCP from the commanded position"),
        ("ik_residual_deg", "REAL", "deg", "... and from the commanded orientation"),
        ("meas_q_rad", "TEXT", "rad", "measured joints, JSON array in `joints` order"),
        ("meas_dq_rad_s", "TEXT", "rad/s", "measured joint speeds, JSON array"),
        ("meas_load_frac", "TEXT", "",
         "measured load per joint, fraction of stall torque, signed, JSON array"),
        ("cmd_q_rad", "TEXT", "rad", "the joints the servos were commanded, JSON array"),
    ]
    return cols


class Timeline:
    """The writer. Only the harness's ingest thread touches it."""

    def __init__(self, path: Path, cameras, joint_names):
        self.path = Path(path)
        self.cols = columns(cameras)
        self.names = [c[0] for c in self.cols]
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("CREATE TABLE timeline (" + ",\n  ".join(f"{n} {t}" for n, t, _, _ in
                                                                  self.cols) + ")")
        self.db.execute("CREATE INDEX timeline_action ON timeline(action_id)")
        self.db.execute("CREATE TABLE joints (idx INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        self.db.execute("CREATE TABLE column_docs (name TEXT PRIMARY KEY, unit TEXT, "
                        "meaning TEXT NOT NULL)")
        self.db.executemany("INSERT INTO joints VALUES (?, ?)", list(enumerate(joint_names)))
        self.db.executemany("INSERT INTO column_docs VALUES (?, ?, ?)",
                            [(n, u, m) for n, _, u, m in self.cols])
        self.db.commit()
        self._insert = (f"INSERT OR REPLACE INTO timeline ({', '.join(self.names)}) VALUES "
                        f"({', '.join('?' for _ in self.names)})")
        self._pending: list[tuple] = []

    def add(self, row: dict) -> None:
        values = []
        for name in self.names:
            v = row.get(name)
            if isinstance(v, (list, tuple)):
                v = json.dumps([round(float(x), 5) for x in v])
            values.append(v)
        self._pending.append(tuple(values))

    def commit(self) -> None:
        if self._pending:
            self.db.executemany(self._insert, self._pending)
            self.db.commit()
            self._pending.clear()

    def relabel(self, ticks: range, action_id, phase: str) -> None:
        """Change who drove a span of ticks already written (an action's end was learnt late)."""
        self.commit()
        self.db.execute("UPDATE timeline SET action_id = ?, phase = ? WHERE tick BETWEEN ? AND ?",
                        (action_id, phase, ticks.start, ticks.stop - 1))
        self.db.commit()

    def close(self) -> None:
        self.commit()
        self.db.close()


__all__ = ["PHASES", "Timeline", "columns"]
