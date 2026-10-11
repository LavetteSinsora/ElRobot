"""One executor run as joint traces on a common clock.

A replay of a rollout — on the robot or through a fitted plant, by whatever
producer the executor mode is configured with — is an ordinary P9 record. What the executor
mode draws from it is per joint: the measured angle, the 500 Hz command P7
sent (the `command` stream), the
target line that command was shaped from, and the tracking error; plus the
knot deadlines and the chunk arrivals as marks.

**Time zero is when the arm starts EXECUTING chunk 1** (its first knot's
deadline minus P6's two-tick anchor), not the run's start: P6 is spawned at
ACTIVE and its IK import can hold chunk 1 for seconds, a delay that differs
run to run and says nothing about the servo. Runs of one source then overlay
knot for knot, and the loaded rollout's own playhead maps onto them through
the same rule (`t_zero`).

Measured samples are put on OUR clock through `LowState.tick` when the record
has it: `t_recv` carries delivery bursts of up to ~200 ms on some runs, the
tick does not. That estimate lives in `dashboards/inspect/clock.py`, which the
viewer's own timeline is built on — one implementation, so the executor's
traces and the rollout beside them cannot sit on two slightly different clocks.

Used by the executor mode (`modes/executor/__init__.py`).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from g1d_deploy.rollout_record_utils.reader import body_joint_index
from g1d_deploy.rollout_record_utils.reader import load as load_record
from g1d_deploy.dashboards.inspect import clock as clock_mod

ARM = 14
GRID_HZ = 1000.0
#: 1 kHz grid -> 250 Hz on the wire: four points per 30 Hz tick is plenty to
#: see a 500 Hz command's shape, and a fifth of the bytes.
DECIM = 4

JOINT_NAMES = [f"{s}_{n}" for s in ("L", "R")
               for n in ("sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_roll", "wr_pitch", "wr_yaw")]


def robot_time(st) -> np.ndarray:
    """Sample times on our clock (see the module docstring).

    Strictly increasing, which `clock.estimate` does not promise: it guarantees
    non-decreasing, and two samples sharing an instant is fine for a zero-order
    hold and not fine for the interpolation this module's traces are resampled
    with. Nudged by a microsecond rather than de-duplicated, so the sample count
    and every array indexed by it stay aligned.
    """
    c = clock_mod.estimate(st)
    if c is None:
        return np.asarray(st["t_recv"], np.float64)
    t = c.t.copy()
    for i in np.flatnonzero(np.diff(t) <= 0):
        t[i + 1] = t[i] + 1e-6
    return t


def _dedupe_targets(tg):
    """P6 republishes a tick it re-solves; keep the LAST row per `k`."""
    order = np.argsort(tg["k"], kind="stable")
    tg = tg[order]
    last = np.r_[tg["k"][1:] != tg["k"][:-1], True]
    return tg[last]


def t_zero(rec: dict) -> float:
    """When the arm starts executing chunk 1, absolute (module docstring)."""
    dc = rec["manifest"].get("deploy_config", {})
    dt = float(dc.get("dt", 1 / 30))
    tg = _dedupe_targets(rec["targets"])
    first = tg["k"] == tg["k"].min()
    return float(tg["deadline"][first][0]) - float(tg["k"][first][0] + 1) * dt


@dataclass
class Run:
    dir: pathlib.Path
    manifest: dict
    labels: dict
    servo: dict
    t0: float
    t: np.ndarray                   # 1 kHz grid, seconds since t0
    qm: np.ndarray                  # measured arm (n, 14)
    qc: np.ndarray | None           # sent command (n, 14), ZOH; None without a command stream
    qlin: np.ndarray | None         # the target line the command was shaped from
    knot_t: np.ndarray              # deadlines since t0
    knot_q: np.ndarray              # (m, 17)
    arrivals: np.ndarray            # chunk ring writes since t0 (chunks 2..)

    @property
    def has_command(self) -> bool:
        return self.qc is not None

    def describe(self) -> dict:
        labels, man = self.labels, self.manifest
        kind = labels.get("robot", "g1d")
        kind = "sim" if kind == "sim" else "robot"
        return {
            "name": self.dir.name, "dir": str(self.dir), "kind": kind,
            "variant": labels.get("variant", "?"),
            "variant_note": labels.get("variant_note", ""),
            "source_name": labels.get("source_name", ""),
            "created": man.get("created", ""),
            "servo": self.servo, "has_command": self.has_command,
            "ik": (man.get("deploy_config", {}).get("ik_solver")
                   or man.get("deploy_config", {}).get("ik")),
            "duration_s": round(float(self.t[-1]), 3),
        }

    def series(self, joint: int) -> dict:
        j = int(joint)
        s = slice(None, None, DECIM)
        out = {
            "t": _round(self.t[s], 4),
            "q_meas": _round(self.qm[s, j]),
            "knot_t": _round(self.knot_t, 4),
            "knot_q": _round(self.knot_q[:, j]),
            "arrivals": _round(self.arrivals, 4),
        }
        if self.qc is not None:
            out["q_cmd"] = _round(self.qc[s, j])
            out["q_line"] = _round(self.qlin[s, j])
            out["track_err_mrad"] = _round((self.qm - self.qc)[s, j] * 1e3, 3)
        return out


    def tracking(self) -> dict:
        """Servo tracking error per executed tick: `(t, err_deg)`.

        For each knot (a tick's deadline, `knot_t`), the mean over the 14 arm
        joints of |measured - commanded| angle, averaged over the 1 kHz
        samples from that deadline to the next (the last tick: one tick
        long). Commanded is the 500 Hz command P7 sent (`command` stream);
        without one there is nothing to track against and `err_deg` is empty.
        `t` is seconds on this run's clock (`t_zero`).
        """
        kt = np.asarray(self.knot_t, np.float64)
        if self.qc is None or not len(kt):
            return {"t": _round(kt, 4), "err_deg": [], "has_command": False}
        per = np.mean(np.abs(self.qm - self.qc), axis=1)          # (n,), rad
        dt = float(np.median(np.diff(kt))) if len(kt) > 1 else 1 / 30
        edges = np.r_[kt, kt[-1] + dt]
        lo = np.searchsorted(self.t, edges[:-1], side="left")
        hi = np.searchsorted(self.t, edges[1:], side="left")
        csum = np.r_[0.0, np.cumsum(per)]
        cnt = hi - lo
        err = np.full(len(kt), np.nan)
        got = cnt > 0
        err[got] = (csum[hi[got]] - csum[lo[got]]) / cnt[got]
        return {"t": _round(kt, 4), "err_deg": _round(np.degrees(err), 4),
                "has_command": True}


def _round(a, nd=5):
    a = np.asarray(a, dtype=float)
    out = np.round(a, nd)
    return [None if not np.isfinite(x) else x for x in out.tolist()]


def load(run_dir) -> Run:
    run_dir = pathlib.Path(run_dir)
    rec = load_record(run_dir)
    man = rec["manifest"]
    dc = man.get("deploy_config", {})
    labels = dict(dc.get("labels") or {})
    servo = dict(dc.get("servo") or {})
    jidx = body_joint_index(man)
    st = rec["state"]
    ts = robot_time(st)
    keep = np.r_[True, np.diff(ts) > 0]
    ts, q = ts[keep], np.asarray(st["q"], float)[keep]

    ch = rec["chunks"]
    ch = ch[np.argsort(ch["chunk_id"], kind="stable")]
    ch = ch[np.r_[True, ch["chunk_id"][1:] != ch["chunk_id"][:-1]]]
    tg = _dedupe_targets(rec["targets"])
    t0 = t_zero(rec)
    n_src = int(labels.get("source_chunks", 0)) or len(ch)

    t_lo = t0 - 0.3
    t_hi = min(float(ts[-1]), float(tg["deadline"][-1]) + 0.5)
    grid = np.arange(t_lo, t_hi, 1.0 / GRID_HZ)
    qa = q[:, jidx[:ARM]]
    qm = np.stack([np.interp(grid, ts, qa[:, j]) for j in range(ARM)], 1)

    cmd = rec.get("command")
    qc = qlin = None
    if cmd is not None and len(cmd) > 10:
        ct = np.asarray(cmd["t"], float)
        i = np.clip(np.searchsorted(ct, grid, side="right") - 1, 0, len(ct) - 1)
        qc = np.asarray(cmd["q"], float)[i, :ARM]
        qlin = np.asarray(cmd["q_lin"], float)[i, :ARM]

    return Run(dir=run_dir, manifest=man, labels=labels, servo=servo, t0=t0,
               t=grid - t0, qm=qm, qc=qc, qlin=qlin,
               knot_t=np.asarray(tg["deadline"], float) - t0,
               knot_q=np.asarray(tg["q"], float),
               arrivals=np.asarray(ch["t_chunk_written"][1:n_src], float) - t0)


__all__ = ["ARM", "DECIM", "JOINT_NAMES", "Run", "load", "robot_time", "t_zero"]
