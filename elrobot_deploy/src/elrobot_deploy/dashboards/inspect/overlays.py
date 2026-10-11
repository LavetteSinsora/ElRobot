"""The IK bench's sidecar beside a recording, read as one view.

`ik_bench.npz` (written by `producers/ik_bench.py`) holds, per solved method and
per tick, the arm joints, the trunk to pose them on, the column and base, and
the residuals, under `<name>/<field>` keys with a shared `k` array tying rows
to body targets. `scene.py` poses every method through one code path.

The layout is the two constants below; the writer imports them from here, so
the two cannot disagree.

Read by `scene.py` at Load; written by `producers/ik_bench.py`. For a user:
`docs/deploying_a_policy/rollout_inspection.md`, "ik".
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

#: The bench sidecar's name and the layout version this reader understands.
BENCH_NAME = "ik_bench.npz"
BENCH_VERSION = 1


def _load_npz(path, version):
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if int(z["version"][0]) != version:
            return None
        return z
    except Exception:                                     # noqa: BLE001
        return None


class Overlays:
    """Dict-like over the sidecar: `"k"`, `"tiers"`, `"off/tcp_err"`, `"<name>/arm"` and so on. Plus `meta`,
    which says what each name is for the switch's tooltip."""

    def __init__(self, arrays: dict, tiers: list[str], meta: dict):
        self._a = arrays
        self.tiers = list(tiers)
        self.meta = meta

    def __getitem__(self, key):
        if key == "tiers":
            return np.array(self.tiers)
        return self._a[key]

    def __contains__(self, key):
        return key == "tiers" or key in self._a

    def get(self, key, default=None):
        return self[key] if key in self else default


def load(run_dir) -> Overlays | None:
    """Every solved method this recording has, or None when it has none."""
    run_dir = pathlib.Path(run_dir)
    bench = _load_npz(run_dir / BENCH_NAME, BENCH_VERSION)
    if bench is None:
        return None

    arrays, tiers, meta = {}, [], {"names": {}, "notes": []}
    fields = ("arm", "waist", "column", "base", "err", "tcp_err", "rot_err")

    if bench is not None:
        arrays["k"] = np.asarray(bench["k"], np.int64)
        try:
            bmeta = json.loads(str(bench["meta"][0]))
        except Exception:                                 # noqa: BLE001
            bmeta = {}
        for name in (str(t) for t in bench["methods"]):
            for f in fields:
                arrays[f"{name}/{f}"] = bench[f"{name}/{f}"]
            if f"{name}/tick_ms" in bench:
                arrays[f"{name}/tick_ms"] = bench[f"{name}/tick_ms"]
            tiers.append(name)
            info = (bmeta.get("methods") or {}).get(name, {})
            meta["names"][name] = {
                "source": "bench", "mode": info.get("mode", ""),
                "describe": info.get("describe", ""),
                "tcp_mean_mm": info.get("tcp_mean_mm")}
        # The deployed answer scored at the TCP — the bench measures the
        # recorded configuration itself, which is the truest "deployed IK" row.
        arrays["off/tcp_err"] = bench["recorded/tcp_err"]
        arrays["off/err"] = bench["recorded/err"]
        arrays["recorded/err"] = bench["recorded/err"]

    if not tiers:
        return None
    return Overlays(arrays, tiers, meta)


__all__ = ["BENCH_NAME", "BENCH_VERSION", "Overlays", "load"]
