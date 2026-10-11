"""The IK mode's producer: the infra's own solvers, replayed against a record.

    python -m g1d_deploy.dashboards.inspect.producers.ik_bench RECORD --methods arm_planner,arm_qp
    python -m g1d_deploy.dashboards.inspect.producers.ik_bench --list-json

For each named solver (`g1d_deploy.runtime.ik.SOLVERS`, resolved BY NAME at run
time — the registry is the solver package's, not this file's), every recorded tick
that carries a Cartesian target is solved again the way P6 solves it:

* ticks are grouped by the chunk that covered them (newest `k_first_action`
  at or before the tick), and a solver that plans
  chunks (`plans_chunks`) is handed each group's poses first, with the trunk
  measured at its first tick — `p6_target._plan_trunk`;
* each tick is then `solve(pose, waist, seed, budget=dt)`, on the planned
  trunk where there is a plan and on the trunk P6 recorded it solved in
  (`waist_ik`) where there is not, seeded with the previous solution;
* the result is scored at the recorded robot's virtual TCP against the
  commanded one, and so is the configuration that actually flew (`recorded`).

A joint-mode rollout has Cartesian targets only where an operator intervened
(PICO poses are EEF); those are benched, and a joint rollout with no
intervention has nothing to bench and says so.

Output, merged into what is there when the ticks agree (a method solved earlier
is kept, one solved again is replaced):

    <record>/ik_bench.npz      version, k, methods, meta, <m>/{arm, waist,
                               column, base, err, tcp_err, rot_err, tick_ms},
                               recorded/{err, tcp_err}

`overlays.py` reads it (BENCH_NAME / BENCH_VERSION) and `scene.py` poses every
method through one loop.

Spawned by the IK mode (`modes/ik.py`) as a subprocess; runnable by hand. For a
user: `docs/deploying_a_policy/rollout_inspection.md`, "ik", and `docs/infra_logic/ik_solvers.md`, "Comparing solvers
on recordings".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from g1d_deploy.dashboards.inspect.overlays import BENCH_NAME, BENCH_VERSION

def methods() -> tuple[str, ...]:
    """The infra's own solvers: exactly what `runtime/ik` `SOLVERS` ships, read
    at run time so a solver added there is benched here with no change."""
    from g1d_deploy.runtime import ik as ik_solvers
    return tuple(ik_solvers.SOLVERS)


def registry() -> dict:
    """`{"methods": [{name, mode, available, why, describe}], "error": ""}`."""
    try:
        from g1d_deploy.runtime import ik as ik_solvers
    except Exception as exc:                                # noqa: BLE001
        return {"methods": [], "error": f"g1d_deploy.runtime.ik: {exc}"}
    rows = []
    for name, (_spec, mode) in ik_solvers.SOLVERS.items():
        row = {"name": name, "mode": mode, "available": False, "why": "",
               "describe": ""}
        try:
            factory = ik_solvers.resolve(name)
            row.update(available=True,
                       describe=(factory.__doc__ or "").strip().split("\n")[0])
        except Exception as exc:                            # noqa: BLE001
            row["why"] = str(exc)
        rows.append(row)
    return {"methods": rows, "error": ""}


def _ticks(rollout):
    """The body-target rows with a Cartesian target, one per tick (the last
    write of a re-solved tick wins), in tick order."""
    body = rollout.body_targets
    if not len(body) or "pose" not in (body.dtype.names or ()):
        return body[:0]
    ok = np.isfinite(body["pose"]).all(axis=(1, 2)) & (body["k"] >= 1)
    rows = body[ok]
    if not len(rows):
        return rows
    rows = rows[np.argsort(rows["k"], kind="stable")]
    last = np.r_[rows["k"][1:] != rows["k"][:-1], True]
    return rows[last]


def _groups(ticks, chunks) -> list[np.ndarray]:
    """Tick indices grouped by the chunk covering them (P6's newest rule)."""
    if not len(chunks):
        return [np.arange(len(ticks))]
    live = chunks[chunks["chunk_id"] > 0]
    first = np.sort(live["k_first_action"].astype(np.int64))
    owner = np.searchsorted(first, ticks["k"].astype(np.int64), side="right") - 1
    out, start = [], 0
    for i in range(1, len(ticks) + 1):
        if i == len(ticks) or owner[i] != owner[start] \
                or ticks["k"][i] != ticks["k"][i - 1] + 1:
            out.append(np.arange(start, i))
            start = i
    return out


def recorded_waist(ticks) -> np.ndarray:
    """`(n, 3)`: the trunk each tick was solved on — `waist_ik` where the
    record has it, else the commanded waist."""
    waist = np.asarray(ticks["q"], np.float64)[:, 14:17].copy()
    if "waist_ik" in (ticks.dtype.names or ()):
        pin = np.asarray(ticks["waist_ik"], np.float64)
        got = np.isfinite(pin).all(axis=1)
        waist[got] = pin[got]
    return waist


def recorded_residuals(rollout, ticks=None) -> tuple:
    """The configuration that FLEW, scored against its commanded pose the way
    the bench scores a method: `(ticks, err_m, tcp_m, rot_rad)`, per tick,
    worst hand. `tcp` is at the recorded robot's virtual TCP (its flange when
    the robot file names none)."""
    ticks = _ticks(rollout) if ticks is None else ticks
    n = len(ticks)
    err, tcp, rot = np.zeros(n), np.zeros(n), np.zeros(n)
    if n:
        score = _Score(rollout.model.urdf, rollout.model)
        q = np.asarray(ticks["q"], np.float64)
        waist = recorded_waist(ticks)
        poses = np.asarray(ticks["pose"], np.float64)
        for i in range(n):
            err[i], tcp[i], rot[i] = score(q[i, :14], waist[i], poses[i])
    return ticks, err, tcp, rot


def _rot_angle(qa, qb) -> float:
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return 2.0 * float(np.arccos(min(1.0, d)))


class _Score:
    """Flange and TCP residuals of a configuration against a commanded pose."""

    def __init__(self, urdf, facts):
        from g1d_deploy.robot.kinematics import ForwardKinematics
        from g1d_deploy.dashboards.inspect.scene import tcp_point

        self.fk = ForwardKinematics(urdf)
        self.tcp = tcp_point(facts)

    def _point(self, pose7):
        from g1d_deploy.dashboards.inspect.scene import _pose7_to_mat
        T = _pose7_to_mat(pose7)
        return T[:3, 3] + T[:3, :3] @ self.tcp

    def __call__(self, arm14, waist3, pose):
        got = self.fk.flange(np.asarray(arm14, np.float64),
                             np.asarray(waist3, np.float64))
        err = max(float(np.linalg.norm(got[h, :3] - pose[h, :3])) for h in (0, 1))
        tcp = max(float(np.linalg.norm(self._point(got[h]) - self._point(pose[h])))
                  for h in (0, 1))
        rot = max(_rot_angle(got[h, 3:], pose[h, 3:]) for h in (0, 1))
        return err, tcp, rot


def solve(record_dir, methods: list[str], *, progress=None, log=print) -> Path:
    """Bench `methods` on the record; merge into `<record>/ik_bench.npz`."""
    from g1d_deploy.runtime import ik as ik_solvers
    from g1d_deploy.dashboards.inspect import loader, record_robot
    from g1d_deploy.dashboards.inspect.modes.replay.record_config import deploy_config

    progress = progress or (lambda **_: None)
    record_dir = Path(record_dir).expanduser().resolve()
    rollout = loader.load(record_dir)
    try:
        ticks = _ticks(rollout)
        if not len(ticks):
            raise ValueError(
                "this record has no Cartesian targets to re-solve"
                + (" (a joint-mode rollout with no operator intervention)"
                   if rollout.action_mode == "joint" else ""))
        cfg = deploy_config(rollout.manifest)
        hw = record_robot.hardware(rollout.manifest)
        facts = record_robot.from_manifest(rollout.manifest)
        score = _Score(facts.urdf, facts)
        dt = cfg.dt
        n = len(ticks)
        q_rec = np.asarray(ticks["q"], np.float64)
        waist_rec = recorded_waist(ticks)
        poses = np.asarray(ticks["pose"], np.float64)
        groups = _groups(ticks, rollout.chunks)

        arrays: dict[str, np.ndarray] = {}
        meta: dict = {"methods": {}, "record": record_dir.name,
                      "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rec_err = np.zeros(n)
        rec_tcp = np.zeros(n)
        for i in range(n):
            rec_err[i], rec_tcp[i], _ = score(q_rec[i, :14], waist_rec[i], poses[i])
        arrays["recorded/err"] = rec_err.astype(np.float32)
        arrays["recorded/tcp_err"] = rec_tcp.astype(np.float32)

        for mi, name in enumerate(methods):
            progress(phase="solving", method=name, method_index=mi + 1,
                     n_methods=len(methods))
            # Built as P6 builds it — with the robot file's TCP, which every
            # shipped solver drives when it is given — so the bench solves the
            # problem the rollout solved, not the flange-only one.
            solver = ik_solvers.build(cfg, hw, name=name)
            arm = np.zeros((n, 14))
            waist = np.zeros((n, 3))
            err = np.zeros(n)
            tcp = np.zeros(n)
            rot = np.zeros(n)
            ms = np.zeros(n)
            seed = q_rec[0, :14].copy()
            for g in groups:
                plan = None
                if getattr(solver, "plans_chunks", False):
                    try:
                        plan = solver.plan_chunk(
                            poses[g], waist_rec[g[0]], seed,
                            budget=max((len(g) - 1) * dt, dt))
                    except Exception as exc:                # noqa: BLE001
                        log(f"[ik_bench] {name}: plan failed ({exc!r}); "
                            f"measured trunk for this chunk")
                        plan = None
                for j, i in enumerate(g):
                    w = (np.asarray(plan[j], np.float64)
                         if plan is not None and j < len(plan) else waist_rec[i])
                    t0 = time.perf_counter()
                    q, _e, ok = solver.solve(poses[i], w, seed, budget=dt)
                    ms[i] = (time.perf_counter() - t0) * 1e3
                    if ok:
                        seed = np.asarray(q, np.float64).reshape(14)
                    arm[i], waist[i] = seed, w
                    err[i], tcp[i], rot[i] = score(seed, w, poses[i])
            for field, arr in (("arm", arm), ("waist", waist),
                               ("column", np.zeros(n)), ("base", np.zeros((n, 3))),
                               ("err", err), ("tcp_err", tcp), ("rot_err", rot),
                               ("tick_ms", ms)):
                arrays[f"{name}/{field}"] = arr.astype(np.float32)
            meta["methods"][name] = {
                "mode": ik_solvers.mode_of(name),
                "describe": f"g1d_deploy.runtime.ik {name!r}",
                "tcp_mean_mm": round(float(tcp.mean() * 1e3), 3)}
            log(f"[ik_bench] {name}: TCP mean {tcp.mean() * 1e3:.2f} mm, "
                f"p95 {np.percentile(tcp, 95) * 1e3:.2f} mm")

        out = record_dir / BENCH_NAME
        k = ticks["k"].astype(np.int64)
        names = list(methods)
        if out.exists():
            try:
                with np.load(out, allow_pickle=False) as old:
                    if (int(old["version"][0]) == BENCH_VERSION
                            and np.array_equal(old["k"], k)):
                        old_meta = json.loads(str(old["meta"][0]))
                        for m in (str(x) for x in old["methods"]):
                            if m in names:
                                continue
                            names.insert(0, m)
                            for key in old.files:
                                if key.startswith(m + "/"):
                                    arrays[key] = old[key]
                            meta["methods"][m] = old_meta.get("methods", {}).get(m, {})
            except Exception:                               # noqa: BLE001
                pass
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez(tmp, version=np.array([BENCH_VERSION]), k=k,
                 methods=np.array(names), meta=np.array([json.dumps(meta)]),
                 **arrays)
        os.replace(tmp, out)
        progress(phase="done")
        return out
    finally:
        rollout.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m g1d_deploy.dashboards.inspect.producers.ik_bench",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("record", nargs="?")
    ap.add_argument("--methods", default=None,
                    help="comma-separated; default: every ik_solvers.SOLVERS name")
    ap.add_argument("--list-json", action="store_true")
    ap.add_argument("--progress-file", default=None)
    args = ap.parse_args(argv)
    if args.list_json:
        print(json.dumps(registry()))
        return 0
    if not args.record:
        ap.error("a record directory is required")

    def progress(**fields):
        if args.progress_file:
            p = Path(args.progress_file)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(fields))
            os.replace(tmp, p)

    try:
        names = (list(methods()) if args.methods is None
                 else [m for m in args.methods.split(",") if m])
        solve(args.record, names,
              progress=progress)
    except Exception as exc:                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
