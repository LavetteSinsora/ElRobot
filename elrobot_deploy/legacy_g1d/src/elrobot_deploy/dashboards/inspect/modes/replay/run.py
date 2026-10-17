"""Re-run a recorded rollout's observations through a policy, offline.

    <policy python> -m elrobot_deploy.dashboards.inspect.modes.replay.run --record DIR
    elrobot_deploy replay-policy --record DIR     (the same, this interpreter)
        [--entry "pkg.mod:Class" | "path/file.py:Class"]
        [--checkpoint STEP_DIR | --run RUN_DIR --step N|latest|alias]
        [--prompt TEXT] [--sample-steps N] [--seed N] [--out DIR]
    elrobot_deploy replay-policy --record DIR --check-prompt TEXT
        [--entry ...] [--checkpoint ... | --run ... --step ...]

Every default is the record's own: the policy entry, the checkpoint (if any),
the prompt and the rollout settings the manifest says the rollout ran with.
What you pass replaces that one thing, so `--sample-steps 5` is "this rollout,
this policy, five sampler steps".

WHAT IT DOES, in P5's order (`runtime/processes/p5_policy.py`), once per recorded
replan:

1. load the class (`policy_contract/loading.py::load_entry`), check its declarations against the
   rollout's cameras (`check_against`), construct it with the checkpoint, the
   recorded prompt and the RECORDED robot (`record_robot.hardware`), bind the
   dummy observation and reader, `warmup()`;
2. take the rollout's settings as P5 does at START (`RolloutLatch._take`):
   `set_prompt` when the prompt differs — which runs `check_prompt_length`, so a
   prompt the policy cannot fit is refused with the dashboard's own sentence —
   and `on_settings` when the settings differ, then one dummy `infer`;
3. for every replan: rebuild the observation the robot saw (`anchors`,
   `observe`), `infer(obs, reader)`, and turn the actions into a chunk with
   P5's own `to_chunk` — validation, deadline shift, clips, the sync cut;
4. write `<record>/policy_replays/<id>/` (`store`), manifest last.

`--check-prompt TEXT` does step 1's load and construction only — never
`warmup()`, never `infer` — and prints `check_prompt`'s JSON verdict on TEXT:
the dashboard's "Check prompt" button. It hides the GPU from itself
(`CPU_ONLY_ENV`), so unlike a replay it is not refused while a live deployment
runs.

**It runs under the POLICY's interpreter** (the deploy file's `[policy]
python`, recorded in the manifest;
`docs/deploying_a_policy/policy_contract.md`, "The adapter's own Python"): the
dashboard spawns it there, by module path so that no
dashboard dependency is imported on the way. `--describe-entry` and
`--check-prompt` need only elrobot_deploy's minimal core in that venv; a full
replay also decodes the recorded JPEGs and draws its summary through the
model, so it needs the `replay` extra there too (`opencv-python-headless`,
`mujoco` — both have numpy-2 wheels).

It never touches a robot, an arena or a ring. It does take the GPU when the
policy does, which is why the dashboard refuses to start one while a live
deployment is running (`dashboards/web/gpu.py`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--record", required=True, metavar="DIR",
                   help="the rollout record directory (holds manifest.json)")
    p.add_argument("--entry", default=None, metavar="SPEC",
                   help="policy class; default: the one the record ran")
    p.add_argument("--checkpoint", default=None, metavar="DIR",
                   help="a step directory; default: the record's. '' = none")
    p.add_argument("--run", default=None, metavar="DIR",
                   help="a run directory; its steps are numeric subdirectories, "
                        "or whatever the class's resolve_checkpoint says")
    p.add_argument("--step", default=None,
                   help="with --run: a number, 'latest', or a policy alias")
    p.add_argument("--prompt", default=None, help="default: the rollout's prompt")
    p.add_argument("--sample-steps", type=int, default=None, metavar="N",
                   help="default: the rollout's active_config sample_steps")
    p.add_argument("--settings", default=None, metavar="JSON",
                   help="a partial RolloutSettings dict, e.g. "
                        "'{\"sample_steps\": 5}'")
    p.add_argument("--seed", type=int, default=None,
                   help="seeds Python's and numpy's RNGs (a JAX policy seeds "
                        "its own); default: drawn, and recorded")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="where to write; default <record>/policy_replays/<id>")
    p.add_argument("--progress-file", default=None, metavar="PATH",
                   help="rewritten atomically with {phase, chunk, n_chunks}")
    p.add_argument("--ignore-live", action="store_true",
                   help="run even while a live deployment holds the GPU "
                        "(dashboards/web/gpu.py); the dashboard never passes it")
    p.add_argument("--describe-entry", action="store_true",
                   help="print the class's declarations as JSON and exit "
                        "(what the dashboard asks before offering steps)")
    p.add_argument("--check-prompt", default=None, metavar="TEXT",
                   help="construct the policy (never warmup, never infer), "
                        "judge TEXT with its check_prompt_length, print the "
                        "verdict as JSON and exit. Runs with the GPU hidden, "
                        "so it is not refused while a live deployment runs")


class _Progress:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None

    def __call__(self, **fields) -> None:
        if self.path is None:
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(fields))
        os.replace(tmp, self.path)


def resolves_own_checkpoints(cls) -> bool:
    """Does the class override `Policy.resolve_checkpoint`?

    Asked because the base class DEFINES the method (`run / step`), so
    `checkpoints.resolve`'s "use the hook when the class has one" would hand
    `latest` to it and get `run/latest` back. Only an override is the class's
    own layout; everything else is the numbered-subdirectory rule.
    """
    from elrobot_deploy.policy_contract.describe import _overrides_resolve_checkpoint
    return _overrides_resolve_checkpoint(cls)


def describe_entry(entry: str) -> dict:
    """The class's declarations, and whether it resolves its own steps —
    `policy_contract.describe`'s answer, in the replay's own JSON shape."""
    from elrobot_deploy.policy_contract import loading as entry_mod
    from elrobot_deploy.policy_contract import PolicyError
    from elrobot_deploy.policy_contract.describe import describe_class

    info = describe_class(entry_mod.load_entry(entry), entry)
    if info.invalid:
        raise PolicyError(info.invalid)
    return {
        "entry": entry, "class": info.class_path,
        "action_mode": info.action_mode, "anchor_camera": info.anchor_camera,
        "uses_cameras": list(info.uses_cameras or ()),
        "history": dict(info.history or {}),
        "max_prompt_tokens": info.max_prompt_tokens,
        "resolves_checkpoints": info.resolves_checkpoints,
        "python": info.python, "numpy": info.numpy_version,
    }


def resolve_checkpoint(cls, manifest, recorded_cfg, *, checkpoint=None,
                       run=None, step=None) -> str:
    """The step directory a replay (or a prompt check) loads, as a string;
    '' = none. `run` + `step` > `checkpoint` > the record's own."""
    from elrobot_deploy.policy_contract import checkpoints

    if run:
        return str(checkpoints.resolve(
            cls if resolves_own_checkpoints(cls) else None, run=run, step=step))
    if checkpoint is not None:
        return str(checkpoint)
    return str((manifest.get("checkpoint") or {}).get("path")
               or recorded_cfg.checkpoint or "")


#: What the prompt check hides from the policy it constructs: a tokenizer
#: needs no GPU, and a check that took one could starve a live deployment.
CPU_ONLY_ENV = {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}


def check_prompt(record_dir, text: str, *, entry: str | None = None,
                 checkpoint: str | None = None, run: str | None = None,
                 step: str | None = None) -> dict:
    """Would the policy accept `text` as its prompt? Constructs the class the
    way a replay does — the entry and checkpoint resolved the same way, the
    RECORDED robot — then asks `count_prompt_tokens` and
    `check_prompt_length`. Never `warmup()`, never `infer`: the answer is the
    tokenizer's, which a policy that has one loads in `__init__`
    (`docs/deploying_a_policy/policy_contract.md`, "The prompt").

    Returns `{ok, n_tokens, max_tokens, message, entry, class, checkpoint}`;
    a refusal's `message` is `PromptTooLong`'s sentence, verbatim.
    """
    from elrobot_deploy.policy_contract import loading as entry_mod
    from elrobot_deploy.policy_contract import validate_policy_class
    from elrobot_deploy.policy_contract.base import PromptTooLong
    from elrobot_deploy.dashboards.inspect import record_robot
    from elrobot_deploy.dashboards.inspect.modes.replay.record_config import deploy_config

    record_dir = Path(record_dir).expanduser().resolve()
    manifest = json.loads((record_dir / "manifest.json").read_text())
    recorded_cfg = deploy_config(manifest)
    rolls = manifest.get("rollouts") or [{}]
    recorded_prompt = (rolls[0] or {}).get("prompt", recorded_cfg.prompt)
    entry = entry or (manifest.get("policy") or {}).get("entry") \
        or recorded_cfg.policy_entry
    if not entry:
        raise ValueError("the record names no policy entry; pass --entry")
    cls = entry_mod.load_entry(entry)
    validate_policy_class(cls)
    ckpt = resolve_checkpoint(cls, manifest, recorded_cfg, checkpoint=checkpoint,
                              run=run, step=step)
    out = {"entry": entry, "class": f"{cls.__module__}.{cls.__qualname__}",
           "checkpoint": ckpt, "prompt": text}
    if not isinstance(text, str):
        raise ValueError(f"prompt must be a str, got {type(text).__name__}")
    try:
        policy = cls(Path(ckpt) if ckpt else None, recorded_prompt,
                     record_robot.hardware(manifest))
        limit = policy.max_prompt_tokens
        n = policy.count_prompt_tokens(text)
        policy.check_prompt_length(text)
    except PromptTooLong as exc:
        return {**out, "ok": False, "n_tokens": exc.n_tokens,
                "max_tokens": exc.max_tokens, "message": str(exc)}
    if limit is None:
        message = "OK (no limit declared)"
    elif n is None:
        message = f"OK (the policy counts no tokens; limit {limit} not applied)"
    else:
        message = f"OK ({n} tokens \u2264 {limit})"
    return {**out, "ok": True, "n_tokens": None if n is None else int(n),
            "max_tokens": None if limit is None else int(limit),
            "message": message}


def _chunk_row(dt: np.dtype, src, chunk, n_exec: int) -> np.ndarray:
    """One replayed chunk in the live record's own chunk dtype.

    Everything that says WHICH replan this is comes from the live row it stands
    in for, so the two line up row for row. The payload the mode does not use is
    NaN, the chunk ring's own convention.
    """
    from elrobot_deploy.policy_contract.actions import MODE_CODE

    row = np.zeros((), dtype=dt)
    for name in dt.names:
        if name in (src.dtype.names or ()) and name not in (
                "pose", "hand", "waist", "arm_q", "n_actions", "n_plan", "mode"):
            row[name] = src[name]
    horizon = dt["hand"].shape[0]
    n = min(chunk.n, horizon)
    for name in ("pose", "arm_q", "waist"):
        if name in dt.names:
            row[name] = np.nan
    row["hand"][:n] = chunk.hand[:n]
    if chunk.pose is not None:
        row["pose"][:n] = chunk.pose[:n]
    if chunk.arm_q is not None and "arm_q" in dt.names:
        row["arm_q"][:n] = chunk.arm_q[:n]
    if chunk.waist is not None and "waist" in dt.names:
        row["waist"][:n] = chunk.waist[:n]
    if "mode" in dt.names:
        row["mode"] = MODE_CODE[chunk.mode]
    row["n_actions"] = min(n_exec, n)
    if "n_plan" in dt.names:
        row["n_plan"] = n
    if "t_observed" in dt.names:
        row["t_observed"] = time.time()
    return row


def _join(chunk):
    """A sync-cut chunk and the tail `truncate` kept, as one: a replay stores
    every row the policy returned (`store`), `n_actions` says what would run."""
    from elrobot_deploy.policy_contract.actions import Chunk

    if chunk.plan is None:
        return chunk
    cat = (lambda a, b: None if a is None else np.concatenate([a, b]))
    return Chunk(hand=np.concatenate([chunk.hand, chunk.plan.hand]),
                 pose=cat(chunk.pose, chunk.plan.pose),
                 arm_q=cat(chunk.arm_q, chunk.plan.arm_q),
                 waist=cat(chunk.waist, chunk.plan.waist))


def _tcp_mm(rows, live, facts, state) -> dict:
    """Median / p99 TCP distance, per arm, from what the live chunk asked.

    Compared in the base frame at the recorded robot's TCP, row for row over
    the rows that would have EXECUTED (`n_actions`), matched by `chunk_id`.
    Joint rows are put through the recorded robot's FK first.
    """
    from elrobot_deploy.dashboards.inspect.modes.replay.overlay import flange_poses

    from elrobot_deploy.dashboards.inspect.scene import tcp_point

    tcp = tcp_point(facts)
    live_by_id = {int(r["chunk_id"]): r for r in live}
    errs = [[], []]
    fk = {"state": state}
    for r in rows:
        ref = live_by_id.get(int(r["chunk_id"]))
        if ref is None:
            continue
        n = min(int(r["n_actions"]), int(ref["n_actions"]))
        a = flange_poses(r, n, facts, fk)
        b = flange_poses(ref, n, facts, fk)
        if a is None or b is None:
            continue
        for h in (0, 1):
            pa = _tcp_points(a[:, h], tcp)
            pb = _tcp_points(b[:, h], tcp)
            errs[h].extend((np.linalg.norm(pa - pb, axis=-1) * 1e3).tolist())
    out = {"rows_compared": len(errs[0]), "tcp_mm": {}}
    for h, side in ((0, "left"), (1, "right")):
        e = np.asarray(errs[h])
        out["tcp_mm"][side] = ({"p50": round(float(np.percentile(e, 50)), 2),
                                "p99": round(float(np.percentile(e, 99)), 2)}
                               if e.size else {"p50": None, "p99": None})
    return out


def _tcp_points(poses, tcp) -> np.ndarray:
    """`(n, 7)` flange poses -> `(n, 3)` TCP points."""
    from elrobot_deploy.dashboards.inspect.scene import _pose7_to_mat

    out = np.zeros((len(poses), 3))
    for i, p in enumerate(poses):
        T = _pose7_to_mat(p)
        out[i] = T[:3, 3] + T[:3, :3] @ tcp
    return out


def replay(record_dir, *, entry: str | None = None, checkpoint: str | None = None,
           run: str | None = None, step: str | None = None,
           prompt: str | None = None, settings: dict | None = None,
           seed: int | None = None, out_dir: str | Path | None = None,
           progress=None, log=print) -> Path:
    """The whole replay. Returns the finished directory."""
    from elrobot_deploy.policy_contract import loading as entry_mod
    from elrobot_deploy.policy_contract import (DummyReader, EEF, Limits, bind_runtime,
                                            camera_info, check_against,
                                            deliver_settings,
                                            dummy_observation,
                                            observation_cameras, validate_policy_class)
    from elrobot_deploy.runtime.processes.p5_policy import PolicyLoop
    from elrobot_deploy.config_loading.rollout_settings import (RolloutSettings,
                                                            check as check_settings)
    from elrobot_deploy.dashboards.inspect import loader, record_robot
    from elrobot_deploy.dashboards.inspect.modes.replay import anchors as anchors_mod
    from elrobot_deploy.dashboards.inspect.modes.replay import observe, store
    from elrobot_deploy.dashboards.inspect.modes.replay.record_config import (deploy_config,
                                                                          widen_history)

    progress = progress or (lambda **_: None)
    record_dir = Path(record_dir).expanduser().resolve()
    _need_replay_extra()
    progress(phase="reading the record")
    rollout = loader.load(record_dir)
    try:
        manifest = rollout.manifest
        recorded_cfg = deploy_config(manifest)
        rolls = manifest.get("rollouts") or [{}]
        recorded_prompt = (rolls[0] or {}).get("prompt", recorded_cfg.prompt)

        recorded_entry = (manifest.get("policy") or {}).get("entry") \
            or recorded_cfg.policy_entry
        entry = entry or recorded_entry
        if not entry:
            raise ValueError("the record names no policy entry; pass --entry")
        cls = entry_mod.load_entry(entry)
        validate_policy_class(cls)

        ckpt = resolve_checkpoint(cls, manifest, recorded_cfg,
                                  checkpoint=checkpoint, run=run, step=step)

        hw = record_robot.hardware(manifest)
        facts = record_robot.from_manifest(manifest)
        base = RolloutSettings.of(recorded_cfg)
        wanted = base.updated(dict(settings or {}))
        cfg = widen_history(recorded_cfg, cls.history)
        cfg = check_settings(wanted, cfg, hw)
        cfg = dataclasses.replace(cfg, checkpoint=ckpt, policy_entry=entry)
        check_against(cls, cfg)
        prompt = recorded_prompt if prompt is None else str(prompt)

        # The observations, before the policy loads: a record that cannot be
        # re-run should say so in a second, not after a 40 s load.
        views_used = observation_cameras(cls, cfg)
        views = anchors_mod.views_of(manifest, set(views_used)
                                     | {cls.anchor_camera}
                                     | {c for c in (cls.history or {}) if c != "state"})
        sources = {views[v].source for v in views_used}
        specs, report = anchors_mod.reconstruct(
            rollout, sources, views[cls.anchor_camera].source)
        log(f"[replay] {report.describe()}")

        seed = int(seed if seed is not None else random.SystemRandom().randrange(2**31))
        random.seed(seed)
        np.random.seed(seed % (2**32))

        out = Path(out_dir) if out_dir else store.replays_dir(record_dir) / store.new_id()
        writer = store.Writer(out, chunk_dtype=rollout.chunks.dtype,
                              cameras=sorted(sources))
        writer.begin()

        try:
            progress(phase="loading the policy")
            t0 = time.monotonic()
            policy = cls(Path(ckpt) if ckpt else None, recorded_prompt, hw)
            frames = observe.RecordFrames(rollout, views)
            dummy = dummy_observation({c: cfg.views[c] for c in views_used}, recorded_prompt)
            bind_runtime(policy, dummy, DummyReader(cfg.views, dict(cls.history or {}),
                                                    state_hz=hw.lowstate_hz),
                         camera_info(cfg), base)
            progress(phase="warming up")
            policy.warmup()
            # P5's START (`RolloutLatch._take`): prompt, then settings, then one
            # dummy inference so a recompile is paid here and not by chunk 1.
            if prompt != getattr(policy, "prompt", None):
                policy.set_prompt(prompt)
            if wanted != base:
                policy.on_settings(wanted)
            deliver_settings(policy, wanted)
            policy.infer(dataclasses.replace(dummy, prompt=prompt), policy.reader())
            t_ready = time.monotonic() - t0
            log(f"[replay] {cls.__name__} ready in {t_ready:.1f} s")

            reader = observe.RecordReader(rollout, frames, dict(cls.history or {}))
            loop = SimpleNamespace(
                mode=cls.action_mode, cfg=cfg, limits=Limits.from_robot(hw),
                policy=policy, arm_idx=np.asarray(hw.joints_arm, np.intp),
                waist_idx=np.asarray(hw.joints_waist, np.intp))
            loop.measured = lambda a: PolicyLoop.measured(loop, a)
            live = anchors_mod.live_chunks(rollout.chunks)
            live_by_id = {int(c["chunk_id"]): c for c in live}
            took = []
            for i, spec in enumerate(specs):
                progress(phase="replaying", chunk=i + 1, n_chunks=len(specs))
                obs = observe.observation(rollout, frames, spec, prompt, views_used)
                reader.now = spec.t_obs_received
                t1 = time.monotonic()
                actions = policy.infer(obs, reader)
                took.append(time.monotonic() - t1)
                anchor = SimpleNamespace(t_anchor=spec.t_anchor, t_trigger=spec.t_trigger,
                                         state=rollout.state[spec.state_row])
                chunk = PolicyLoop.to_chunk(loop, actions, anchor)
                n_exec = chunk.n
                writer.add(_chunk_row(rollout.chunks.dtype, live_by_id[spec.chunk_id],
                                      _join(chunk), n_exec), spec, took[-1])

            progress(phase="writing")
            rows = writer.rows()
            ms = np.asarray(took) * 1e3
            manifest_out = {
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": {"record": str(record_dir), "name": record_dir.name},
                "entry": entry, "class": f"{cls.__module__}.{cls.__qualname__}",
                "recorded_entry": recorded_entry, "action_mode": cls.action_mode,
                "checkpoint": ckpt, "run": run, "step": step,
                "prompt": prompt, "recorded_prompt": recorded_prompt,
                "settings": wanted.as_dict(), "recorded_settings": base.as_dict(),
                "seed": seed, "n_chunks": writer.n,
                "anchors": report.as_json(),
                "timing": {"ready_s": round(t_ready, 2),
                           "infer_ms_p50": round(float(np.median(ms)), 2) if ms.size else None,
                           "infer_ms_max": round(float(ms.max()), 2) if ms.size else None},
                "summary": _tcp_mm(rows, live, facts, rollout.state),
                "deadline_shift": bool(policy.deadline_shift),
                "eef": cls.action_mode == EEF,
            }
            done = writer.close(manifest_out)
        except BaseException as exc:
            writer.fail(exc)
            raise
        progress(phase="done", chunk=len(specs), n_chunks=len(specs))
        log(f"[replay] {writer.n} chunks -> {done}")
        return done
    finally:
        rollout.close()


def _need_replay_extra() -> None:
    """A full replay decodes JPEGs (cv2) and runs FK (mujoco) in THIS
    interpreter — the policy's. Say so up front rather than as an
    ImportError from deep inside the first frame."""
    import importlib.util
    missing = [m for m in ("cv2", "mujoco") if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            f"a policy replay needs {missing} in the policy's interpreter "
            f"({sys.executable}); install elrobot_deploy's `replay` extra into that "
            f"venv (opencv-python-headless, mujoco). --describe-entry and "
            f"--check-prompt need only the minimal core.")


def main(argv=None) -> int:
    """`python -m elrobot_deploy.dashboards.inspect.modes.replay.run ...` — what
    the dashboard spawns, under the policy's interpreter."""
    p = argparse.ArgumentParser(
        prog="python -m elrobot_deploy.dashboards.inspect.modes.replay.run",
        description="re-run a rollout's observations through a policy")
    add_arguments(p)
    return main_args(p.parse_args(argv))


def main_args(args) -> int:
    if args.describe_entry:
        from elrobot_deploy.policy_contract import loading as entry_mod
        from elrobot_deploy.dashboards.inspect import loader  # noqa: F401  (a bad record should fail here too)

        spec = args.entry
        if not spec:
            man = json.loads((Path(args.record) / "manifest.json").read_text())
            spec = (man.get("policy") or {}).get("entry") \
                or (man.get("deploy_config") or {}).get("policy_entry")
        spec = entry_mod.normalize(spec, base_dir=Path.cwd()) if spec else spec
        print(json.dumps(describe_entry(spec)))
        return 0
    if args.check_prompt is not None:
        # No GPU refusal: the GPU is hidden from this process instead, before
        # the policy's module (and any jax/torch) is imported.
        os.environ.update(CPU_ONLY_ENV)
        try:
            verdict = check_prompt(args.record, args.check_prompt, entry=args.entry,
                                   checkpoint=args.checkpoint, run=args.run,
                                   step=args.step)
        except Exception as exc:                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
            return 1
        print(json.dumps(verdict))
        return 0
    from elrobot_deploy.dashboards.web import gpu

    why = "" if args.ignore_live else gpu.refusal()
    if why:
        print(f"REFUSED: {why}", file=sys.stderr, flush=True)
        return 1
    settings = json.loads(args.settings) if args.settings else {}
    if args.sample_steps is not None:
        settings["sample_steps"] = int(args.sample_steps)
    try:
        replay(args.record, entry=args.entry, checkpoint=args.checkpoint,
               run=args.run, step=args.step, prompt=args.prompt,
               settings=settings, seed=args.seed, out_dir=args.out,
               progress=_Progress(args.progress_file))
    except Exception as exc:                                  # noqa: BLE001
        # The last line is what the dashboard shows. A refused prompt reads
        # exactly as the live dashboard's refusal does.
        import traceback
        traceback.print_exc()
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        _Progress(args.progress_file)(phase="failed", error=str(exc))
        return 1
    return 0


__all__ = ["CPU_ONLY_ENV", "add_arguments", "check_prompt", "describe_entry",
           "main", "main_args", "replay", "resolve_checkpoint"]


if __name__ == "__main__":
    raise SystemExit(main())
