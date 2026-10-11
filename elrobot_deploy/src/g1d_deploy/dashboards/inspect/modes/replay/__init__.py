"""Mode `replay`: what a policy would have commanded on this rollout's own
observations. For the user: `docs/deploying_a_policy/rollout_inspection.md`,
"policy checkpoint (`replay`)".

A REPLAY re-runs one recorded rollout's observations — the frames and state the
policy saw at each replan, rebuilt from what P9 kept (`anchors`, `observe`) —
through a policy class (and its checkpoint, if it has one) via the ordinary `Policy` interface
(`infer(obs, reader)`), and lands beside the record:

    <record>/policy_replays/<id>/     (`store`)

Each replay is its own directory and its own switch, including repeats of one
checkpoint at one setting: the spread between those is the sampler's noise,
the floor every other difference is read against.

What can vary, every default being the rollout's own (its manifest):

* the policy entry and the checkpoint — a run directory picked in the browser
  (`browse`) and a step in it, or a pasted step directory;
* the prompt — refused with the live dashboard's own sentence when the policy's
  `check_prompt_length` says it does not fit (checked by the replay itself,
  before any inference, since only a constructed policy owns a tokenizer).
  "Check prompt" asks the same question without replaying: a subprocess
  constructs the class, never warms it up, and answers (`run.check_prompt`);
* `sample_steps` — handed to the policy through `on_settings`, as P5 does at
  START, so a policy that caches a compiled sampler per step count does here
  what it does on the robot;
* how many repeats to run.

Everything else in the rollout settings is shown read-only: the replay re-uses
the RECORDED replan instants, so `P`, sync/async and the control rate are the
rollout's by construction, and the IK and the servo act after the policy and do
not change what it returns.

Two halves that never import each other, meeting at that directory:

  * **this mode** — numpy and mujoco. Lists replays, poses one onto the display
    frames (`overlay`), browses for checkpoints, spawns the producer.
  * **the producer** — `run`, spawned by module path
    (`python -m g1d_deploy.dashboards.inspect.modes.replay.run`) under the
    POLICY's interpreter (`policy_python`): it imports the policy, which may
    take the GPU, and a dashboard that did that in-process would
    OOM the next replay it spawned. It is refused while a live deployment holds
    the GPU (`dashboards/web/gpu.py`). The class description and the prompt check are
    spawned the same way.

**Which interpreter.** The form's `python` when it names one; else the one
the rollout ran its policy under (the manifest's `deploy_config.policy_python`)
when it exists on this machine; else the dashboard's own
(`Options.python`). A record inspected on another machine therefore falls back
to the dashboard's interpreter, which must then hold the policy's stack.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from g1d_deploy.dashboards.web import gpu
from g1d_deploy.dashboards.web.jobs import RUNNER
from g1d_deploy.dashboards.inspect.modes import describe
from g1d_deploy.dashboards.inspect.modes.replay import browse, store

#: The producer, by module path: spawning `-m g1d_deploy` would import the
#: CLI's dependencies in the policy's interpreter, which holds only the core.
PRODUCER = "g1d_deploy.dashboards.inspect.modes.replay.run"

#: The rollout-settings fields a replay may change: those that change what the
#: POLICY returns. The rest are execution-side or fixed by the recorded replans.
EDITABLE = ("sample_steps",)

#: How long `--describe-entry` may take: it imports the policy's module.
DESCRIBE_TIMEOUT_S = 120

#: How long "Check prompt" may take: it imports the policy's module and
#: constructs the class (its `__init__`, the cheap load —
#: `docs/deploying_a_policy/policy_contract.md`, "Lifecycle").
CHECK_PROMPT_TIMEOUT_S = 180

#: Repeats one Replay press may queue.
MAX_REPEATS = 20


class ReplayMode:
    name = "replay"
    title = "policy checkpoint"
    blurb = ("Re-run this rollout's own observations through a policy — this "
             "checkpoint, another step or run, another prompt or sample-step "
             "count — and draw each result as a ghost against what flew.")

    def __init__(self, options=None):
        self.options = options
        self._lock = threading.Lock()
        self._builder = None
        self._cache: dict[str, tuple[dict, bytes]] = {}
        self._described: dict[str, dict] = {}
        self._prompt_checks: dict[tuple, dict] = {}
        self._state = None

    # --- lifecycle ---------------------------------------------------------

    def _on_load(self, loaded, record_dir: Path) -> None:
        # Imported here, not at the top: the producer (`run`) is spawned as
        # a module of this package in the policy's interpreter, and importing
        # the package must not pull mujoco in there.
        from g1d_deploy.dashboards.inspect.modes.replay.overlay import OverlayBuilder

        with self._lock:
            # Rebuilt per rollout: an overlay is per DISPLAY FRAME.
            self._builder = OverlayBuilder(loaded.rollout, loaded.timeline,
                                           loaded.scene)
            self._cache.clear()

    def _close(self) -> None:
        with self._lock:
            self._builder = None
            self._cache.clear()

    def _current(self):
        cur = self._state.current if self._state else None
        if cur is None:
            raise RuntimeError("load a rollout first")
        return cur

    # --- what the rollout ran with ----------------------------------------

    def recorded(self) -> dict:
        """The defaults: the rollout's own policy, checkpoint, prompt, settings."""
        from g1d_deploy.dashboards.inspect import discover

        cur = self._current()
        d = discover.specifics(cur.rollout.manifest)
        active = d.get("active_config") or {}
        if not active:
            from g1d_deploy.dashboards.inspect.modes.replay.record_config import (deploy_config)
            from g1d_deploy.config_loading.rollout_settings import RolloutSettings
            active = RolloutSettings.of(deploy_config(cur.rollout.manifest)).as_dict()
        deploy = str((cur.rollout.manifest.get("deploy") or {}).get("path") or "")
        return {"entry": d.get("policy_entry") or "",
                # The deploy file the rollout ran from, if it is still there.
                "deploy": deploy if deploy and os.path.isfile(deploy) else "",
                "class": d.get("policy_class") or "",
                "python": self.policy_python({}),
                "checkpoint": d.get("checkpoint") or "",
                "prompt": d.get("prompt") or "", "settings": active,
                "action_mode": cur.rollout.action_mode}

    def defaults(self) -> dict:
        rec = self.recorded()
        cur = self._current()
        return {
            **rec, "editable": list(EDITABLE),
            "starts": browse.starts(self.options, rec["checkpoint"]),
            "record": str(cur.rollout.path),
            "gpu": gpu.refusal(), "busy": RUNNER.busy(),
            "max_repeats": MAX_REPEATS,
        }

    def check_settings(self, changes: dict) -> dict:
        """A settings request, checked the way the live dashboard checks one
        (`rollout_settings.check`), against the rollout it would replay."""
        from g1d_deploy.dashboards.inspect import record_robot
        from g1d_deploy.dashboards.inspect.modes.replay.record_config import (deploy_config)
        from g1d_deploy.config_loading.rollout_settings import RolloutSettings, check

        cur = self._current()
        bad = sorted(set(changes) - set(EDITABLE))
        if bad:
            raise ValueError(f"a replay cannot change {bad}: it re-uses the "
                             f"recorded replans; only {list(EDITABLE)} change "
                             f"what the policy returns")
        cfg = deploy_config(cur.rollout.manifest)
        wanted = RolloutSettings.of(cfg).updated(changes)
        check(wanted, cfg, record_robot.hardware(cur.rollout.manifest))
        return wanted.as_dict()

    def policy_python(self, req: dict) -> str:
        """The interpreter the policy's subprocesses run under (module
        docstring): the form's, the recorded one if it exists here, else the
        dashboard's own."""
        asked = (req or {}).get("python")
        if asked:
            return os.path.abspath(os.path.expanduser(str(asked)))
        cur = self._state.current if self._state else None
        if cur is not None:
            rec = str((cur.rollout.manifest.get("deploy_config") or {})
                      .get("policy_python") or "")
            if rec and os.access(rec, os.X_OK):
                return rec
        return getattr(self.options, "python", None) or _python()

    def describe_entry(self, entry: str, python: str | None = None) -> dict:
        """The class's declarations, asked of a subprocess (importing a policy
        module may import jax; the dashboard must not)."""
        entry = self._normalize(entry)
        py = self.policy_python({"python": python})
        with self._lock:
            hit = self._described.get((entry, py))
        if hit is not None:
            return hit
        cur = self._current()
        cmd = [py, "-m", PRODUCER, "--record", str(cur.rollout.path),
               "--describe-entry", "--entry", entry]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=DESCRIBE_TIMEOUT_S, cwd=os.getcwd())
        except subprocess.TimeoutExpired:
            raise ValueError(f"{entry}: the class did not load in "
                             f"{DESCRIBE_TIMEOUT_S}s") from None
        if proc.returncode != 0:
            lines = (proc.stderr or proc.stdout).strip().splitlines()
            raise ValueError(f"the policy adapter could not be loaded: "
                             f"{lines[-1] if lines else 'no output'}  ({entry})")
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        with self._lock:
            self._described[(entry, py)] = out
        return out

    def _checkpoint_args(self, req: dict, rec: dict) -> tuple[list[str], tuple]:
        """The producer's checkpoint flags for a form request, and a cache key
        for them: `run` + `step`, else `checkpoint` (None = the record's)."""
        run, step = req.get("run"), req.get("step")
        if run:
            step = str(step or "latest")
            return ["--run", str(run), "--step", step], ("run", str(run), step)
        checkpoint = req.get("checkpoint")
        ckpt = rec["checkpoint"] if checkpoint is None else str(checkpoint)
        return ["--checkpoint", ckpt], ("checkpoint", ckpt)

    def check_prompt(self, req: dict) -> dict:
        """Would the policy in the form accept this prompt? Asked of a
        subprocess (`replay-policy --check-prompt`) that constructs the class
        with its checkpoint — the tokenizer is the policy's — but never warms
        it up or infers. Not refused while a live deployment runs: the
        subprocess hides the GPU from itself. Cached per (record, entry,
        checkpoint, prompt), so a second press is instant."""
        cur = self._current()
        rec = self.recorded()
        entry = self._normalize(req.get("entry") or rec["entry"])
        if not entry:
            raise ValueError("the record names no policy entry; choose one")
        prompt = req.get("prompt")
        prompt = rec["prompt"] if prompt is None else str(prompt)
        ck_args, ck_key = self._checkpoint_args(req, rec)
        py = self.policy_python(req)
        key = (str(cur.rollout.path), entry, ck_key, prompt, py)
        with self._lock:
            hit = self._prompt_checks.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        from g1d_deploy.dashboards.inspect.modes.replay.run import CPU_ONLY_ENV

        # `--check-prompt=TEXT`, one token: a prompt starting with "-" must
        # not read as a flag.
        cmd = [py, "-m", PRODUCER, "--record", str(cur.rollout.path),
               "--entry", entry, *ck_args, f"--check-prompt={prompt}"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=CHECK_PROMPT_TIMEOUT_S, cwd=os.getcwd(),
                                  env={**os.environ, **CPU_ONLY_ENV})
        except subprocess.TimeoutExpired:
            raise ValueError(
                f"the prompt check timed out after {CHECK_PROMPT_TIMEOUT_S} s: "
                f"constructing the policy took too long "
                f"(docs/deploying_a_policy/policy_contract.md: "
                f"__init__ should load only what is cheap: config, tokenizer and norm "
                f"stats)") from None
        if proc.returncode != 0:
            lines = (proc.stderr or proc.stdout).strip().splitlines()
            refused = [ln for ln in lines if ln.startswith("REFUSED: ")]
            raise ValueError(refused[-1][len("REFUSED: "):] if refused
                             else f"{entry}: " + " | ".join(lines[-3:]))
        verdict = json.loads(proc.stdout.strip().splitlines()[-1])
        with self._lock:
            self._prompt_checks[key] = verdict
        return {**verdict, "cached": False}

    @staticmethod
    def _normalize(entry: str) -> str:
        from g1d_deploy.policy_contract import loading as entry_mod
        return entry_mod.normalize(str(entry).strip(), base_dir=Path.cwd())

    # --- the list ------------------------------------------------------------

    def replays(self) -> dict:
        cur = self._current()
        rec = self.recorded()
        from g1d_deploy.dashboards.inspect.modes.replay.anchors import live_chunks

        live = store.ReplayInfo(
            store.LIVE_ID, "live",
            f"live · {store.checkpoint_label(rec['checkpoint'])}",
            entry=rec["entry"], policy_class=rec["class"],
            checkpoint=rec["checkpoint"], prompt=rec["prompt"],
            settings=rec["settings"], n_chunks=len(live_chunks(cur.rollout.chunks)),
            action_mode=rec["action_mode"], note="what the robot actually commanded")
        rows = [live] + store.list_replays(cur.rollout.path, rec)
        return {"name": cur.rollout.path.name,
                "replays": [r.as_json() for r in rows]}

    def overlay(self, rid: str) -> tuple[dict, bytes]:
        """Build (or return) one replay's overlay; cached per loaded rollout."""
        from g1d_deploy.dashboards.web import pack
        from g1d_deploy.dashboards.inspect.modes.replay.anchors import live_chunks

        cur = self._current()
        with self._lock:
            if self._builder is None:
                raise RuntimeError("nothing loaded")
            hit = self._cache.get(rid)
            if hit is not None:
                return hit
            live = live_chunks(cur.rollout.chunks)
            if rid == store.LIVE_ID:
                chunks, reference = live, None
            else:
                info = {r.id: r for r in store.list_replays(cur.rollout.path)}.get(rid)
                if info is None or info.state != "ready":
                    raise KeyError(rid)
                chunks, reference = store.load_chunks(Path(info.dir)), live
            built = self._builder.build(chunks, reference)
            meta = dict(built["meta"], id=rid, layout=pack.spans(built["parts"]))
            payload = (meta, pack.pack(built["parts"]))
            self._cache[rid] = payload
            return payload

    def strip(self, rid: str) -> dict:
        """The timeline strip for one replay (`overlay.action_diff`)."""
        from g1d_deploy.dashboards.inspect.modes.replay.anchors import live_chunks
        from g1d_deploy.dashboards.inspect.modes.replay.overlay import action_diff

        cur = self._current()
        if rid == store.LIVE_ID:
            return {"t": [], "diff_mm": [], "chunk_id": [], "unit": "mm",
                    "note": "the recorded actions themselves"}
        info = {r.id: r for r in store.list_replays(cur.rollout.path)}.get(rid)
        if info is None or info.state != "ready":
            raise KeyError(rid)
        with self._lock:
            if self._builder is None:
                raise RuntimeError("nothing loaded")
            poses = self._builder.poses
        return action_diff(store.load_chunks(Path(info.dir)),
                           live_chunks(cur.rollout.chunks), poses,
                           float(cur.timeline.t0))

    # --- starting ------------------------------------------------------------

    def start(self, req: dict) -> dict:
        cur = self._current()
        why = gpu.refusal()
        if why:
            raise RuntimeError(why)
        rec = self.recorded()
        entry = self._normalize(req.get("entry") or rec["entry"])
        if not entry:
            raise ValueError("the record names no policy entry; choose one")
        settings = self.check_settings(dict(req.get("settings") or {}))
        changes = {k: settings[k] for k in EDITABLE
                   if settings[k] != rec["settings"].get(k)}
        repeats = int(req.get("repeats", 1))
        if not 1 <= repeats <= MAX_REPEATS:
            raise ValueError(f"repeats = {repeats}; want 1..{MAX_REPEATS}")
        ck_args, ck_key = self._checkpoint_args(req, rec)
        args = ["--record", str(cur.rollout.path), "--entry", entry, *ck_args]
        if ck_key[0] == "run":
            browse.remember(self._state_file(), ck_key[1])
            where = f"{Path(ck_key[1]).name} step {ck_key[2]}"
        else:
            ckpt = ck_key[1]
            if ckpt:
                browse.remember(self._state_file(), str(Path(ckpt).parent))
            where = store.checkpoint_label(ckpt)
        prompt = req.get("prompt")
        if prompt is not None and prompt != rec["prompt"]:
            args += ["--prompt", str(prompt)]
        if changes:
            args += ["--settings", json.dumps(changes)]
        jobs = []
        for i in range(repeats):
            job = RUNNER.submit(
                kind=self.name,
                label=f"{where}" + (f" · {changes}" if changes else "")
                      + (f" · {i + 1}/{repeats}" if repeats > 1 else ""),
                cmd=[self.policy_python(req), "-m", PRODUCER, *args],
                cwd=os.getcwd(), jobs_dir=store.replays_dir(cur.rollout.path) / "_jobs",
                meta={"record": cur.rollout.path.name, "entry": entry})
            jobs.append(job.as_json())
        return {"jobs": jobs}

    def _state_file(self) -> Path:
        from g1d_deploy.dashboards.inspect.options import DEFAULT_STATE
        return Path(getattr(self.options, "state_file", None)
                    or DEFAULT_STATE.expanduser())

    # --- mounting ----------------------------------------------------------

    def mount(self, app, state) -> None:
        self._state = state
        state.on_load(self._on_load)
        app.on_close(self._close)

        def guarded(fn):
            def route(req):
                try:
                    fn(req)
                except RuntimeError as exc:
                    req.error(str(exc), 409)
            return route

        app.get("/api/replay/defaults")(guarded(lambda req: req.json(self.defaults())))
        app.get("/api/replay/replays")(guarded(lambda req: req.json(self.replays())))
        app.get("/api/replay/describe")(guarded(
            lambda req: req.json(self.describe_entry(req.arg("entry", ""),
                                                     req.arg("python", "") or None))))
        app.get("/api/replay/browse")(guarded(
            lambda req: req.json(browse.listing(req.arg("path", "")))))
        app.get("/api/replay/pick")(guarded(
            lambda req: req.json(browse.native_pick(req.arg("what", "ckpt"),
                                                    req.arg("start", "")))))
        app.get("/api/replay/deploy")(guarded(
            lambda req: req.json(browse.deploy_of(req.arg("path", "")))))

        @app.get("/api/replay/overlay")
        def overlay_meta(req):
            meta, _ = self.overlay(req.arg("id", ""))
            req.json(meta)

        @app.get("/api/replay/strip")
        def strip(req):
            try:
                req.json(self.strip(req.arg("id", "")))
            except KeyError as exc:
                req.error(f"no ready replay {exc}", 404)
            except RuntimeError as exc:
                req.error(str(exc), 409)

        @app.get("/api/replay/overlay.bin")
        def overlay_bin(req):
            _, blob = self.overlay(req.arg("id", ""))
            req.bytes(blob, "application/octet-stream")

        app.post("/api/replay/settings")(guarded(
            lambda req: req.json({"settings": self.check_settings(req.body())})))
        # No GPU guard: the check hides the GPU from its subprocess.
        app.post("/api/replay/check-prompt")(guarded(
            lambda req: req.json(self.check_prompt(req.body()))))
        app.post("/api/replay/start")(guarded(
            lambda req: req.json(self.start(req.body()))))

        @app.get("/api/replay/jobs")
        def jobs(req):
            req.json({"jobs": [j.as_json() for j in RUNNER.jobs(kind=self.name)],
                      "busy": RUNNER.busy(), "gpu": gpu.refusal()})

        @app.post("/api/replay/cancel")
        def cancel(req):
            req.json({"cancelled": RUNNER.cancel(req.body()["job"])})

    def describe(self) -> dict:
        return describe(self)


def _python() -> str:
    import sys
    return sys.executable


__all__ = ["CHECK_PROMPT_TIMEOUT_S", "EDITABLE", "PRODUCER", "ReplayMode"]
