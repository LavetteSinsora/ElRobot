"""The deploy file (`deploy.toml`) -> `DeployConfig` + `Hardware`.

Run by every command before any process exists (`cli/main.py`), and again by
the inspection replay on a record's copy of the file. Design, key by key:
`docs/deploying_a_policy/deploy_toml.md`.

The operator's file: which robot, which policy, an optional checkpoint, the
prompt, the home/exit posture, the default rollout settings, and where records
go. A complete, commented example is `examples/deploy.toml`.

**Layered, later wins:**

    infra defaults  <  the policy's trained_*  <  deploy.toml  <  command line

Every command-line override is applied to the deploy file's TREE — `--set
control.replan_period=10` and `--replan-period 10` are the same edit — and the
tree is resolved once, so there is one path from text to config and the result
can say, per field, which layer it came from (`Resolved.sources`, printed at
startup and written to every manifest).

Two rules are enforced at resolution rather than left to be discovered:

* `replan_period <= action_horizon` (in `validate.py`, which refuses it
  before anything spawns), and `action_horizon` no larger than the policy's
  declared `trained_action_horizon` — a chunk cannot contain rows the policy
  never produces.
* a control rate different from the declared one is a WARNING, not an error: it
  is a deliberate speed A/B, and the operator should see that it is one.

The policy CLASS is read here (its declarations — among them the optional
`trained_control_hz` / `trained_action_horizon` class attributes — and its
optional `resolve_checkpoint`), never instantiated: construction is P5's
(`policy_contract.make_policy`), which receives the resolved values through
`DeployConfig` — `policy_entry`, `checkpoint`, `prompt`, `policy_python`. It
is read IN THE POLICY'S INTERPRETER (`[policy] python`) when one is named:
`policy_contract.describe` imports it there and hands back a `PolicyInfo`, because the
policy's dependencies need not be importable here.
"""

from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from g1d_deploy.policy_contract import checkpoints, loading as entry
from g1d_deploy.runtime.shared_memory import schema
from g1d_deploy.config_loading.resolved import (DeployConfig, Hardware, ServoShaping,
                                                camera_views, load_hardware,
                                                not_a_view_hint,
                                                tomllib)
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading.machine_file import Machine, load_machine
from g1d_deploy.config_loading.robot_file import DEFAULT_ROBOT, Robot, load_robot

#: The four timing values every deploy needs and `DeployConfig` has no default
#: for. The lowest layer: a policy's `trained_*` attributes or the deploy file
#: replaces them. 25 of 50 at 30 Hz is a common action-chunking setting.
INFRA_DEFAULTS = {"sync": True, "replan_period_ticks": 25,
                  "action_horizon": 50, "control_hz": 30.0}

#: Where records go when `[record] out` is not given: beside the deploy file,
#: under the name inspection looks for.
DEFAULT_RECORD_DIR = "rollout_record"

# The file's vocabulary, table by table: TOML key -> DeployConfig field. A key
# not listed is refused by name — a misspelt `replan_peroid` would otherwise
# silently run at the default.
_TOP = {"robot", "prompt", "policy", "checkpoint", "home", "exit", "control",
        "intervene", "record", "labels"}
_POLICY = {"entry": None, "anchor_camera": "primary_camera",
           "uses_cameras": "observed_cameras", "env": "policy_env",
           "python": "policy_python"}
_CHECKPOINT = {"path", "run", "step"}
_POSE = {"arm", "waist", "hand"}
#: The infra's default home, used when a deploy file gives no `[home] arm`:
#: radians, left arm then right, each shoulder pitch/roll/yaw, elbow, wrist
#: roll/pitch/yaw (the robot file's `[joints] arm` order). Forearms forward
#: and a little down in front of the torso, grippers apart, elbows clear of
#: the body: a posture suited to tabletop work and reachable from the damped
#: pose without sweeping through the table.
DEFAULT_HOME_ARM = (1.00535, 0.05224, -0.18189, -0.38960, -0.00396, -0.58212, -0.09228,
                    1.00533, -0.05232, 0.18193, -0.38960, 0.00397, -0.58212, 0.09220)

_HOME_EXTRA = {"tol_arm": "home_tol_arm", "tol_waist": "home_tol_waist",
               "min_s": "home_min_s", "velocity_margin": "home_velocity_margin"}
_CONTROL = {"sync": "sync", "replan_period": "replan_period_ticks",
            "action_horizon": "action_horizon", "control_hz": "control_hz",
            "sample_steps": "sample_steps", "blend": "blend",
            "temporal_ensemble_m": "temporal_ensemble_m",
            "state_history_s": "state_history_s", "ik": "ik",
            "ik_posture_cost": "ik_posture_cost",
            "ik_collision_min_dist": "ik_collision_min_dist",
            "ik_early_stop": "ik_early_stop", "ik_max_iters": "ik_max_iters",
            "solve_lead_ticks": "solve_lead_ticks",
            "plan_lookahead_ticks": "plan_lookahead_ticks", "servo": None}
_SERVO = {f.name for f in fields(ServoShaping)}
#: `[control.servo]` keys that name a module rather than set a number: per
#: launch, so they live on `DeployConfig`, not in the per-rollout `ServoShaping`.
_SERVO_MODULES = {"smoother": "servo_smoother", "feedforward": "servo_feedforward"}
_INTERVENE = {"lead_s", "stale_budget_s", "scale", "hand_tcp",
              "gripper_rate", "squeeze_margin_rad", "up_closes"}
_RECORD = {"out", "enabled", "task"}

#: Command-line flags that are shorthands for a deploy-file key. Applied to the
#: tree before any `--set`, so a `--set` of the same key wins.
FLAG_KEYS = {
    "prompt": "prompt", "robot": "robot",
    "step": "checkpoint.step", "out": "record.out",
    "home_tol_arm": "home.tol_arm", "home_tol_waist": "home.tol_waist",
    "sample_steps": "control.sample_steps",
    "replan_period": "control.replan_period",
    "action_horizon": "control.action_horizon",
    "control_hz": "control.control_hz",
    "ik": "control.ik",
    "servo_filter_hz": "control.servo.filter_hz",
    "velocity_ff": "control.servo.velocity_ff",
    "torque_ff": "control.servo.torque_ff",
    "arm_gain_scale": "control.servo.arm_gain_scale",
}


@dataclass(frozen=True)
class Resolved:
    """Everything a launch needs, and where each piece came from."""

    cfg: DeployConfig
    hw: Hardware
    robot: Robot
    machine: Machine
    #: The policy class as its own interpreter described it
    #: (`policy_contract.describe.PolicyInfo`), or None when no entry is named.
    policy: object | None
    deploy_path: str | None
    deploy_text: str | None
    #: The deploy tree AFTER command-line overrides — what actually ran.
    tree: dict
    #: The per-rollout record base, or None for `--no-record` / enabled = false.
    record_dir: Path | None
    #: DeployConfig field -> "infra" | "trained" | "deploy" | "cli".
    sources: dict[str, str] = field(default_factory=dict)
    trained: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# --- the tree ---------------------------------------------------------------


def parse_value(text: str):
    """A `--set` value, read as a TOML value when it is one (`10`, `true`,
    `[1, 2]`, `"x"`) and as a bare string otherwise (`waist`, `/runs/x`)."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def set_key(tree: dict, dotted: str, value) -> None:
    """`tree[a][b][c] = value` for `"a.b.c"`, creating tables on the way."""
    keys = dotted.split(".")
    if not all(keys):
        raise ConfigError(f"bad key {dotted!r}")
    node = tree
    for k in keys[:-1]:
        nxt = node.setdefault(k, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"--set {dotted}: {k!r} is a value, not a table")
        node = nxt
    node[keys[-1]] = value


def apply_overrides(tree: dict, flags: dict | None = None,
                    sets: list[str] | tuple[str, ...] = (),
                    checkpoint: str | None = None) -> dict:
    """The deploy tree with command-line overrides applied, as a copy.

    `flags` maps `FLAG_KEYS` names to values (None = not given); `sets` are
    `key=value` strings; `checkpoint` is `--checkpoint PATH`, which REPLACES
    `[checkpoint]` rather than merging into it — a path and a run are two
    different ways to name it, and leaving the file's `run` beside a
    command-line `path` would be refused as ambiguous.
    """
    out = copy.deepcopy(tree)
    if checkpoint is not None:
        out["checkpoint"] = {"path": checkpoint}
    for name, value in (flags or {}).items():
        if value is not None:
            set_key(out, FLAG_KEYS[name], value)
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ConfigError(f"--set {item!r}: expected key=value")
        set_key(out, key.strip(), parse_value(raw.strip()))
    return out


def load_tree(path: str | Path) -> tuple[Path, str, dict]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ConfigError(f"no deploy file at {p}")
    text = p.read_text()
    try:
        return p, text, tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"deploy file {p}: {exc}") from exc


def _check_keys(table: dict, allowed, where: str) -> None:
    unknown = set(table) - set(allowed)
    if unknown:
        raise ConfigError(f"deploy file {where} has unknown keys "
                          f"{sorted(unknown)}; known: {sorted(allowed)}")


# --- the policy class's declarations -----------------------------------------


def trained_values(cls) -> dict:
    """The class's `trained_action_horizon` / `trained_control_hz`, as
    DeployConfig field names; {} for no class, or attributes left at None.

    Read from the CLASS, never an instance: resolution happens before P5
    constructs the policy, and a class attribute needs no construction. `cls`
    is a class or its `describe.PolicyInfo`."""
    if cls is None:
        return {}
    from g1d_deploy.policy_contract.base import class_name
    name = class_name(cls)
    out = {}
    h = getattr(cls, "trained_action_horizon", None)
    hz = getattr(cls, "trained_control_hz", None)
    if h is not None:
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
            raise ConfigError(f"{name}.trained_action_horizon = {h!r}; "
                              f"want a positive int or None")
        out["action_horizon"] = int(h)
    if hz is not None:
        if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not hz > 0:
            raise ConfigError(f"{name}.trained_control_hz = {hz!r}; "
                              f"want a positive number of Hz or None")
        out["control_hz"] = float(hz)
    return out


def _check_end_effector(cls, hw) -> None:
    """A Policy class's `end_effector` against the robot file's, at load —
    before any process exists (P5 checks it again, `policy_contract.make_policy`).
    `cls` is the class's `describe.PolicyInfo`."""
    from g1d_deploy.policy_contract.base import PolicyError, check_end_effector
    if not cls.is_policy:
        return
    try:
        check_end_effector(cls, hw)
    except PolicyError as exc:
        raise ConfigError(str(exc)) from None


def _class_attr(cls, name: str):
    return getattr(cls, name, None) if cls is not None else None


# --- flattening the tree into DeployConfig fields ----------------------------


def _hand(value, hw: Hardware, where: str) -> np.ndarray:
    """A deploy-file hand value -> the (2, 6) block. `"open"`, `"closed"`, one
    grip value for both hands, or `[left, right]` — in the hand's own units
    (Dex-1 radians, 0 = closed; BrainCo [0, 1], 0 = open)."""
    closed = hw.hand_max if hw.hand_close_is_max else hw.hand_min
    if value == "open":
        grips = [hw.hand_open] * schema.N_HANDS
    elif value == "closed":
        grips = [closed] * schema.N_HANDS
    elif isinstance(value, (int, float)):
        grips = [float(value)] * schema.N_HANDS
    elif isinstance(value, list) and len(value) == schema.N_HANDS:
        grips = [float(v) for v in value]
    else:
        raise ConfigError(f"deploy file {where} = {value!r}: \"open\", "
                          f"\"closed\", a number, or [left, right]")
    out = np.zeros((schema.N_HANDS, schema.HAND_DIM))
    out[:, 0] = grips
    return out


def _flatten(tree: dict, hw: Hardware) -> dict:
    """The deploy tree -> `{DeployConfig field: value}` for every key it SETS.

    Pure: no layering here, so the same function reads the file alone and the
    file-plus-overrides, and the difference between the two is the command
    line's contribution.
    """
    _check_keys(tree, _TOP, "top level")
    out: dict = {}
    if "prompt" in tree:
        out["prompt"] = str(tree["prompt"])

    pol = tree.get("policy", {})
    _check_keys(pol, _POLICY, "[policy]")
    if "anchor_camera" in pol:
        out["primary_camera"] = str(pol["anchor_camera"])
    if "uses_cameras" in pol:
        out["observed_cameras"] = tuple(str(c) for c in pol["uses_cameras"])
    if "env" in pol:
        out["policy_env"] = {str(k): str(v) for k, v in pol["env"].items()}
    if "python" in pol:
        # As written; made absolute in `resolve`, which knows the directory.
        out["policy_python"] = str(pol["python"])

    _check_keys(tree.get("checkpoint", {}), _CHECKPOINT, "[checkpoint]")

    home = tree.get("home", {})
    _check_keys(home, _POSE | set(_HOME_EXTRA), "[home]")
    ext = tree.get("exit", {})
    _check_keys(ext, _POSE, "[exit]")
    for prefix, table in (("home", home), ("exit", ext)):
        if "arm" in table:
            out[f"{prefix}_q"] = np.asarray(table["arm"], dtype=float)
        if "waist" in table:
            out[f"{prefix}_waist"] = np.asarray(table["waist"], dtype=float)
        if "hand" in table:
            out[f"{prefix}_hand"] = _hand(table["hand"], hw, f"[{prefix}] hand")
    for key, name in _HOME_EXTRA.items():
        if key in home:
            out[name] = float(home[key])

    ctl = tree.get("control", {})
    if "ik_solver" in ctl:
        raise ConfigError(
            "deploy file [control] ik_solver is gone: `ik` names the solver "
            "now (arm_planner | arm_qp | waist_planner | waist_qp, or "
            "module:factory). Write ik = <that name>.")
    _check_keys(ctl, _CONTROL, "[control]")
    for key, name in _CONTROL.items():
        if name and key in ctl:
            out[name] = ctl[key]
    servo = dict(ctl.get("servo", {}))
    _check_keys(servo, _SERVO | set(_SERVO_MODULES), "[control.servo]")
    for key, name in _SERVO_MODULES.items():
        if key in servo:
            out[name] = str(servo.pop(key))
    if servo:
        out["servo"] = servo

    iv = tree.get("intervene", {})
    if "tcp" in iv:
        raise ConfigError(
            "deploy file [intervene] tcp is gone: the flange -> TCP offset is "
            "a property of the end effector, the robot file's "
            "[end_effector] tcp. Put it there (a custom robot file "
            "with extends = ...), or drop it to use the shipped value.")
    _check_keys(iv, _INTERVENE, "[intervene]")
    for key in _INTERVENE:
        if key in iv:
            v = iv[key]
            if key == "hand_tcp":
                v = tuple(tuple(float(x) for x in row) for row in v)
            out[f"intervene_{key}"] = v

    _check_keys(tree.get("record", {}), _RECORD, "[record]")
    if "labels" in tree:
        out["labels"] = {str(k): str(v) for k, v in tree["labels"].items()}
    return out


def _coerce(name: str, value):
    """TOML integers and floats into what each field means."""
    if name in ("replan_period_ticks", "action_horizon", "sample_steps",
                "ik_max_iters", "solve_lead_ticks", "plan_lookahead_ticks"):
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{name} = {value} must be a whole number")
        return int(value)
    if name in ("control_hz", "state_history_s", "ik_posture_cost",
                "ik_collision_min_dist", "ik_early_stop"):
        return float(value)
    if name == "sync":
        return bool(value)
    return value


# --- resolution ---------------------------------------------------------------


def _record_dir(rec: dict, machine, base_dir: Path, path) -> Path:
    """Where this launch records, first found wins:

    1. `[record] out` (or `--out`): relative to the deploy file;
    2. the machine file's `[paths] records` / `<task>`, where `task` is
       `[record] task` or the deploy file's folder name — one box, one record
       root, one folder per task;
    3. `rollout_record/` beside the deploy file.
    """
    if rec.get("out"):
        out = Path(str(rec["out"])).expanduser()
        return (out if out.is_absolute() else base_dir / out).resolve()
    root = getattr(machine, "records", "") if machine is not None else ""
    if root:
        task = str(rec.get("task") or (Path(path).parent.name if path else ""))
        if not task or "/" in task or task in (".", ".."):
            raise ConfigError(
                f"[record] task = {task!r}: the record folder under the "
                f"machine's [paths] records ({root}) must be one plain name")
        return (Path(root) / task).resolve()
    return (base_dir / DEFAULT_RECORD_DIR).resolve()


def resolve(deploy_path: str | Path | None = None, *, tree: dict | None = None,
            flags: dict | None = None, sets=(), checkpoint: str | None = None,
            machine=None, record: bool = True) -> Resolved:
    """Resolve a deploy file plus command-line overrides into a launch.

    `deploy_path` names the file (its directory anchors every relative path in
    it); `tree` may be given instead, for a programmatic launch. Raises
    `ConfigError` for anything the files or flags get wrong; `validate.validate`
    is still the caller's, on the result.
    """
    if deploy_path is not None:
        path, text, base_tree = load_tree(deploy_path)
        base_dir = path.parent
    else:
        path, text, base_tree = None, None, copy.deepcopy(tree or {})
        base_dir = Path.cwd()
    final_tree = apply_overrides(base_tree, flags, sets, checkpoint)

    # The robot first: hand values ("open") and camera names mean nothing
    # without it.
    robot_name = final_tree.get("robot", DEFAULT_ROBOT)
    robot = load_robot(robot_name, relative_to=base_dir)
    if not isinstance(machine, Machine):
        machine = load_machine(machine)
    hw = load_hardware(robot, machine)

    # The policy class: described in its own interpreter, never constructed
    # (module docstring).
    pol = final_tree.get("policy", {})
    policy_entry = ""
    policy_python = (policy_interpreter(pol["python"], base_dir)
                     if pol.get("python") else "")
    cls = None
    if pol.get("entry"):
        from g1d_deploy.policy_contract.describe import DescribeError, describe
        policy_entry = entry.normalize(str(pol["entry"]), base_dir=base_dir)
        try:
            cls = describe(policy_entry, python=policy_python or None)
        except DescribeError as exc:
            raise ConfigError(f"[policy] entry {policy_entry!r}: {exc}") from None
        _check_version(cls)
        _check_end_effector(cls, hw)

    trained = trained_values(cls)
    from_file = _flatten(base_tree, hw)
    from_all = _flatten(final_tree, hw)

    values: dict = {}
    sources: dict = {}
    for name, v in INFRA_DEFAULTS.items():
        values[name], sources[name] = v, "infra"
    for name, v in trained.items():
        values[name], sources[name] = v, "trained"
    for attr, name in (("anchor_camera", "primary_camera"),
                       ("uses_cameras", "observed_cameras")):
        v = _class_attr(cls, attr)
        if v is not None:
            values[name] = v if isinstance(v, str) else tuple(v)
            sources[name] = "trained"
    for name, v in from_all.items():
        values[name] = v
        sources[name] = ("deploy" if name in from_file and _same(from_file[name], v)
                         else "cli")

    # The home posture: the deploy file's, else the infra's default.
    if "home_q" not in values:
        values["home_q"], sources["home_q"] = np.array(DEFAULT_HOME_ARM), "infra"
    if "home_waist" not in values:
        values["home_waist"], sources["home_waist"] = np.zeros(schema.WAIST_DOF), "infra"
    if "home_hand" not in values:
        values["home_hand"], sources["home_hand"] = _hand("open", hw, "[home] hand"), "infra"
    for part in ("q", "waist", "hand"):
        if f"exit_{part}" not in values:
            values[f"exit_{part}"] = np.array(values[f"home_{part}"], copy=True)
            sources[f"exit_{part}"] = f"= home_{part}"

    # Every policy-facing camera name is a VIEW: a side_by_side camera is its
    # two halves (`resolved.camera_views`).
    views = camera_views(robot.cameras)
    if "primary_camera" not in values:
        # One WIRE camera needs no naming even when it is split: both eyes of a
        # side_by_side frame carry the same `t_recv`, so either one pins the
        # same instant, and the left (the first view) is taken.
        if len(robot.cameras) == 1:
            values["primary_camera"] = next(iter(views))
            sources["primary_camera"] = "infra"
        else:
            raise ConfigError(
                f"no anchor camera: the policy class declares no `anchor_camera` "
                f"and the deploy file sets no [policy] anchor_camera. Its frame "
                f"time IS t_anchor, so it has to be named; this robot has "
                f"{sorted(views)}.")

    # Camera history sizes the arena, and the arena is built by
    # every process from the config alone — so the class's declaration is
    # copied into it here, the one place the class is known before anything
    # is spawned. `"state"` is not a camera; its ring is `state_history_s`.
    hist = _class_attr(cls, "history") or {}
    camera_history = {str(k): float(v) for k, v in hist.items() if k != "state"}
    stray = sorted(set(camera_history) - set(views))
    if stray:
        raise ConfigError(
            f"the policy's history names {stray}, which are not cameras of this "
            f"robot ({sorted(views)})."
            + "".join(not_a_view_hint(n, robot.cameras) for n in stray))
    if camera_history:
        values["camera_history_s"] = camera_history
        sources["camera_history_s"] = "policy"

    if "servo" in values:
        values["servo"] = ServoShaping(**values["servo"])
    # Plug-ins named by FILE (`"my_ik.py:factory"`) are relative to this
    # deploy file, like `[policy] entry`; made absolute here, once, so every
    # spawned process imports the same file wherever it was started from.
    for key in ("ik", "blend", "servo_smoother", "servo_feedforward"):
        spec = values.get(key)
        if isinstance(spec, str) and ":" in spec and \
                entry.is_path(spec.rpartition(":")[0]):
            values[key] = entry.normalize(spec, base_dir=base_dir)
    values["policy_env"] = {**machine.env, **values.get("policy_env", {})}
    if machine.env:
        sources["policy_env"] = sources.get("policy_env", "machine")

    ck = final_tree.get("checkpoint", {})
    ckpt = checkpoints.resolve(cls, path=ck.get("path"), run=ck.get("run"),
                               step=ck.get("step"), base_dir=base_dir)
    ck_file = base_tree.get("checkpoint", {})
    sources["checkpoint"] = "deploy" if ck == ck_file else "cli"

    if "policy_python" in values:
        values["policy_python"] = policy_python

    values = {k: _coerce(k, v) for k, v in values.items()}
    cfg = DeployConfig(prompt=values.pop("prompt", ""), checkpoint=ckpt,
                       cameras=dict(robot.cameras),
                       robot=str(robot_name), policy_entry=policy_entry,
                       # No entry = the EchoPolicy dry run, which is eef.
                       policy_action_mode=(cls.action_mode if cls is not None
                                           else "eef") or "",
                       **values)

    warnings = []
    if "action_horizon" in trained and cfg.H > trained["action_horizon"]:
        raise ConfigError(
            f"action_horizon = {cfg.H} ({sources['action_horizon']}) exceeds the "
            f"{trained['action_horizon']} the policy declares "
            f"(trained_action_horizon); a chunk cannot contain rows the policy "
            f"never produces.")
    if "control_hz" in trained and cfg.control_hz != trained["control_hz"]:
        warnings.append(
            f"control_hz = {cfg.control_hz:g} ({sources['control_hz']}) but the "
            f"policy declares {trained['control_hz']:g} Hz (trained_control_hz): every action "
            f"tick now lasts {1 / cfg.control_hz * 1e3:.1f} ms instead of "
            f"{1 / trained['control_hz'] * 1e3:.1f} ms. A deliberate speed A/B, "
            f"or a mistake.")

    rec = final_tree.get("record", {})
    record_dir = None
    if record and rec.get("enabled", True):
        record_dir = _record_dir(rec, machine, base_dir, path)

    return Resolved(cfg=cfg, hw=hw, robot=robot, machine=machine,
                    policy=cls, deploy_path=str(path) if path else None,
                    deploy_text=text, tree=final_tree, record_dir=record_dir,
                    sources=sources, trained=trained, warnings=tuple(warnings))


def policy_interpreter(value, base_dir: Path) -> str:
    """`[policy] python` -> an absolute path. A path is taken relative to the
    deploy file's directory; a bare name (`python3.11`) is looked up on PATH.
    NOT resolved through symlinks: a venv's `bin/python` is a symlink to the
    base interpreter, and resolving it would silently leave the venv."""
    text = os.path.expanduser(str(value))
    if os.path.isabs(text):
        return text
    if "/" in text:
        return os.path.abspath(os.path.join(base_dir, text))
    found = shutil.which(text)
    if found is None:
        raise ConfigError(f"[policy] python = {value!r}: not found on PATH; "
                          f"give the venv's interpreter as a path")
    return os.path.abspath(found)


def _check_version(info) -> None:
    """The policy venv's g1d_deploy must speak the runtime's arena schema:
    P5 attaches to the arena with it. `Arena.attach` would refuse the
    mismatch too, but only after the whole launch has been paid for."""
    if info.schema_version and info.schema_version != schema.VERSION:
        raise ConfigError(
            f"the policy interpreter {info.python} has a g1d_deploy with arena "
            f"schema version {info.schema_version}; this runtime's is "
            f"{schema.VERSION}. Install the same g1d_deploy into both venvs.")


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


__all__ = ["DEFAULT_RECORD_DIR", "FLAG_KEYS", "INFRA_DEFAULTS", "Resolved",
           "apply_overrides", "load_tree", "parse_value", "policy_interpreter",
           "resolve",
           "set_key", "trained_values"]
