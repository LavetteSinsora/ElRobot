"""P10 — the live dashboard.

The failures worth catching here are the ones that produce a page that LOOKS
right: an overlay drawn from a target that is not in force, a banner rendered
from what the page sent rather than from what the arena says, a command that
reports success without storing anything.

The server is exercised over a real arena on a real socket, because most of what
could go wrong here is in the seam between them.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from elrobot_deploy.config_loading import resolved as cfgmod
from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.dashboards.live.scene import LiveScene
from elrobot_deploy.dashboards.live.server import Live, serve

T0 = 1000.0


@pytest.fixture
def scene(cfg, hw, rings):
    """The compiled model. Slow enough (a model compile and a mesh unpack) that
    the tests below share one."""
    return LiveScene(cfg, hw, rings)


@pytest.fixture
def live(cfg, hw, rings, scene):
    return Live(cfg, hw, rings, scene, deploy_py="my_task/deploy.py",
                step="9999")


@pytest.fixture
def url(live):
    """A server on an ephemeral port. Port 0 rather than a fixed one so a
    developer's own dashboard cannot make the suite fail."""
    server = serve(live, host="127.0.0.1", port=0)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.stopping.set()
        server.shutdown()
        server.server_close()


def _state(rings, hw, q_body17, *, grip=0.0, t=T0):
    """One measured sample, the way P1 publishes it."""
    q29 = np.zeros(schema.N_JOINTS)
    q29[list(hw.joints_arm) + list(hw.joints_waist)] = q_body17
    rings["state"].write(t_recv=t, q=q29, dq=np.zeros(schema.N_JOINTS),
                         flange=np.tile(np.array([0.3, 0.0, 0.2, 1.0, 0, 0, 0]),
                                        (schema.N_HANDS, 1)),
                         grip_meas=np.full(schema.GRIP_DIM, grip), t_grip_recv=t)


def _target(rings, k, *, q=0.0, pose=True):
    """One target row: the joints, the hand command and the IK's columns.

    Was two writes, one per ring, held together by the shared-deadline rule. It has to be ONE now —
    two writes to the same `k` land in the same slot, and the second would
    overwrite the first's `q` and `err`. That is the merge working: a partial
    write is no longer expressible without visibly destroying the rest of the
    row."""
    body = np.full(schema.BODY_DOF, q)
    poses = (np.tile(np.array([0.3, 0.0, 0.2, 1.0, 0, 0, 0]),
                     (schema.N_HANDS, 1)) if pose
             else np.full((schema.N_HANDS, schema.POSE_DIM), np.nan))
    rings["targets"].write(k, T0, body,
                           np.zeros((schema.N_HANDS, schema.HAND_DIM)),
                           err=0.001, pose=poses, intervene=0)


def _get(url, path):
    with urllib.request.urlopen(f"{url}{path}", timeout=5) as r:
        return r.status, r.read(), dict(r.headers)


def _get_json(url, path):
    _, body, _ = _get(url, path)
    return json.loads(body)


def _frame(url):
    """`/api/frame`: a uint32 length, that many bytes of JSON, padding, arrays.

    The padding is the point — see `test_the_payload_starts_on_an_alignment_the
    _browser_can_read`. The payload is returned already sliced off it, so the
    `layout` offsets index what this returns.
    """
    _, body, _ = _get(url, "/api/frame")
    n = int.from_bytes(body[:4], "little")
    base = (4 + n + 3) & ~3
    return json.loads(body[4:4 + n]), body[base:]


def _post(url, payload):
    req = urllib.request.Request(
        f"{url}/api/command", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# --- The write path ---------------------------------------------


def test_a_command_stores_the_request_word_and_nothing_else(url, rings):
    """P10's entire write surface. The arena is otherwise read-only to it, and
    the session word in particular has one writer."""
    before = (rings["control"].session, rings["control"].k_exec)

    status, body = _post(url, {"request": "reset"})

    assert status == 200 and body["stored"] == "reset"
    assert rings["control"].request == schema.REQ_RESET
    assert (rings["control"].session, rings["control"].k_exec) == before


@pytest.mark.parametrize("name,want", [
    ("start", schema.REQ_START),
    ("reset", schema.REQ_RESET),
    ("exit", schema.REQ_EXIT),
])
def test_every_button_maps_to_the_word_p0_reads(url, rings, name, want):
    _post(url, {"request": name})
    assert rings["control"].request == want


def test_an_unknown_request_is_refused_rather_than_guessed(url, rings):
    """The verbs are a fixed table, not `getattr(schema, ...)`: a typo in a POST
    body must not be able to reach for an arbitrary attribute of the schema."""
    status, body = _post(url, {"request": "REQ_EXIT"})

    assert status == 400
    assert "unknown request" in body["error"]
    assert rings["control"].request == schema.REQ_NONE


# --- The banner -------------------------------------------------------------


def test_the_banner_reports_the_arena_and_not_what_was_sent(url, rings):
    """The whole reason a dropped request is tolerable. The page renders these
    words, so a command that lost its race shows up as a chip that never
    changed — rather than being hidden behind an optimistic label."""
    rings["control"].session = schema.READY
    rings["control"].k_exec = 41

    _post(url, {"request": "start"})
    banner = _get_json(url, "/api/banner")

    assert banner["session_name"] == "READY"
    assert banner["k_exec"] == 41
    # Still pending: nothing has consumed it, and the banner says so.
    assert banner["request_name"] == "START"

    # P0 takes it. The banner follows the word, with nothing to invalidate.
    rings["control"].request = schema.REQ_NONE
    assert _get_json(url, "/api/banner")["request_name"] == "NONE"


def test_the_banner_names_which_checkpoint_is_being_served(url):
    """Free — it is already in `cfg` — and it closes the failure of finishing a
    session unsure which weights produced the rollouts just recorded."""
    banner = _get_json(url, "/api/banner")
    assert banner["deployment"] == "my_task"
    assert banner["step"] == "9999"


# --- The robot --------------------------------------------------------------


def test_a_frame_before_p1_has_published_is_not_ready_rather_than_wrong(url):
    """No state, no pose. The alternative is a robot drawn at whatever zero
    happens to be, which is a picture of a machine that does not exist."""
    head, payload = _frame(url)
    assert head["ready"] is False and payload == b""
    # The banner still works: the phase is knowable before the robot is.
    assert head["banner"]["session_name"] == "INIT"


def test_measured_is_drawn_in_every_phase(url, rings, hw):
    """The dashboard is a window onto the robot before it is a window onto a
    policy. A measurement exists at INIT, mid-ramp and at READY alike."""
    _state(rings, hw, np.zeros(schema.BODY_DOF))
    for phase in (schema.INIT, schema.HOME, schema.WARMUP, schema.READY):
        rings["control"].session = phase
        head, payload = _frame(url)
        assert head["ready"] is True, phase
        span = head["layout"]["transforms"]
        assert span[1] > 0 and len(payload) >= span[0] + span[1]


def test_no_commanded_overlay_outside_a_rollout(url, rings, hw):
    """The operator's own rule: measured only during HOME and READY.

    The ring may genuinely hold P0's homing ramp — real targets, honestly
    written — and drawing them would show the arm chasing a ramp nobody chose.
    """
    _state(rings, hw, np.zeros(schema.BODY_DOF))
    rings["control"].k_exec = 7
    _target(rings, 7)

    for phase in (schema.HOME, schema.WARMUP, schema.READY):
        rings["control"].session = phase
        head, _ = _frame(url)
        assert head["solved"] is None, phase
        assert head["ghost"] is None, phase
        assert "solved" not in head["layout"], phase


def test_both_overlays_appear_once_a_rollout_is_running(url, rings, hw):
    _state(rings, hw, np.zeros(schema.BODY_DOF))
    rings["control"].session = schema.ACTIVE
    rings["control"].k_exec = 7
    _target(rings, 7)

    head, payload = _frame(url)

    assert head["solved"]["k"] == 7
    assert head["solved"]["err"] == pytest.approx(0.001)
    assert head["ghost"] is not None
    for name in ("solved", "solved_tcp", "ghost", "ghost_tcp"):
        span = head["layout"][name]
        assert span[1] > 0 and len(payload) >= span[0] + span[1], name


def test_a_joint_space_ramp_has_solved_but_no_ghost(url, rings, hw):
    """The reason the two overlays are separate rather than one switch. P0's
    ramps carry joint angles with `pose` deliberately NaN — they have no
    Cartesian target behind them — so the solved robot is drawable and
    the ghost is not. Inventing one would be a fabricated command."""
    _state(rings, hw, np.zeros(schema.BODY_DOF))
    rings["control"].session = schema.ACTIVE
    rings["control"].k_exec = 9
    _target(rings, 9, pose=False)

    head, _ = _frame(url)

    assert head["solved"] is not None and head["solved"]["k"] == 9
    assert head["ghost"] is None


def test_a_joint_mode_tick_says_why_there_is_no_commanded_pose(url, rings, hw):
    """Joint-mode chunks carry `pose` = NaN by contract. The joint
    targets are drawn; the Cartesian overlay is absent AND says so, so the page
    can grey it rather than look broken."""
    _state(rings, hw, np.zeros(schema.BODY_DOF))
    rings["control"].session = schema.ACTIVE
    rings["control"].k_exec = 9
    _target(rings, 9, q=0.1, pose=False)

    head, _ = _frame(url)

    assert head["solved"]["q"][0] == pytest.approx(0.1)
    assert head["ghost"] is None
    assert "no Cartesian target" in head["ghost_absent"]


def test_the_overlay_follows_k_exec_and_not_the_newest_target(url, rings, hw):
    """`k_exec` names the tick being executed toward, which is what the
    operator means by "what is it being told to do". The ring also holds ticks
    P6 has solved ahead of it, and drawing those would show the future."""
    _state(rings, hw, np.zeros(schema.BODY_DOF))
    rings["control"].session = schema.ACTIVE
    rings["control"].k_exec = 4
    _target(rings, 4, q=0.10)
    _target(rings, 5, q=0.90)                   # already solved, not yet executed

    head, _ = _frame(url)

    assert head["solved"]["k"] == 4
    assert head["solved"]["q"][0] == pytest.approx(0.10)


# --- The static half --------------------------------------------------------


def test_the_model_is_described_once_and_names_its_cameras(url, cfg):
    meta = _get_json(url, "/api/meta")
    assert meta["n_geoms"] > 0 and meta["n_triangles"] > 0
    assert meta["solved_geoms"] and meta["ghost_geoms"]
    assert [c["name"] for c in meta["cameras"]] == sorted(cfg.cameras)

    # Nothing in this config is record-only, so every camera reads as an input
    # and exactly one of them is the primary.
    by_name = {c["name"]: c for c in meta["cameras"]}
    # The captured shape reaches the page, which needs it to letterbox a pane
    # rather than crop a frame to fill one.
    assert (by_name["head"]["height"], by_name["head"]["width"]) == (8, 16)
    assert (by_name["rwrist"]["height"], by_name["rwrist"]["width"]) == (8, 8)
    assert all(c["used"] for c in meta["cameras"])
    assert by_name["rwrist"]["primary"] is True
    assert by_name["head"]["primary"] is False
    assert sum(c["primary"] for c in meta["cameras"]) == 1


def test_a_camera_the_policy_is_not_reading_says_so_and_comes_last(
        cfg, hw, rings, scene):
    """`cfg.cameras` is what EXISTS; `cfg.observed` is what the policy reads.

    A pane captioned only with its name sits beside the views the arm is being
    driven from and looks exactly like them. That is fine while every camera is
    an input and wrong the moment one is not — the head is recorded on every
    run and read by nothing — so the operator is told which is which,
    and the record-only pane is moved out of the leading position that plain
    alphabetical ordering would have given it.
    """
    watched = cfgmod.replace(cfg, observed_cameras=("rwrist",))
    cams = Live(watched, hw, rings, scene).cameras()

    # Observed first, record-only last — NOT alphabetical, which would have put
    # `head` in the leftmost pane during a live run.
    assert [c["name"] for c in cams] == ["rwrist", "head"]
    assert cams[0]["used"] is True and cams[0]["primary"] is True
    assert cams[1]["used"] is False and cams[1]["primary"] is False


def test_the_dashboard_does_not_need_an_experiment_to_describe_a_camera(
        cfg, hw, rings, scene):
    """`camera_slots` belongs to an experiment's config subclass, not to the
    framework, and this module must not import an experiment to read one. A
    config that has none still describes every camera; only the slot NAME is
    missing, and the page falls back to saying "used"."""
    cams = Live(cfg, hw, rings, scene).cameras()
    assert not hasattr(cfg, "camera_slots")
    assert all(c["slot"] is None for c in cams)
    assert all(c["used"] for c in cams)


def test_a_binocular_camera_is_flagged_so_one_eye_can_be_shown(
        cfg, hw, rings, scene):
    """The head sends both eyes in one frame and the page shows the left one.
    Which frames those are is a property of the camera, so it rides along here
    rather than being guessed from an unusually wide aspect ratio."""
    wide = cfgmod.replace(cfg, cameras=dict(
        cfg.cameras, head=cfgmod.CameraSpec(8, 16, binocular=True)))
    cams = {c["name"]: c for c in Live(wide, hw, rings, scene).cameras()}
    assert cams["head"]["binocular"] is True
    assert cams["rwrist"]["binocular"] is False


def test_the_payload_starts_on_an_alignment_the_browser_can_read(url, rings, hw):
    """`new Float32Array(buffer, byteOffset, n)` THROWS on a byteOffset that is
    not a multiple of four — it does not round, and it does not warn.

    The frame response is a uint32 length, a JSON header of ARBITRARY length,
    then the arrays. Without padding between the header and the payload, where
    the payload begins depends on how many characters the JSON happened to need,
    and roughly three frames in four would fail to parse in the browser while
    every one of them looked correct from Python. So the padding is asserted on
    the wire, over headers of deliberately different lengths.
    """
    _state(rings, hw, np.zeros(schema.BODY_DOF))
    seen = set()
    for k, session in ((3, schema.ACTIVE), (4, schema.HOME), (5, schema.ACTIVE)):
        rings["control"].session = session
        rings["control"].k_exec = k
        _target(rings, k)

        _, body, _ = _get(url, "/api/frame")
        n = int.from_bytes(body[:4], "little")
        seen.add(n)
        base = (4 + n + 3) & ~3
        assert base % 4 == 0
        assert body[4 + n:base] == b"\0" * (base - 4 - n), "padding is not zeros"

        head = json.loads(body[4:4 + n])
        for name, (offset, nbytes) in head.get("layout", {}).items():
            assert (base + offset) % 4 == 0, f"{name} lands on a bad offset"
            assert base + offset + nbytes <= len(body), f"{name} runs off the end"

    assert len(seen) > 1, "every header was the same length; the test proved nothing"


def test_the_mesh_buffers_match_the_layout_they_are_sent_with(url):
    """The browser builds `Float32Array(buffer, offset, length)` from this map,
    which throws on a bad offset and silently draws nonsense on a bad length."""
    _, body, headers = _get(url, "/api/scene.bin")
    layout = json.loads(headers["X-Layout"])
    for name in ("vertices", "normals", "indices"):
        offset, nbytes = layout[name]
        assert offset % 4 == 0, f"{name} is not 4-byte aligned"
        assert offset + nbytes <= len(body)


# --- The camera stream ------------------------------------------------------


def test_the_stream_sends_the_bytes_the_robot_sent(url, rings):
    """Passthrough, not a re-encode. P2 keeps the compressed frame beside the
    pixels it decoded, so the operator sees exactly what the camera
    produced — one lossy generation, not two, and no CPU spent on either side."""
    import threading
    import time

    name = sorted(rings["cameras"])[0]
    payload = b"\xff\xd8\xff" + b"not-really-a-jpeg" + b"\xff\xd9"

    # A publisher, because the stream sends only frames it has not sent before —
    # which is the property that makes it self-pacing, and which means one
    # published frame produces exactly one part and then silence.
    stop = threading.Event()

    def publish():
        while not stop.is_set():
            rings["cameras"][name].publish(T0, jpeg=payload)
            time.sleep(0.005)

    pub = threading.Thread(target=publish, daemon=True)
    pub.start()

    chunks: list[bytes] = []

    def read():
        with urllib.request.urlopen(f"{url}/api/stream/{name}", timeout=5) as r:
            assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
            chunks.append(r.read(512))

    try:
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout=5)
    finally:
        stop.set()
        pub.join(timeout=2)

    assert chunks, "the stream sent nothing"
    assert payload in chunks[0], "the frame was re-encoded on its way out"
    assert b"Content-Type: image/jpeg" in chunks[0]
    assert chunks[0].count(payload) > 1, "the stream stopped after one frame"


def test_an_unknown_camera_is_named_rather_than_hung(url, rings):
    """A stream route that 404s is a broken `<img>`; one that blocks is a thread
    held forever on a name that will never publish."""
    status, body = None, None
    try:
        urllib.request.urlopen(f"{url}/api/stream/nosuchcam", timeout=5)
    except urllib.error.HTTPError as e:
        status, body = e.code, json.loads(e.read())

    assert status == 404
    assert "no such camera" in body["error"]
    assert sorted(rings["cameras"])[0] in body["error"]


def test_the_renderer_is_the_inspectors_own(url):
    """`gl.js` is served from the shared `dashboards/web/static/`, not copied. Two views of the
    same robot that disagree about what it looks like would be worse than one."""
    status, body, _ = _get(url, "/static/gl.js")
    assert status == 200 and b"export class Viewer" in body


def test_concurrent_frames_do_not_crash_the_process(live, rings, hw):
    """Two tabs poll `/api/frame` at once, each on its own server thread.

    `mj_forward` releases the GIL, so without serialisation both requests pose
    into the same MjData and the process segfaults — no traceback, no 500, just
    a dashboard that is gone while P0 still says READY. A crash here takes the
    whole test process down, which is the failure being guarded.
    """
    import threading
    import time

    _state(rings, hw, np.zeros(schema.BODY_DOF))
    stop = time.monotonic() + 3.0
    errors: list[BaseException] = []

    def poll():
        try:
            while time.monotonic() < stop:
                meta, _ = live.frame()
                assert meta["ready"]
        except BaseException as e:                        # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=poll) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# --- The rollout settings: the prompt and the Deploy settings ------------------------------


def _seed(rings, cfg):
    """What P0 writes at INIT, before P10 exists: the launch settings in both
    blocks and the launch prompt as the one in use."""
    from elrobot_deploy.config_loading.rollout_settings import RolloutSettings
    launch = RolloutSettings.of(cfg).to_record()
    rings["control"].request_config(launch)
    rings["control"].activate(launch, cfg.prompt)


def _post_to(url, path, payload):
    req = urllib.request.Request(
        f"{url}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_settings_reads_back_requested_and_active(url, rings, cfg):
    _seed(rings, cfg)
    st = _get_json(url, "/api/settings")
    assert st["requested"] == st["active"] == st["launch"]
    assert st["requested"]["replan_period"] == cfg.P
    assert st["limits"]["replan_period"] == [2, cfg.H]
    assert st["options"]["ik"] == ["arm_planner", "arm_qp", "waist_planner",
                                   "waist_qp"]
    assert st["launch"]["ik"] == "arm_planner", "the default solver"
    assert st["editable"] and st["generation"] == 0
    assert st["prompt"]["active"] == cfg.prompt
    assert st["refusal"] == {"seq": 0, "text": ""}


def test_a_settings_request_is_stored_but_not_activated(url, rings, cfg):
    """P10 writes `requested_config` and nothing else: the active block — and
    so every process's settings — moves only when P0 accepts a START."""
    _seed(rings, cfg)
    rings["control"].session = schema.READY
    code, st = _post_to(url, "/api/settings",
                        {"replan_period": 2, "servo": {"filter_hz": 0}})
    assert code == 200
    assert st["requested"]["replan_period"] == 2
    assert st["requested"]["servo"]["filter_hz"] == 0
    assert st["requested"]["servo"]["torque_ff"] == st["launch"]["servo"]["torque_ff"]
    assert st["active"] == st["launch"], "P10 activated something"
    assert rings["control"].rollout_generation == 0


@pytest.mark.parametrize("body, why", [
    ({"replan_period": 99}, "action_horizon"),
    ({"replan_period": 1}, ">= 2"),
    ({"ik": "waist", "sync": False}, "sync"),
    ({"ik": "waist_qp", "sync": False}, "sync"),
    ({"ik": "no_such_solver"}, "no_such_solver"),
    ({"bogus": 1}, "unknown settings"),
    ({"servo": {"arm_gain_scale": 9}}, "arm_gain_scale"),
])
def test_a_bad_settings_request_is_refused_at_once_and_not_stored(
        url, rings, cfg, body, why):
    _seed(rings, cfg)
    before = rings["control"].requested_config().copy()
    code, d = _post_to(url, "/api/settings", body)
    assert code == 400 and why in d["error"]
    assert rings["control"].requested_config() == before


def test_settings_and_prompt_are_refused_while_a_rollout_runs(url, rings, cfg):
    _seed(rings, cfg)
    rings["control"].session = schema.ACTIVE
    code, d = _post_to(url, "/api/settings", {"replan_period": 2})
    assert code == 409 and "between rollouts" in d["error"]
    code, d = _post_to(url, "/api/prompt", {"prompt": "open the box"})
    assert code == 409
    assert rings["control"].prompt_request() == (0, "")
    assert _get_json(url, "/api/settings")["editable"] is False


def test_a_prompt_is_stored_and_its_verdict_read_back(url, rings, cfg):
    """The policy judges (in P5); the page learns the answer by matching the
    verdict's `of` to the seq it was given."""
    _seed(rings, cfg)
    rings["control"].session = schema.READY
    code, d = _post_to(url, "/api/prompt", {"prompt": "open the box"})
    assert code == 200 and d["seq"] == 2
    assert rings["control"].prompt_request() == (2, "open the box")
    st = _get_json(url, "/api/settings")
    assert st["prompt"]["request"] == {"seq": 2, "text": "open the box"}
    assert st["prompt"]["verdict"]["of"] != 2, "judged before P5 looked"

    rings["control"].judge(2, schema.PROMPT_REFUSED, "9 tokens > 8")
    v = _get_json(url, "/api/settings")["prompt"]["verdict"]
    assert v == {"of": 2, "name": "refused", "reason": "9 tokens > 8"}
    # The prompt in use is still the launch one: nothing was accepted.
    assert _get_json(url, "/api/banner")["prompt"] == cfg.prompt


@pytest.mark.parametrize("prompt", ["", "   ", None,
                                    "x" * (schema.PROMPT_BYTES + 1)])
def test_an_empty_or_oversized_prompt_is_refused_before_it_is_stored(
        url, rings, cfg, prompt):
    _seed(rings, cfg)
    code, d = _post_to(url, "/api/prompt", {"prompt": prompt})
    assert code == 400 and "prompt" in d["error"]
    assert rings["control"].prompt_request() == (0, "")


def test_the_banner_echoes_the_prompt_in_use_not_the_launch_one(url, rings,
                                                                 cfg):
    _seed(rings, cfg)
    rings["control"].activate(rings["control"].requested_config(), "stack cups")
    rings["control"].rollout_generation = 1
    banner = _get_json(url, "/api/banner")
    assert banner["prompt"] == "stack cups" and banner["generation"] == 1


def test_a_refused_start_is_reported(url, rings, cfg):
    _seed(rings, cfg)
    rings["control"].refuse("replan_period_ticks = 9 > action_horizon = 8")
    assert _get_json(url, "/api/settings")["refusal"] == {
        "seq": 2, "text": "replan_period_ticks = 9 > action_horizon = 8"}


def test_the_page_carries_the_prompt_field_and_the_settings_menu(url):
    _, body, _ = _get(url, "/")
    html = body.decode()
    for needle in ('id="prompt-form"', 'id="prompt-input"', 'id="btn-settings"',
                   'id="settings-panel"', 'name="replan_period"',
                   'name="servo.arm_gain_scale"', 'name="sample_steps"'):
        assert needle in html, needle


def test_joint_actions_say_the_ik_is_not_used(cfg, hw, rings, scene):
    """In joint mode P6 runs no IK: the status line and the settings panel
    say so instead of offering a solver."""
    jcfg = cfgmod.replace(cfg, policy_action_mode="joint")
    _seed(rings, jcfg)
    live = Live(jcfg, hw, rings, scene, deploy_py="t", step=None)
    assert live.settings()["joint_actions"] is True
    r = live.rollout()
    assert (r["ik"], r["ik_mode"]) == ("not used", "joint actions")
    eef = Live(cfg, hw, rings, scene, deploy_py="t", step=None)
    assert eef.settings()["joint_actions"] is False
    assert eef.rollout()["ik"] == cfg.ik
