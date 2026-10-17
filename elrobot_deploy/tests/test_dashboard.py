"""The dashboard (`dashboard/`): the URDF as the page draws it, the state it polls, the requests
its buttons store, and the files it serves."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from elrobot_deploy.dashboard.robot import describe
from elrobot_deploy.dashboard.server import Source, serve
from elrobot_deploy.runtime.shared_memory import schema


def test_the_urdf_description_has_every_joint_and_mesh(cfg):
    robot, meshes = describe(cfg.urdf)
    names = {j["name"] for j in robot["joints"]}
    assert set(robot["arm_joints"]) <= names and len(robot["arm_joints"]) == 8
    mimic = [j for j in robot["joints"] if j["mimic"]]
    assert {j["mimic"]["joint"] for j in mimic} == {"rev_motor_08"} and len(mimic) == 2
    assert robot["root"] == "base_link" and robot["tcp"] in robot["links"]
    assert meshes and all(m.is_file() for m in meshes)


def test_the_state_reads_the_arena(rings, cfg):
    q = cfg.poses.home
    rings["state"].write(t=0.0, q=q, q_cmd=q + 0.01)
    rings["control"].session = schema.CHECK
    rings["control"].wait = schema.WAIT_HOME
    rings["control"].set_report(json.dumps({"servos": [], "problems": [], "warnings": []}))
    s = Source(cfg, rings).state()
    assert s["live"] and s["session"] == "CHECK" and s["wait"] == "HOME"
    assert np.allclose(s["measured"], q) and np.allclose(s["command"], q + 0.01)
    assert s["report"] == {"servos": [], "problems": [], "warnings": []}
    assert s["phone"]["samples"] == 0 and np.allclose(s["home"], cfg.poses.home)


def test_without_a_session_it_is_a_preview(cfg):
    s = Source(cfg).state()
    assert not s["live"] and len(s["home"]) == 8 and len(s["lower"]) == 8
    assert not Source(cfg).request("HOME")


@pytest.fixture
def server(rings, cfg):
    httpd = serve(Source(cfg, rings), "127.0.0.1", 0)
    yield f"http://127.0.0.1:{httpd.server_address[1]}", rings
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_server_serves_the_page_the_robot_and_the_meshes(server):
    base, _ = server
    status, ctype, body = get(base + "/")
    assert status == 200 and "html" in ctype and b"importmap" in body
    for path in ("/static/app.js", "/static/vendor/three/three.module.min.js",
                 "/static/vendor/three/STLLoader.js"):
        assert get(base + path)[0] == 200, path
    robot = json.loads(get(base + "/api/robot")[2])
    first = robot["links"]["base_link"]["visuals"][0]["mesh"]
    status, ctype, body = get(base + first)
    assert status == 200 and len(body) > 1000
    assert json.loads(get(base + "/api/state")[2])["live"]


@pytest.mark.parametrize("path", ["/static/../server.py", "/static/%2e%2e/server.py",
                                  "/mesh/999", "/mesh/x", "/etc/passwd"])
def test_the_server_serves_nothing_else(server, path):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + path)
    assert e.value.code == 404


def test_a_button_stores_a_request_and_nothing_else(server):
    base, rings = server
    assert post(base + "/api/request", {"request": "home"}) == (200, {"ok": True,
                                                                      "request": "HOME"})
    assert rings["control"].request == schema.REQ_HOME
    status, body = post(base + "/api/request", {"request": "MOVE_ARM"})
    assert status == 409 and not body["ok"]
    assert rings["control"].request == schema.REQ_HOME
