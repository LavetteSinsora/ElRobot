"""`dashboards/inspect/overlays.py` — the IK bench's sidecar, read.

A recording offers exactly the solved methods it has, by name, and nothing at
all when the sidecar is missing or of another layout version.
"""

from __future__ import annotations

import json

import numpy as np

from elrobot_deploy.dashboards.inspect import overlays


def _bench(tmp_path, names, k, version=overlays.BENCH_VERSION):
    n = len(k)
    out = {"version": np.array([version]), "k": np.asarray(k, np.int64),
           "methods": np.array(names),
           "meta": np.array([json.dumps({"methods": {
               nm: {"mode": "arm", "describe": f"about {nm}", "tcp_mean_mm": 1.5}
               for nm in names}})]),
           "recorded/err": np.zeros(n), "recorded/tcp_err": np.full(n, 0.002)}
    for nm in names:
        out[f"{nm}/arm"] = np.zeros((n, 14))
        out[f"{nm}/waist"] = np.zeros((n, 3))
        out[f"{nm}/waist_cmd"] = np.zeros((n, 3))
        out[f"{nm}/column"] = np.zeros(n)
        out[f"{nm}/base"] = np.zeros((n, 3))
        for f in ("err", "tcp_err", "rot_err", "tick_ms"):
            out[f"{nm}/{f}"] = np.zeros(n)
    np.savez_compressed(tmp_path / overlays.BENCH_NAME, **out)


def test_nothing_on_disk_is_none(tmp_path):
    assert overlays.load(tmp_path) is None


def test_bench_alone_offers_its_methods_with_their_descriptions(tmp_path):
    _bench(tmp_path, ["plain_arm", "arm_pull_home"], [1, 2, 3])
    ov = overlays.load(tmp_path)
    assert list(ov["tiers"]) == ["plain_arm", "arm_pull_home"]
    assert ov.meta["names"]["arm_pull_home"]["describe"] == "about arm_pull_home"
    assert ov.meta["names"]["arm_pull_home"]["source"] == "bench"
    # The deployed row of the readout is the RECORDED configuration scored at
    # the TCP, which only the bench measures.
    assert np.allclose(ov["off/tcp_err"], 0.002)
    assert ov["arm_pull_home/arm"].shape == (3, 14)


def test_a_stale_bench_version_reads_as_absent(tmp_path):
    _bench(tmp_path, ["x"], [1, 2], version=overlays.BENCH_VERSION + 1)
    assert overlays.load(tmp_path) is None


def test_scene_meta_carries_the_overlay_meta():
    from elrobot_deploy.dashboards.inspect import scene as scene_mod

    s = scene_mod.Scene(
        transforms=np.zeros((2, 1, 12), np.float32), geoms=[],
        vertices=np.zeros(3, np.float32), normals=np.zeros(3, np.float32),
        indices=np.zeros(3, np.uint32), tcp=np.zeros((2, 2, 3), np.float32),
        ghost=np.zeros((2, 1, 12), np.float32), ghost_geoms=[],
        ghost_tcp=np.zeros((2, 2, 3), np.float32),
        ghost_valid=np.zeros(2, np.uint8),
        solved=np.zeros((2, 1, 12), np.float32), solved_geoms=[],
        solved_tcp=np.zeros((2, 2, 3), np.float32),
        solved_valid=np.zeros(2, np.uint8),
        solved_pinned=np.zeros(0, np.float32),
        solved_pinned_tcp=np.zeros(0, np.float32),
        wbik_meta={"names": {"a": {"source": "bench"}}, "notes": []})
    assert s.meta()["wbik_meta"]["names"]["a"]["source"] == "bench"
