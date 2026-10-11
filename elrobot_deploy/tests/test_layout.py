"""The package's top level is its seven role groups and nothing else.

`README.md` ("Source layout") says what each holds. A module added beside them
would be an eighth, unexplained group; a pre-2026-09-28 name coming back would be
a second place for the same code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import elrobot_deploy

PKG = Path(elrobot_deploy.__file__).resolve().parent

GROUPS = {"cli", "policy_contract", "config_loading", "robot", "runtime",
          "rollout_record_utils", "dashboards"}


def test_the_top_level_is_the_role_groups():
    entries = {p.name for p in PKG.iterdir() if p.name != "__pycache__"}
    assert entries == GROUPS | {"__init__.py", "__main__.py"}


@pytest.mark.parametrize("old", [
    "affinity", "arena", "blend", "board", "checkpoints", "clock", "codec",
    "config", "crop", "dashboard", "deployfile", "end_effectors", "entry",
    "fish_eye_image_server", "ik", "ik_solvers", "inspection", "intervene",
    "kin", "machine", "model", "pico_service", "policy", "preflight",
    "processes", "provenance", "record_read", "records", "settings", "wbik",
    "wire"])
def test_no_pre_relayout_name_is_importable(old):
    assert importlib.util.find_spec(f"elrobot_deploy.{old}") is None
