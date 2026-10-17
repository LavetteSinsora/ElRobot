"""Fixtures every section's tests share: a fresh arena per test (it is ~0.4 MB and creates in
well under a millisecond), the shipped profile on the simulated bus, and the repo's fake
serial bus (`ElRobot/tests/fake_bus.py`) on the import path."""

from __future__ import annotations

import os
import sys
import uuid

import pytest

from elrobot_deploy.config_loading.config import REPO, load
from elrobot_deploy.runtime.shared_memory import Arena, build, open_all

#: `fake_bus.py` lives with the bring-up package's tests.
sys.path.insert(0, str(REPO / "tests"))


def arena_name() -> str:
    # macOS caps a shared-memory name at 31 characters.
    return f"elt{os.getpid() % 100000}{uuid.uuid4().hex[:8]}"


@pytest.fixture
def arena():
    a = Arena.create(arena_name(), build())
    try:
        yield a
    finally:
        a.close()
        a.unlink()


@pytest.fixture
def rings(arena):
    return open_all(arena)


@pytest.fixture
def cfg():
    """The shipped profile, on simulated servos."""
    return load(sets=["bus.driver=sim"])
