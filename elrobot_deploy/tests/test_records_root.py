"""`[paths] records` in the machine file: one record root per box.

A deploy file with no `[record] out` records to `<records>/<task>/` (task =
`[record] task`, else the deploy file's folder), and `g1d_deploy inspect`
with no argument opens `<records>`. `out` / `--out` still win; with neither,
a run records beside its deploy file as before.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from g1d_deploy.config_loading import deploy_file
from g1d_deploy.config_loading.errors import ConfigError
from g1d_deploy.config_loading.machine_file import load_machine
from g1d_deploy.dashboards.inspect import options

TESTS = Path(__file__).resolve().parent
EXAMPLES = TESTS.parent / "examples"


def _machine(tmp_path, records: str | None) -> Path:
    text = (TESTS / "machine.toml").read_text()
    if records is not None:
        text += f'\n[paths]\nrecords = "{records}"\n'
    p = tmp_path / "machine.toml"
    p.write_text(text)
    return p


def _task(tmp_path, name="pick_cup", record: str = "") -> Path:
    from . import example_files
    p = example_files.task(tmp_path / name)
    p.write_text(p.read_text().replace("[record]\n", "[record]\n" + record))
    return p


def test_a_relative_root_is_the_machine_files_and_the_env_overrides_it(
        tmp_path, monkeypatch):
    monkeypatch.delenv("G1D_RECORDS", raising=False)
    m = load_machine(_machine(tmp_path, "recs"))
    assert m.records == str(tmp_path / "recs")
    monkeypatch.setenv("G1D_RECORDS", "/data/elsewhere")
    assert load_machine(_machine(tmp_path, "recs")).records == "/data/elsewhere"


def test_no_out_records_under_the_root_by_task(tmp_path, monkeypatch):
    monkeypatch.delenv("G1D_RECORDS", raising=False)
    machine = str(_machine(tmp_path, str(tmp_path / "recs")))
    r = deploy_file.resolve(_task(tmp_path), machine=machine)
    assert r.record_dir == tmp_path / "recs" / "pick_cup"
    r = deploy_file.resolve(_task(tmp_path, "t2", 'task = "wipe"\n'),
                            machine=machine)
    assert r.record_dir == tmp_path / "recs" / "wipe"


def test_out_and_the_flag_still_win(tmp_path, monkeypatch):
    monkeypatch.delenv("G1D_RECORDS", raising=False)
    machine = str(_machine(tmp_path, str(tmp_path / "recs")))
    f = _task(tmp_path, record='out = "here"\n')
    assert deploy_file.resolve(f, machine=machine).record_dir == f.parent / "here"
    assert deploy_file.resolve(f, machine=machine,
                               flags={"out": "/tmp/x"}).record_dir == Path("/tmp/x")


def test_without_a_root_it_records_beside_the_deploy_file(tmp_path, monkeypatch):
    monkeypatch.delenv("G1D_RECORDS", raising=False)
    f = _task(tmp_path)
    r = deploy_file.resolve(f, machine=str(_machine(tmp_path, None)))
    assert r.record_dir == f.parent / "rollout_record"


def test_a_task_must_be_one_plain_name(tmp_path, monkeypatch):
    monkeypatch.delenv("G1D_RECORDS", raising=False)
    with pytest.raises(ConfigError, match="one plain name"):
        deploy_file.resolve(_task(tmp_path, record='task = "a/b"\n'),
                            machine=str(_machine(tmp_path, str(tmp_path))))


def test_inspect_with_no_root_opens_the_machines_records(tmp_path, monkeypatch):
    monkeypatch.delenv("G1D_RECORDS", raising=False)
    o = options.resolve(None, machine=str(_machine(tmp_path, str(tmp_path / "r"))))
    assert o.root == tmp_path / "r"
    with pytest.raises(ConfigError, match=r"\[paths\] records"):
        options.resolve(None, machine=str(_machine(tmp_path, None)))
