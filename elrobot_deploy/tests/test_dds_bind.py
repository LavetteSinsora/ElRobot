"""`runtime/drivers/dds/domain.py::bind_address`: which address DDS binds by.

The failure it prevents was silent: a NIC holding the office AND the robot
address, bound by name, received zero `rt/lowstate` samples while P0 waited in
CHECK forever.
"""

from elrobot_deploy.runtime.drivers.dds import domain as dds


def test_a_nic_with_office_and_robot_addresses_binds_by_the_robot_one():
    assert dds.bind_address("enp4s0", ["192.168.1.111", "192.168.123.118"]) == "192.168.123.118"
    assert dds.bind_address("enp4s0", ["192.168.123.118", "10.0.0.2"]) == "192.168.123.118"


def test_one_address_keeps_binding_by_name():
    assert dds.bind_address("enp4s0", ["192.168.123.118"]) is None
    assert dds.bind_address("enp4s0", []) is None


def test_no_robot_address_keeps_binding_by_name():
    assert dds.bind_address("enp4s0", ["192.168.1.111", "10.0.0.2"]) is None


def test_an_explicit_address_is_used_as_is():
    assert dds.bind_address("192.168.123.118") == "192.168.123.118"


# --- The diagnosis CHECK and preflight run -----------------------------------
#
# The case these cover: a card that holds only a DHCP lease, because the static
# robot-subnet address was added by hand and did not survive a re-plug. "Sometimes
# on the robot subnet, sometimes not" is then not a DDS question; it is a missing
# address, and the remedy is one nmcli line, which the diagnosis must NAME.


def _diag(addrs, *, nic=True, ownip=None, reach=None):
    ping = False if reach is None else (lambda robot, src: reach)
    return dds.diagnose("enp4s0", "192.168.123.164", addrs=addrs, nic=nic,
                        ownip=ownip, ping=ping)


def test_a_nic_without_a_robot_subnet_address_is_the_first_fault_and_names_the_fix():
    d = _diag(["192.168.1.111"])
    assert d.bound is None and not d.ok
    text = d.explain()
    assert "192.168.1.111" in text and "nmcli con mod" in text
    assert dds.STATIC_ADDR in text


def test_a_nic_that_does_not_exist_is_told_apart_from_one_off_the_subnet():
    d = _diag([], nic=False)
    assert "not a NIC" in d.explain() and "nmcli" in d.explain()


def test_the_bound_address_is_the_robot_subnet_one_even_when_the_nic_is_unambiguous():
    assert _diag(["192.168.123.103"]).bound == "192.168.123.103"
    assert _diag(["192.168.1.111", "192.168.123.118"]).bound == "192.168.123.118"


def test_a_cyclone_trace_on_the_wrong_address_outranks_the_ping():
    d = _diag(["192.168.1.111", "192.168.123.118"], ownip="192.168.1.111", reach=True)
    assert not d.ok and "bound 192.168.1.111" in d.explain()


def test_a_silent_robot_that_does_not_answer_is_the_cable_or_its_power():
    d = _diag(["192.168.123.118"], reach=False)
    assert not d.ok and "does not answer ping" in d.explain()


def test_when_every_precondition_holds_the_answer_points_at_the_robot_not_the_network():
    d = _diag(["192.168.123.118"], ownip="192.168.123.118", reach=True)
    assert d.ok and "network is fine" in d.explain()


def test_ping_is_not_attempted_without_an_address_to_send_from():
    calls = []
    d = dds.diagnose("enp4s0", "192.168.123.164", addrs=["192.168.1.111"], nic=True,
                     ownip=None, ping=lambda r, s: calls.append((r, s)) or True)
    assert calls == [] and d.reach is None


def test_the_trace_is_ignored_when_older_than_the_run(tmp_path, monkeypatch):
    log = tmp_path / "cdds.LOG"
    log.write_text("config: ...\nownip: udp/192.168.1.111\nownip: udp/192.168.123.118\n")
    monkeypatch.setattr(dds, "CDDS_LOG", str(log))
    assert dds.own_ip() == "192.168.123.118", "the LAST line: every process rewrites it"
    assert dds.own_ip(since=log.stat().st_mtime + 60) is None
    monkeypatch.setattr(dds, "CDDS_LOG", str(tmp_path / "absent"))
    assert dds.own_ip() is None
