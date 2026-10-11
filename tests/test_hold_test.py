from elrobot import hold_test as H
from elrobot.calibration import JointCal
from elrobot.st3215 import registers as R
from elrobot.st3215.bus import Bus

from fake_bus import FakeSerial, FakeServo, SimClock


def test_cases_run_and_pid_is_restored():
    clock = SimClock()
    servos = [FakeServo(j, position=2048, stops=(0, 4095), clock=clock) for j in range(1, 9)]
    for s in servos:
        s.set(R.P_COEF, 32); s.set(R.D_COEF, 32)
    ser = FakeSerial(servos)
    cals = {j: JointCal(j, f"j{j}", 1, 2048, 0, 900, 3200, "t") for j in range(1, 9)}
    t = H.HoldTest(Bus(serial_port=ser, timeout=0.01), cals, [3, 4], clock=clock.now, sleep=clock.sleep)
    t.set_pid({3: (48, 0, 32), 4: (32, 8, 32)})
    assert ser.servos[3].get(R.P_COEF) == 48 and ser.servos[4].get(R.I_COEF) == 8
    res = t.hold(ours=True)
    assert abs(res[3]["mean_err_deg"]) < 0.1   # the fake does not sag
    t.restore()
    assert [ser.servos[3].get(r) for r in (R.P_COEF, R.I_COEF, R.D_COEF)] == [32, 0, 32]
    assert [ser.servos[4].get(r) for r in (R.P_COEF, R.I_COEF, R.D_COEF)] == [32, 0, 32]
    assert all(s.get(R.TORQUE_ENABLE) == 0 for s in ser.servos.values())
