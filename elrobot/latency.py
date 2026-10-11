"""Bus latency benchmark. Moves nothing.

    uv run python -m elrobot.latency            # servos 1-8, 500 iterations per test
    uv run python -m elrobot.latency -n 2000

Reads only, except one optional write test: it sends TORQUE_ENABLE = 0 as a sync write, and
only when every servo already reports torque off. That write changes nothing.
"""

import argparse
import statistics
import sys
import time

import serial

from .st3215 import registers as R
from .st3215.bus import Bus, BusError

BIT_US = 10.0  # 8N1 at 1 Mbps: 10 bits per byte -> 10 us per byte


def stats(name: str, samples: list[float]):
    s = sorted(samples)
    pct = lambda p: s[min(len(s) - 1, int(p * len(s)))] * 1e3
    med = statistics.median(s)
    print(f"  {name:<38} median {med * 1e3:6.2f} ms  p95 {pct(.95):6.2f}  p99 {pct(.99):6.2f}  "
          f"max {s[-1] * 1e3:6.2f}   -> {1 / med:5.0f} Hz")


def timed(fn, n: int) -> list[float]:
    for _ in range(20):  # warm-up
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--ids", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("-n", type=int, default=500)
    a = ap.parse_args(argv)
    ids, n = a.ids, a.n

    try:
        with Bus(a.port) as bus:
            missing = [i for i in ids if not bus.ping(i)]
            if missing:
                print(f"no reply from {missing}; fix the bus or pass --ids", file=sys.stderr)
                return 1
            delays = bus.sync_read_reg(ids, R.RETURN_DELAY)
            torque = bus.sync_read_reg(ids, R.TORQUE_ENABLE)
            print(f"port {bus.ser.port} @ {bus.ser.baudrate} bps, servos {ids}")
            print("return delay (us): " + ", ".join(f"{i}:{2 * d}" for i, d in delays.items()))

            # Theoretical wire time for one sync read of present_position (2 bytes per servo).
            req = 8 + len(ids)                       # FF FF FE len 82 addr n ids.. chk
            rep = len(ids) * (6 + 2)                 # FF FF id len err p p chk, per servo
            wire = (req + rep) * BIT_US + sum(2 * d for d in delays.values())
            print(f"wire time for one group position read: {wire / 1e3:.2f} ms "
                  f"({req} B request + {rep} B replies + return delays); the rest is USB + OS\n")

            pos = lambda: bus.sync_read_reg(ids, R.PRESENT_POSITION)
            stats("sync_read position (all servos)", timed(pos, n))
            stats("sync_read pos+speed+load (6 B each)", timed(lambda: bus.sync_read(ids, 56, 6), n))
            stats("read position, one servo at a time", timed(lambda: [bus.read_reg(i, R.PRESENT_POSITION) for i in ids], n // 4))
            stats("ping one servo (single round trip)", timed(lambda: bus.ping(ids[0]), n))

            if any(torque.values()):
                print("\n  write tests skipped: torque is ON for "
                      f"{[i for i, t in torque.items() if t]} (power-cycle to release, then rerun)")
                return 0

            zero = {i: 0 for i in ids}

            def write():
                bus.sync_write_reg(R.TORQUE_ENABLE, zero)
                bus.ser.flush()  # block until the bytes have left the host
            stats("sync_write (no reply), until sent", timed(write, n))
            stats("control cycle: sync_read + sync_write", timed(lambda: (pos(), write()), n))
            print("\n(write packet for goal positions is 8 + 3*N bytes = "
                  f"{(8 + 3 * len(ids)) * BIT_US / 1e3:.2f} ms on the wire for {len(ids)} servos)")
            return 0
    except (BusError, serial.SerialException) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
