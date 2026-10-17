"""A 500 Hz smoother and torque feedforward of your own.

    [control.servo]
    filter_hz = 6.0                              # > 0: the smoother is on
    torque_ff = 0.3                              # > 0: the feedforward is on
    smoother = "servo.py:make_smoother"
    feedforward = "servo.py:make_feedforward"

    elrobot_deploy check-policy examples/custom_modules/deploy.toml

P7 walks the line through P6's joint targets and, with a smoother on, hands it
the line read `lead_s` ahead; the smoother returns the command's position,
velocity and acceleration, and the feedforward turns velocity and acceleration
into a torque added to gravity compensation
(`docs/infra_logic/500hz_execution.md`, "Writing your own smoother or
feedforward").

`SecondOrderLowPass` is a critically damped second-order filter written as
plainly as possible (semi-implicit Euler, split into substeps short enough to
be stable at any cutoff); `ViscousFeedforward` is `scale * b * v` on the arm joints with one
damping coefficient. Neither is better than the shipped ones — they are the
shape of the two slots.

Both run inside P7's 2 ms period, at 500 Hz, during homing and the exit ramp
too: numpy only, no allocation to speak of, no IK imports.
"""

from __future__ import annotations

import numpy as np

ARM = 14                     # the 17 joints are 14 arm (left, right) + 3 waist


class SecondOrderLowPass:
    def __init__(self, fc: float, q0):
        self.w = 2.0 * np.pi * float(fc)
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.v = np.zeros_like(self.q)
        self.lead_s = 2.0 / self.w          # its lag behind a ramp: read that far ahead

    def step(self, reference, h):
        n = max(1, int(np.ceil(float(h) * self.w / 0.5)))   # w*dt <= 0.5: stable
        dt = float(h) / n
        for _ in range(n):
            a = self.w * self.w * (reference - self.q) - 2.0 * self.w * self.v
            self.v = self.v + a * dt
            self.q = self.q + self.v * dt
        return self.q, self.v, a


class ViscousFeedforward:
    def __init__(self, scale: float, b: float = 0.3, tau_max: float = 2.0):
        self.gain = float(scale) * b
        self.tau_max = tau_max
        self.out = np.zeros(17)

    def tau(self, v, a):
        self.out[:ARM] = np.clip(self.gain * v[:ARM], -self.tau_max, self.tau_max)
        return self.out


def make_smoother(servo, hw, q0):
    """`servo` is the rollout's ServoShaping: its `filter_hz` is our cutoff."""
    return SecondOrderLowPass(servo.filter_hz, q0)


def make_feedforward(servo, hw):
    """`servo.torque_ff` is our scale."""
    return ViscousFeedforward(servo.torque_ff)
