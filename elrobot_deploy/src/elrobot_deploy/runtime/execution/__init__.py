"""From joint targets to what the motors are sent: blending overlapping chunks
into one action per tick (`chunk_blend`), the critically damped servo reference
and its feedforward (`servo_filter`), and interpolation between ticks at 500 Hz
(`servo_segment`). `chunk_blend` runs in P6, the other two in P7 (and P8).
Design: `docs/infra_logic/chunk_blending.md`, `docs/infra_logic/500hz_execution.md`."""
