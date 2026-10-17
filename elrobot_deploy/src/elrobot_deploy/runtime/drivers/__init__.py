"""The servo bus: `st3215_driver` (the real ST3215 bus, through the repo's `elrobot` package,
and `SimDriver`, a simulation of it with the same interface). A driver holds no scheduling
logic and no rings: the bus process decides *when*, a driver only *how to say it*."""
