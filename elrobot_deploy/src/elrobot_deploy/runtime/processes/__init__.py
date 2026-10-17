"""The processes, one module each, spawned by P0 (`runtime/launcher/`) by dotted path and never
imported across each other.

    P0  launcher                      runtime/launcher/
    P6  target (the IK)    ~60 Hz     p6_target.py
    P7  the bus process    250 Hz     p7_bus.py       (owns the serial port)
    P11 the phone source   ~60 Hz     p11_phone.py    (UDP from the iPhone)

The numbers are g1d's, kept so its documentation still maps onto this code; P1, P2-P5 and
P8-P10 either merged into P7 (one half-duplex bus, one owner) or are not ported yet
(`legacy_g1d/`).

Deliberately empty of imports: re-exporting the modules here would put every one of them (and
mujoco) in every child's address space.
"""
