"""From joint targets to what the servos are sent, inside the bus process: the segment walk
between ticks (`servo_segment`), the critically damped command filter and its look-ahead
(`servo_filter`), and one tick of all of it plus the clamps (`bus_servo`)."""
