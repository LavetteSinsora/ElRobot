"""A simulated ST3215 bus that behaves like a pyserial port."""

from elrobot.st3215 import protocol
from elrobot.st3215 import registers as R


class SimClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeServo:
    """Without `stops` the servo never moves. With stops (unwrapped encoder steps, may lie
    outside 0..4095 to model a range that crosses the wrap) it follows GOAL_POSITION at
    GOAL_SPEED in reported coordinates, like the real position controller."""

    def __init__(self, servo_id: int, position: int = 2048, stops=None, clock=None,
                 offset_sign: int = 1, friction=None, give=None, wrong_way=False, runaway=0):
        self.mem = bytearray(R.TABLE_END)
        self.alarm = 0
        self.stops, self.clock, self.offset_sign = stops, clock, offset_sign
        # (u_lo, u_hi, torque_needed): a sticky stretch the servo only passes with enough torque
        self.friction = friction
        # (steps, torque): the upper stop yields this far when pushed with at least this torque,
        # like plastic squeezing at the gripper's closed stop
        self.give = give
        self.wrong_way = wrong_way  # drives away from the goal (seen on the real gripper)
        self.runaway = runaway      # +1/-1: controller a turn off, drives this way whatever the goal
        self.max_u = float(position)
        self.u = float(position)
        self.last_t = clock.now() if clock else 0.0
        for reg, value in [(R.MODEL, R.STS3215_MODEL), (R.ID, servo_id), (R.FIRMWARE_MAJOR, 3),
                           (R.FIRMWARE_MINOR, 10), (R.MAX_POSITION_LIMIT, 4095), (R.LOCK, 1),
                           (R.PRESENT_VOLTAGE, 121), (R.PRESENT_TEMPERATURE, 29),
                           (R.PRESENT_POSITION, position)]:
            self.set(reg, value)

    def reported(self):
        return (self.u - self.offset_sign * self.get(R.OFFSET)) % R.STEPS_PER_TURN

    def step(self):
        if self.clock is None:
            return
        now = self.clock.now()
        dt, self.last_t = now - self.last_t, now
        load = 0
        if self.stops and self.get(R.TORQUE_ENABLE):
            r, goal = self.reported(), self.get(R.GOAL_POSITION)
            move = max(-self.get(R.GOAL_SPEED) * dt, min(self.get(R.GOAL_SPEED) * dt, goal - r))
            if self.wrong_way:
                move = -move
            if self.runaway:
                move = self.runaway * self.get(R.GOAL_SPEED) * dt
            lo, hi = self.stops
            if self.give and self.get(R.TORQUE_LIMIT) >= self.give[1]:
                hi += self.give[0]
            if self.friction and move:
                f_lo, f_hi, needed = self.friction
                if f_lo <= self.u + move <= f_hi and self.get(R.TORQUE_LIMIT) < needed:
                    move = 0
                    load = self.get(R.TORQUE_LIMIT)
            new_u = min(hi, max(lo, self.u + move))
            if new_u != self.u + move:
                load = self.get(R.TORQUE_LIMIT)
            self.u = new_u
            self.max_u = max(self.max_u, new_u)
        self.set(R.PRESENT_POSITION, round(self.reported()) % R.STEPS_PER_TURN)
        self.set(R.PRESENT_LOAD, load)

    def set(self, reg, value):
        self.mem[reg.addr : reg.addr + reg.size] = reg.encode(value)

    def get(self, reg):
        return reg.decode(self.mem[reg.addr : reg.addr + reg.size])

    def status(self, params: bytes = b"") -> bytes:
        body = bytes([self.get(R.ID), len(params) + 2, self.alarm]) + params
        return protocol.HEADER + body + bytes([protocol.checksum(body)])


class FakeSerial:
    def __init__(self, servos: list[FakeServo], echo: bool = False):
        self.servos = {s.get(R.ID): s for s in servos}
        self.echo = echo
        self.rx = bytearray()
        self.sent: list[bytes] = []

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, n=1):
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def reset_input_buffer(self):
        self.rx.clear()

    def close(self):
        pass

    def write(self, packet: bytes):
        for s in self.servos.values():
            s.step()
        self.sent.append(packet)
        if self.echo:
            self.rx += packet
        sid, instr, params = packet[2], packet[4], packet[5:-1]
        if instr == protocol.SYNC_READ:
            addr, length, ids = params[0], params[1], params[2:]
            for i in ids:
                if s := self.servos.get(i):
                    self.rx += s.status(bytes(s.mem[addr : addr + length]))
            return
        if instr == protocol.SYNC_WRITE:
            addr, length = params[0], params[1]
            rest = params[2:]
            for k in range(0, len(rest), length + 1):
                if s := self.servos.get(rest[k]):
                    s.mem[addr : addr + length] = rest[k + 1 : k + 1 + length]
            return
        s = self.servos.get(sid)
        if s is None:
            return
        if instr == protocol.PING:
            self.rx += s.status()
        elif instr == protocol.READ:
            addr, length = params
            self.rx += s.status(bytes(s.mem[addr : addr + length]))
        elif instr == protocol.WRITE:
            addr = params[0]
            s.mem[addr : addr + len(params) - 1] = params[1:]
            s.step()
            self.rx += s.status()
