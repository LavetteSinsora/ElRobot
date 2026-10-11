"""Serial bus for ST3215 servos on the Waveshare bus servo adapter.

Single-servo operations (ping, read, write) wait for that servo's status reply. Group
operations: `sync_read` sends one packet and collects one reply per servo; `sync_write`
sends one packet that every listed servo applies at once, with no reply.

The error byte in a status reply reports the servo's current alarm flags (overload,
overheat, ...), not a failed command, so it is recorded in `Bus.alarms` rather than raised.

Some USB-TTL adapters echo transmitted bytes back to RX. If the bytes received start with
the exact packet we sent, they are dropped before parsing.
"""

import time

import serial
from serial.tools import list_ports

from . import protocol, registers
from .registers import Register

DEFAULT_BAUD = 1_000_000

# USB ids of the WCH bridge chips used on Waveshare servo adapters (CH343, CH340).
ADAPTER_USB_IDS = {(0x1A86, 0x55D3), (0x1A86, 0x7523)}


class BusError(Exception):
    pass


def find_adapter_port() -> str:
    ports = [p for p in list_ports.comports() if (p.vid, p.pid) in ADAPTER_USB_IDS]
    if not ports:
        raise BusError("no Waveshare servo adapter found (is it plugged in, jumper on B?)")
    if len(ports) > 1:
        names = ", ".join(p.device for p in ports)
        raise BusError(f"several adapters found ({names}); pass --port")
    # On macOS prefer the call-out device (cu.*) over the dial-in one (tty.*).
    return ports[0].device.replace("/dev/tty.", "/dev/cu.")


class Bus:
    def __init__(self, port: str | None = None, baudrate: int = DEFAULT_BAUD,
                 timeout: float = 0.05, serial_port=None):
        """`serial_port` lets tests pass a fake object with read/write/reset_input_buffer."""
        self.timeout = timeout
        if serial_port is None:
            serial_port = serial.Serial(port or find_adapter_port(), baudrate, timeout=0)
        self.ser = serial_port
        self._rx = bytearray()
        self._echo = b""
        self.alarms: dict[int, int] = {}  # servo id -> error byte of its latest reply

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- transport --------------------------------------------------------------------

    def _send(self, packet: bytes):
        self.ser.reset_input_buffer()
        self._rx.clear()
        self.ser.write(packet)
        self._echo = packet

    def _receive(self, deadline: float) -> protocol.Status | None:
        while True:
            if self._echo and len(self._rx) >= len(self._echo):
                if self._rx.startswith(self._echo):
                    del self._rx[: len(self._echo)]
                self._echo = b""
            if not self._echo:
                status = protocol.parse_status(self._rx)
                if status is not None:
                    return status
            if time.monotonic() >= deadline:
                return None
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self._rx += chunk
            elif self._echo and self._rx and not self._echo.startswith(bytes(self._rx)):
                # What arrived cannot be an echo; stop waiting for one.
                self._echo = b""

    def _transact(self, servo_id: int, packet: bytes, retries: int = 1) -> protocol.Status:
        """Send and wait for the reply, resending on timeout. Resending is safe because every
        packet here is idempotent (reads, or writes of absolute values). The first transaction
        after opening the port has been seen to time out once on the real adapter."""
        for _ in range(retries + 1):
            self._send(packet)
            deadline = time.monotonic() + self.timeout
            while (status := self._receive(deadline)) is not None:
                if status.servo_id == servo_id:
                    self.alarms[servo_id] = status.error
                    return status
        raise BusError(f"servo {servo_id}: no reply")

    # -- single servo -----------------------------------------------------------------

    def ping(self, servo_id: int) -> bool:
        try:
            self._transact(servo_id, protocol.ping(servo_id))
            return True
        except BusError:
            return False

    def read(self, servo_id: int, addr: int, length: int) -> bytes:
        status = self._transact(servo_id, protocol.read(servo_id, addr, length))
        if len(status.params) != length:
            raise BusError(f"servo {servo_id}: expected {length} bytes, got {len(status.params)}")
        return status.params

    def write(self, servo_id: int, addr: int, data: bytes):
        self._transact(servo_id, protocol.write(servo_id, addr, data))

    def read_reg(self, servo_id: int, reg: Register) -> int:
        return reg.decode(self.read(servo_id, reg.addr, reg.size))

    def write_reg(self, servo_id: int, reg: Register, value: int):
        self.write(servo_id, reg.addr, reg.encode(value))

    def write_eeprom_reg(self, servo_id: int, reg: Register, value: int):
        """Unlock, write, lock, then read back. Without the unlock the value is lost at power-off."""
        if not reg.eeprom:
            raise ValueError(f"{reg.name} is not an EEPROM register")
        self.write_reg(servo_id, registers.LOCK, 0)
        try:
            self.write_reg(servo_id, reg, value)
        finally:
            self.write_reg(servo_id, registers.LOCK, 1)
        if (got := self.read_reg(servo_id, reg)) != value:
            raise BusError(f"servo {servo_id}: wrote {reg.name}={value}, read back {got}")

    def read_table(self, servo_id: int) -> dict[str, int]:
        """Read the whole control table in one packet and decode it."""
        return registers.decode_block(0, self.read(servo_id, 0, registers.TABLE_END))

    # -- group ------------------------------------------------------------------------

    def sync_read(self, ids: list[int], addr: int, length: int) -> dict[int, bytes]:
        """One request, one reply per servo. Raises if any servo does not answer."""
        self._send(protocol.sync_read(ids, addr, length))
        # Each reply is 6 + length bytes (~10 us/byte at 1 Mbps) plus the servo's return delay.
        deadline = time.monotonic() + self.timeout + len(ids) * 0.001
        pending = set(ids)
        out = {}
        while pending:
            status = self._receive(deadline)
            if status is None:
                raise BusError(f"sync_read: no reply from {sorted(pending)}")
            if status.servo_id in pending and len(status.params) == length:
                self.alarms[status.servo_id] = status.error
                out[status.servo_id] = status.params
                pending.discard(status.servo_id)
        return out

    def sync_read_reg(self, ids: list[int], reg: Register) -> dict[int, int]:
        return {i: reg.decode(d) for i, d in self.sync_read(ids, reg.addr, reg.size).items()}

    def sync_write(self, addr: int, length: int, data_by_id: dict[int, bytes]):
        self._send(protocol.sync_write(addr, length, data_by_id))

    def sync_write_reg(self, reg: Register, values: dict[int, int]):
        self.sync_write(reg.addr, reg.size, {i: reg.encode(v) for i, v in values.items()})
