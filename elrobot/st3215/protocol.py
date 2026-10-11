"""Feetech STS packet format (ST3215).

Instruction packet:  FF FF  id  len  instr  params...  checksum
Status packet:       FF FF  id  len  error  params...  checksum

len = number of params + 2, checksum = ~(id + len + instr/error + sum(params)) & 0xFF.
"""

from dataclasses import dataclass

HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE

PING = 0x01
READ = 0x02
WRITE = 0x03
SYNC_READ = 0x82
SYNC_WRITE = 0x83


class ProtocolError(Exception):
    pass


def checksum(body: bytes) -> int:
    return ~sum(body) & 0xFF


def instruction(servo_id: int, instr: int, params: bytes = b"") -> bytes:
    if not 0 <= servo_id <= BROADCAST_ID:
        raise ValueError(f"bad servo id {servo_id}")
    body = bytes([servo_id, len(params) + 2, instr]) + params
    return HEADER + body + bytes([checksum(body)])


def ping(servo_id: int) -> bytes:
    return instruction(servo_id, PING)


def read(servo_id: int, addr: int, length: int) -> bytes:
    return instruction(servo_id, READ, bytes([addr, length]))


def write(servo_id: int, addr: int, data: bytes) -> bytes:
    return instruction(servo_id, WRITE, bytes([addr]) + data)


def sync_read(ids: list[int], addr: int, length: int) -> bytes:
    return instruction(BROADCAST_ID, SYNC_READ, bytes([addr, length, *ids]))


def sync_write(addr: int, length: int, data_by_id: dict[int, bytes]) -> bytes:
    params = bytearray([addr, length])
    for servo_id, data in data_by_id.items():
        if len(data) != length:
            raise ValueError(f"id {servo_id}: {len(data)} bytes, expected {length}")
        params += bytes([servo_id]) + data
    return instruction(BROADCAST_ID, SYNC_WRITE, bytes(params))


@dataclass
class Status:
    servo_id: int
    error: int
    params: bytes


def parse_status(buf: bytearray) -> Status | None:
    """Pop the first complete, valid status packet from `buf`.

    Leading garbage and packets with bad checksums are discarded. Returns None (leaving any
    partial packet in `buf`) when no complete packet is available yet.
    """
    while True:
        start = buf.find(HEADER)
        if start < 0:
            # Keep a trailing 0xFF: it may be the first half of the next header.
            del buf[: len(buf) - (1 if buf[-1:] == b"\xff" else 0)]
            return None
        # Skip extra 0xFF padding before the id (the id itself is never 0xFF).
        while start + 2 < len(buf) and buf[start + 2] == 0xFF:
            start += 1
        del buf[:start]
        if len(buf) < 4:
            return None
        length = buf[3]
        if length < 2:
            del buf[:2]
            continue
        total = 4 + length
        if len(buf) < total:
            return None
        packet = bytes(buf[:total])
        if checksum(packet[2:-1]) != packet[-1]:
            del buf[:2]
            continue
        del buf[:total]
        return Status(servo_id=packet[2], error=packet[4], params=packet[5:-1])
