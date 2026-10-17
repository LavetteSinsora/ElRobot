"""The `rt/lowcmd` CRC, at C speed, with no native library.

The G1's firmware ignores a `unitree_hg LowCmd_` whose `crc` word is wrong.
unitree_sdk2py computed it (`utils/crc.py`, `CRC().Crc(msg)`): pack the message
into the SDK's C-struct layout (`'<2B2x' + 'B3x5fI' * 35 + '5I'`, 1004 bytes,
4-byte aligned, little-endian), read it back as uint32 words, drop the last
(the `crc` word itself), and run CRC-32/MPEG-2 over them (poly 0x04C11DB7,
init 0xFFFFFFFF, unreflected, no final xor) — through a Linux-only `.so`,
or a pure-Python bit loop costing ~1 ms per message everywhere else.

Here the packing is the SDK's, verbatim, and the CRC is zlib's C
implementation through a reflected-CRC identity:

    mpeg2(msg) = bitrev32( zlib.crc32(bitrev_bytes(msg)) ^ 0xFFFFFFFF )

with the byte-wise bit reversal done by `bytes.translate` (also C). Verified
exact against the SDK's own `CRC().Crc` on random messages
(`tests/test_wire_idl.py`, wherever the SDK is installed); ~5 us a message.

Used by `runtime/drivers/arm_command.py` on every `rt/lowcmd` publish.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

_REV = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))

#: unitree_sdk2py `utils/crc.py` `__packFmtHGLowCmd`: size 1004.
HG_LOWCMD_FMT = "<2B2x" + "B3x5fI" * 35 + "5I"
_HG_LOWCMD = struct.Struct(HG_LOWCMD_FMT)


def _bitrev32(x: int) -> int:
    x = ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555)
    x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333)
    x = ((x & 0x0F0F0F0F) << 4) | ((x >> 4) & 0x0F0F0F0F)
    return ((x << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | (x >> 24)) & 0xFFFFFFFF


def crc32_mpeg2_words(words) -> int:
    """CRC-32/MPEG-2 over a sequence of uint32 words, MSB-first per word —
    bit-identical to unitree_sdk2py's `CRC._crc_py` / its native `crc32_core`."""
    raw = np.asarray(words, dtype=">u4").tobytes()
    return _bitrev32(zlib.crc32(raw.translate(_REV)) ^ 0xFFFFFFFF)


def hg_lowcmd_crc(msg) -> int:
    """The `crc` word for a `unitree_hg LowCmd_` (`idl/unitree_hg.py`) —
    what unitree_sdk2py's `CRC().Crc(msg)` returns for it."""
    data = [msg.mode_pr, msg.mode_machine]
    for m in msg.motor_cmd:
        data += (m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve)
    data += list(msg.reserve)
    data.append(msg.crc)
    packed = _HG_LOWCMD.pack(*data)
    # The SDK's `__Trans`: little-endian words, all but the last.
    words = np.frombuffer(packed, dtype="<u4")[:-1]
    return crc32_mpeg2_words(words)


__all__ = ["HG_LOWCMD_FMT", "crc32_mpeg2_words", "hg_lowcmd_crc"]
