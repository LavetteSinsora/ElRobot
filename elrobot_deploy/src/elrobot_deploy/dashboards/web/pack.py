"""Binary framing between a dashboard's Python and its browser.

A section per array, concatenated, each starting on a 4-byte boundary. Not
cosmetic: the browser builds its views with `new Float32Array(buffer,
byteOffset, length)`, which THROWS on a byteOffset that is not a multiple of 4.
Validity bytes are uint8 and a rollout's frame count is not generally a
multiple of four, so any float section packed after one lands on a live
grenade — and the failure is at load time, on some rollouts and not others,
with the length as the only clue.

The browser is told where each array starts (`spans`) rather than deriving it,
so a buffer gaining a section here cannot silently shift one there. `pack` and
`spans` take the same list and apply the same padding rule, so the map and the
bytes cannot come to disagree about where anything is.

Used by both dashboards' servers; the browser side is `dashboards/web/static/gl.js`.
"""

from __future__ import annotations

import numpy as np

ALIGN = 4


def pack(parts: list[tuple[str, np.ndarray]]) -> bytes:
    """The sections concatenated, aligned. Pairs with `spans`."""
    out = bytearray()
    for _, arr in parts:
        out += b"\0" * (-len(out) % ALIGN)
        out += arr.tobytes()
    return bytes(out)


def spans(parts: list[tuple[str, np.ndarray]]) -> dict[str, list[int]]:
    """`[(name, array), ...]` -> `{name: [offset, nbytes]}`, packed in order."""
    out, offset = {}, 0
    for name, arr in parts:
        offset += -offset % ALIGN
        out[name] = [offset, int(arr.nbytes)]
        offset += int(arr.nbytes)
    return out


__all__ = ["ALIGN", "pack", "spans"]
