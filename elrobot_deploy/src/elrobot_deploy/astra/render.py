"""The images Astra is shown: strip A ("now") and sheet H2 (an action's history).

    strip A   the cameras side by side in `[astra] cameras` order (exo | wrist), each view 480
              px tall, under a 32 px label bar (`now  tick=...  t=...`): exactly 512 px, the
              height of one image tile (taller spills into a second row and nearly doubles the
              cost). A fisheye view is 480 x 480, a 16:9 view 853 x 480.
    sheet H2  rows = cameras, columns = ticks (an action's start, 1/3 and 2/3; or the ticks an
              `inspect_observation` asked for, up to 4). Every column 341 px wide, every view
              keeping its own aspect (fisheye 341 x 341, 16:9 341 x 192), a label bar per
              column with its tick and `t`.

A fisheye view is the circle of the recorded lens square (P9 records the square), outside it
black, no undistortion. Only the tick / `t` labels are drawn into pixels: every number Astra
needs is in the text.
"""

from __future__ import annotations

import numpy as np

VIEW_H = 480
BAR_H = 32
COL_W = 341
GAP = 4
#: Background between views and of a missing frame.
GRAY = (40, 40, 40)
_MASKS: dict = {}


def decode(jpg: bytes | None) -> np.ndarray | None:
    import cv2
    if not jpg:
        return None
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def encode(img: np.ndarray, quality: int = 90) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _circle_mask(side: int) -> np.ndarray:
    m = _MASKS.get(side)
    if m is None:
        yy, xx = np.mgrid[:side, :side]
        c = (side - 1) / 2
        m = ((xx - c) ** 2 + (yy - c) ** 2) <= (side / 2) ** 2
        _MASKS[side] = m
    return m


def view_size(kind: str, *, height: int | None = None, width: int | None = None,
              aspect: float = 16 / 9) -> tuple[int, int]:
    """(width, height) of one camera's view: a fisheye is square; a normal view keeps
    `aspect` (its frame's width / height)."""
    a = 1.0 if kind == "fisheye" else aspect
    if height is not None:
        return round(height * a), int(height)
    return int(width), round(width / a)


def view(img: np.ndarray | None, kind: str, size: tuple[int, int]) -> np.ndarray:
    """One camera's view at `size` (w, h): resized, a fisheye masked to its circle. A missing
    frame is a gray panel that says so."""
    import cv2
    w, h = size
    if img is None:
        out = np.full((h, w, 3), GRAY, np.uint8)
        cv2.putText(out, "no frame", (max(4, w // 2 - 50), h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 1, cv2.LINE_AA)
        return out
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if kind == "fisheye":
        out[~_circle_mask(w)] = 0
    return out


def bar(width: int, text: str) -> np.ndarray:
    import cv2
    out = np.full((BAR_H, width, 3), (24, 24, 24), np.uint8)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1,
                cv2.LINE_AA)
    return out


def label(tick: int, t: float, prefix: str = "") -> str:
    return f"{prefix}tick={tick}  t={t:.2f} s".strip()


def strip(frames: list, kinds: list, text: str, aspects: list | None = None) -> np.ndarray:
    """Strip A: `frames[i]` (BGR or None) of a camera of kind `kinds[i]`, left to right."""
    aspects = aspects or [16 / 9] * len(frames)
    views = [view(f, k, view_size(k, height=VIEW_H, aspect=a))
             for f, k, a in zip(frames, kinds, aspects)]
    gap = np.full((VIEW_H, GAP, 3), GRAY, np.uint8)
    row = views[0]
    for v in views[1:]:
        row = np.hstack([row, gap, v])
    return np.vstack([bar(row.shape[1], text), row])


def sheet(columns: list, kinds: list, aspects: list | None = None) -> np.ndarray:
    """Sheet H2: `columns` = [(text, [frame per camera])], one column each, rows = cameras."""
    aspects = aspects or [16 / 9] * len(kinds)
    sizes = [view_size(k, width=COL_W, aspect=a) for k, a in zip(kinds, aspects)]
    cols = []
    for text, frames in columns:
        parts = [bar(COL_W, text)]
        for f, k, s in zip(frames, kinds, sizes):
            parts.append(np.full((GAP, COL_W, 3), GRAY, np.uint8))
            parts.append(view(f, k, s))
        cols.append(np.vstack(parts))
    h = cols[0].shape[0]
    gap = np.full((h, GAP, 3), GRAY, np.uint8)
    out = cols[0]
    for c in cols[1:]:
        out = np.hstack([out, gap, c])
    return out


__all__ = ["BAR_H", "COL_W", "VIEW_H", "bar", "decode", "encode", "label", "sheet", "strip",
           "view", "view_size"]
