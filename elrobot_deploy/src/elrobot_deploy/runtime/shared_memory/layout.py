"""The arena: one mmap, statically divided, validated by a layout hash.

No allocator, no pointers (virtual addresses differ per process), no
Python objects. Every process computes the layout from the same config and
refuses to attach if the hash disagrees — which turns "P6 and P7 were built from
different configs" from a wrong joint target into a startup error.

P0 creates it (`Arena.create`), every other process attaches (`Arena.attach`).
Design: `docs/infra_logic/arena_layout_and_reads.md`, "The regions" and "Size,
and `/dev/shm`". Adding or resizing a region changes the hash; changing what a
region MEANS needs a `schema.VERSION` bump as well.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory

import numpy as np

from elrobot_deploy.runtime.shared_memory import schema

ALIGN = 64          # one cache line, so no two regions share one

# Names this process created. Attaching to one of them is the same-process case
# (a test, or P0 opening its own arena), where the resource-tracker workaround
# below must not run: unregistering a name the tracker is still tracking for the
# creator makes it raise on the eventual unlink.
_CREATED: set[str] = set()


@dataclass(frozen=True)
class Region:
    name: str
    dtype: np.dtype
    count: int
    offset: int

    @property
    def nbytes(self) -> int:
        return self.dtype.itemsize * self.count


@dataclass(frozen=True)
class Layout:
    regions: dict[str, Region]
    size: int
    hash: int
    cameras: tuple[str, ...]

    def region(self, name: str) -> Region:
        return self.regions[name]

    def describe(self) -> str:
        lines = [f"arena {self.size/1e6:.2f} MB  hash={self.hash:#018x}"]
        for r in self.regions.values():
            lines.append(f"  {r.name:<24} {r.count:>6} x {r.dtype.itemsize:>9} B "
                         f"= {r.nbytes/1e6:8.3f} MB  @ {r.offset}")
        return "\n".join(lines)


def build(cfg, derived) -> Layout:
    """Pure: config in, region table out. Called identically by every process."""
    regions: dict[str, Region] = {}
    offset = 0

    def add(name: str, dtype: np.dtype, count: int) -> None:
        nonlocal offset
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        regions[name] = Region(name, np.dtype(dtype), count, offset)
        offset += np.dtype(dtype).itemsize * count

    add("header", schema.HEADER, 1)
    add("control", schema.CONTROL, 1)

    add("state_seq", np.uint64, derived.state_slots)
    add("state_rec", schema.STATE, derived.state_slots)

    add("chunk_seq", np.uint64, derived.chunk_slots)
    add("chunk_rec", schema.chunk_dtype(cfg.H), derived.chunk_slots)

    add("target_seq", np.uint64, derived.n_target)
    add("target_rec", schema.TARGET, derived.n_target)

    # The operator's stream. Allocated unconditionally, whether or not
    # the run was launched with `--intervene`: the layout hash is what every
    # process checks before it attaches, so it must not depend on a flag one
    # process was given and another was not. An unwritten region costs address
    # space and nothing else.
    add("intervene_seq", np.uint64, derived.intervene_slots)
    add("intervene_rec", schema.INTERVENE, derived.intervene_slots)
    add("intervene_meta", schema.INTERVENE_META, 1)

    # P7's publish log, read by P9 only.
    add("command_seq", np.uint64, derived.command_slots)
    add("command_rec", schema.COMMAND, derived.command_slots)

    # One RING per camera VIEW, and the rings are NOT uniform:
    # a fisheye wrist is 720x1280, each eye of the split head 480x640, and the
    # depth of each is the history the policy declared for it. Three regions:
    #
    #   cam_<v>_head  (1,) uint64   frame_id of the newest PUBLISHED frame
    #   cam_<v>_seq   (N,) uint64   per-slot sequence counter, odd = writing
    #   cam_<v>_rec   (N,) record   schema.camera_dtype — metadata + pixels
    #
    # The read rules are `rings.CameraRing`'s and
    # `docs/infra_logic/arena_layout_and_reads.md`'s ("Read rules").
    views = cfg.views
    for name in sorted(views):
        v = views[name]
        n = int(getattr(derived, "camera_slots", {}).get(name, v.slots))
        add(f"cam_{name}_head", np.uint64, 1)
        add(f"cam_{name}_seq", np.uint64, n)
        add(f"cam_{name}_rec",
            schema.camera_dtype(v.height, v.width, v.jpeg_capacity), n)

    digest = hashlib.blake2b(digest_size=8)
    for r in regions.values():
        digest.update(f"{r.name}|{r.dtype.str}|{r.dtype.itemsize}|"
                      f"{r.count}|{r.offset}".encode())
    layout_hash = int.from_bytes(digest.digest(), "little")

    return Layout(regions=regions, size=offset, hash=layout_hash,
                  cameras=tuple(sorted(views)))


class LayoutMismatch(RuntimeError):
    """Attached to an arena built from a different config."""


#: Left free in /dev/shm beyond the arena itself: other tenants of the same
#: tmpfs (a DataLoader's worker queues, a previous run's leaked segment) must
#: not be what turns into this run's SIGBUS.
SHM_HEADROOM_BYTES = 64 << 20


class ShmTooSmall(RuntimeError):
    """/dev/shm cannot hold the arena. Raised before it is created."""


def shm_free(path: str = "/dev/shm") -> tuple[int, int]:
    """`(free, total)` bytes of the tmpfs at `path`."""
    import os
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize


def check_shm(layout: Layout, *, path: str = "/dev/shm", free=None) -> str:
    """Refuse, with the numbers, an arena /dev/shm cannot hold.

    **This has to be asked before the arena exists, and it has to be asked at
    all.** `SharedMemory(create=True)` is an `ftruncate` on a tmpfs file, which
    reserves nothing: pages are allocated when first written. So an arena
    bigger than /dev/shm's free space is created without complaint, and the
    process that first touches a page past the limit dies of SIGBUS — a camera
    process some seconds into CHECK, or P9 on the frame that finally wraps a
    4 s ring. Nothing grows after this point (the arena is never resized), so
    this one comparison is the whole guarantee.

    `free` is injectable for tests: a callable returning `(free, total)`.
    Returns the line P0 prints when it passes.
    """
    avail, total = (free or (lambda: shm_free(path)))()
    need = layout.size + SHM_HEADROOM_BYTES
    cams = sorted(((r.name[len("cam_"):-len("_rec")], r)
                   for r in layout.regions.values()
                   if r.name.startswith("cam_") and r.name.endswith("_rec")),
                  key=lambda nr: -nr[1].nbytes)
    detail = ", ".join(f"{n} {r.nbytes / 1e6:.0f} MB ({r.count} slots)"
                       for n, r in cams)
    line = (f"arena {layout.size / 1e6:.1f} MB [{detail}]; {path} has "
            f"{avail / 1e6:.0f} MB free of {total / 1e6:.0f} MB")
    if avail < need:
        raise ShmTooSmall(
            f"{line}: not enough (need the arena plus "
            f"{SHM_HEADROOM_BYTES >> 20} MB headroom = {need / 1e6:.0f} MB). "
            f"Camera rings are sized from the policy's declared `history` "
            f"(N = ceil(s x fps x 1.25) + 3 slots per camera): shorten it, "
            f"or grow the tmpfs — `sudo mount -o remount,size=<N>G {path}`.")
    return line


class Arena:
    """A mapped arena plus typed views onto its regions."""

    def __init__(self, shm: shared_memory.SharedMemory, layout: Layout, *,
                 owner: bool):
        self.shm = shm
        self.layout = layout
        self.owner = owner
        self._views: dict[str, np.ndarray] = {}

    # --- lifecycle ---------------------------------------------------------

    @classmethod
    def create(cls, name: str, layout: Layout) -> "Arena":
        shm = shared_memory.SharedMemory(name=name, create=True, size=layout.size)
        _CREATED.add(name)
        arena = cls(shm, layout, owner=True)
        arena.view("header")[0] = (schema.MAGIC, schema.VERSION, layout.hash)
        return arena

    @classmethod
    def attach(cls, name: str, layout: Layout) -> "Arena":
        shm = shared_memory.SharedMemory(name=name, create=False)
        # A child that attaches also registers with the resource tracker, which
        # then unlinks the segment when THAT child exits and warns about a leak
        # for every other one. The creator owns the lifetime; children do not.
        if name not in _CREATED:
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass
        arena = cls(shm, layout, owner=False)
        hdr = arena.view("header")[0]
        if int(hdr["magic"]) != schema.MAGIC:
            raise LayoutMismatch(f"arena {name!r} is not a elrobot_deploy arena")
        if int(hdr["version"]) != schema.VERSION:
            # The hash compares sizes and offsets; this compares MEANINGS. A
            # reader built against another schema version would read a ring by
            # the wrong rules (`docs/infra_logic/arena_layout_and_reads.md`,
            # "What is stable").
            raise LayoutMismatch(
                f"arena {name!r} is schema version {int(hdr['version'])}, this "
                f"code reads version {schema.VERSION}. Every process must run "
                f"the same elrobot_deploy.")
        if int(hdr["layout_hash"]) != layout.hash:
            raise LayoutMismatch(
                f"arena {name!r} was built from a different config: "
                f"header {int(hdr['layout_hash']):#018x} != {layout.hash:#018x}. "
                f"Every process must compute the same layout.")
        return arena

    def close(self) -> None:
        self._views.clear()
        self.shm.close()

    def unlink(self) -> None:
        if self.owner:
            self.shm.unlink()

    # --- views -------------------------------------------------------------

    def view(self, name: str) -> np.ndarray:
        v = self._views.get(name)
        if v is None:
            r = self.layout.region(name)
            v = np.ndarray((r.count,), dtype=r.dtype, buffer=self.shm.buf,
                           offset=r.offset)
            self._views[name] = v
        return v


class ReadOnlyArena:
    """An attached arena whose every view refuses writes.

    What a policy's `reader.arena` is built on (`rings.open_read_only`): the
    same ring classes as the processes use, over views with numpy's
    `writeable` flag off, so a ring's `write` — or any stray assignment into a
    region — raises `ValueError: assignment destination is read-only` instead
    of corrupting a region that has exactly one writer.

    A guard against mistakes, not a security boundary: the policy runs in the
    same process as P5 and could reach the writable arena if it tried.
    """

    def __init__(self, arena: Arena):
        self._arena = arena
        self.layout = arena.layout
        self._views: dict[str, np.ndarray] = {}

    def view(self, name: str) -> np.ndarray:
        v = self._views.get(name)
        if v is None:
            v = self._arena.view(name).view()
            v.flags.writeable = False
            self._views[name] = v
        return v
