"""The arena: one shared-memory segment, statically divided, validated by a layout hash.

No allocator, no pointers (virtual addresses differ per process), no Python objects. Every
process computes the layout from the same `Sizes` and refuses to attach if the hash disagrees,
which turns "P6 and the bus process were built from different configs" from a wrong joint
target into a startup error.

P0 creates it (`Arena.create`), every other process attaches (`Arena.attach`). Adding or
resizing a region changes the hash; changing what a region MEANS needs a `schema.VERSION` bump
as well. The arena is small (well under 1 MB without cameras).
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

    def region(self, name: str) -> Region:
        return self.regions[name]

    def describe(self) -> str:
        lines = [f"arena {self.size/1e6:.2f} MB  hash={self.hash:#018x}"]
        for r in self.regions.values():
            lines.append(f"  {r.name:<24} {r.count:>6} x {r.dtype.itemsize:>9} B "
                         f"= {r.nbytes/1e6:8.3f} MB  @ {r.offset}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Sizes:
    """How many slots each ring has. The only input to the layout, so every process that
    builds it from the same `Sizes` gets the same hash."""

    #: ~2 s of targets at 60 Hz. The bus process only looks a lead (~40 ms) ahead; P0's ramps
    #: pace themselves to stay inside it (`ramps.write_ramp`).
    n_target: int = 128
    #: ~4 s of bus reads at 250 Hz.
    n_state: int = 1024
    #: ~4 s of phone samples at 60 Hz.
    n_intervene: int = 256


DEFAULT_SIZES = Sizes()


def build(sizes: Sizes = DEFAULT_SIZES) -> Layout:
    """Pure: sizes in, region table out. Called identically by every process."""
    regions: dict[str, Region] = {}
    offset = 0

    def add(name: str, dtype: np.dtype, count: int) -> None:
        nonlocal offset
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        regions[name] = Region(name, np.dtype(dtype), count, offset)
        offset += np.dtype(dtype).itemsize * count

    add("header", schema.HEADER, 1)
    add("control", schema.CONTROL, 1)

    add("state_seq", np.uint64, sizes.n_state)
    add("state_rec", schema.STATE, sizes.n_state)

    add("target_seq", np.uint64, sizes.n_target)
    add("target_rec", schema.TARGET, sizes.n_target)

    # The operator's stream. Allocated whether or not this launch runs the phone source: the
    # layout hash every process checks must not depend on a flag one process was given and
    # another was not. An unwritten region costs address space and nothing else.
    add("intervene_seq", np.uint64, sizes.n_intervene)
    add("intervene_rec", schema.INTERVENE, sizes.n_intervene)
    add("intervene_meta", schema.INTERVENE_META, 1)

    digest = hashlib.blake2b(digest_size=8)
    for r in regions.values():
        digest.update(f"{r.name}|{r.dtype.str}|{r.dtype.itemsize}|"
                      f"{r.count}|{r.offset}".encode())
    layout_hash = int.from_bytes(digest.digest(), "little")

    return Layout(regions=regions, size=offset, hash=layout_hash)


class LayoutMismatch(RuntimeError):
    """Attached to an arena built from a different config."""


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
    def create(cls, name: str, layout: Layout) -> Arena:
        shm = shared_memory.SharedMemory(name=name, create=True, size=layout.size)
        _CREATED.add(name)
        arena = cls(shm, layout, owner=True)
        arena.view("header")[0] = (schema.MAGIC, schema.VERSION, layout.hash)
        return arena

    @classmethod
    def attach(cls, name: str, layout: Layout, *, standalone: bool = False) -> Arena:
        """Attach to an arena another process created.

        Python registers every attach with a resource tracker, which unlinks the segment when
        the tracker's processes are all gone. A child SPAWNED by the creator (every process P0
        starts) shares the creator's tracker, so its registration is the creator's own and must
        be left alone: unregistering it would make the creator's `unlink` fail. A process
        started on its own (`standalone`, e.g. a tool attaching to a running session) has its
        own tracker, which would unlink the arena under everyone when it exits, so it
        unregisters.
        """
        shm = shared_memory.SharedMemory(name=name, create=False)
        if standalone and name not in _CREATED:
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
