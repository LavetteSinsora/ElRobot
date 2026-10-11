"""The shared-memory arena: `schema` (every record), `layout` (the regions and
the hash), `primitives` (the seqlock), `rings` (the typed accessors). Created by
P0, attached by every process. Documented in
`docs/infra_logic/arena_layout_and_reads.md`."""

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.layout import (Arena, Layout, LayoutMismatch,
                                                     ReadOnlyArena, Region,
                                                     ShmTooSmall, build, check_shm)
from elrobot_deploy.runtime.shared_memory.primitives import Seqlock
from elrobot_deploy.runtime.shared_memory.rings import (ArenaRings, CameraRegion,
                                                    CameraRing, ChunkRing,
                                                    Contribution, Control,
                                                    InterveneRing, InterveneRow,
                                                    StateRing, TargetEntry, TargetRing,
                                                    open_all, open_read_only)

__all__ = [
    "Arena", "ArenaRings", "CameraRegion", "CameraRing", "ChunkRing",
    "Contribution", "Control", "InterveneRing", "InterveneRow", "Layout",
    "LayoutMismatch", "ReadOnlyArena", "Region", "Seqlock", "ShmTooSmall",
    "StateRing", "TargetEntry", "TargetRing", "build", "check_shm", "open_all",
    "open_read_only", "schema",
]
