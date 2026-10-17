"""The shared-memory arena: `schema` (every record), `layout` (the regions and the hash),
`primitives` (the seqlock and the memory fence), `rings` (the typed accessors). Created by P0,
attached by every process. How it fits together: `docs/architecture.md`, "The arena"."""

from elrobot_deploy.runtime.shared_memory import schema
from elrobot_deploy.runtime.shared_memory.layout import (Arena, Layout, LayoutMismatch,
                                                         ReadOnlyArena, Region, Sizes, build,
                                                         layout_for)
from elrobot_deploy.runtime.shared_memory.primitives import Seqlock, fence
from elrobot_deploy.runtime.shared_memory.rings import (CameraFrame, CameraRing, Control,
                                                        InterveneRing, InterveneRow, StateRing,
                                                        TargetEntry, TargetRing, open_all)

__all__ = [
    "Arena", "CameraFrame", "CameraRing", "Control", "InterveneRing", "InterveneRow", "Layout",
    "LayoutMismatch", "ReadOnlyArena", "Region", "Seqlock", "Sizes", "StateRing",
    "TargetEntry", "TargetRing", "build", "fence", "layout_for", "open_all", "schema",
]
