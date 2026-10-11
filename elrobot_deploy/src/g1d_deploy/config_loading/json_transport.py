"""`DeployConfig` / `Hardware` as JSON, for a process in ANOTHER interpreter.

P5 runs under the policy's own Python (`[policy] python`;
`docs/deploying_a_policy/policy_contract.md`, "The adapter's own Python"), so
P0 cannot hand it the config by pickling, the
way `multiprocessing` hands every other child: a pickle names numpy's private
modules (`numpy.core` in 1.x, `numpy._core` in 2.x), and the two sides may run
different numpy majors. This is the pickle-free path: plain JSON with a small
tag for each type JSON lacks, read back into the SAME dataclasses.

Only what the configs actually hold is supported — dataclasses of this
package, dicts with string keys, lists, tuples, `Path`s, numpy arrays and
scalars, and JSON's own scalars. Anything else is refused at `dumps`, so a new
config field of an unsupported type fails a test instead of a launch. numpy
arrays are carried as `(dtype.str, shape, flat list)`, which reads back
bit-identically: every float is written by `repr` and JSON keeps it exact.

Written by P0 (`runtime/launcher/children.py`), read by
`runtime/processes/policy_host.py` in the policy's interpreter.

numpy + stdlib only: it is part of the minimal core a policy venv installs.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
from pathlib import Path, PurePath

import numpy as np

#: Dataclasses may only be rebuilt from this package — a spec file is data,
#: and must not name an arbitrary class to construct.
_PACKAGE = __name__.rpartition(".")[0] or "g1d_deploy"

_TAG = "__g1d__"


def _float(v: float):
    # JSON has no inf/nan; `json` writes them as bare tokens other readers
    # refuse, so they are tagged.
    return v if math.isfinite(v) else {_TAG: "float", "v": repr(v)}


def encode(obj):
    """`obj` -> a JSON-able tree."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return _float(obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in "biuf":
            raise TypeError(f"codec: array of dtype {obj.dtype} is not supported")
        flat = [_float(float(x)) if obj.dtype.kind == "f" else
                (bool(x) if obj.dtype.kind == "b" else int(x))
                for x in obj.reshape(-1).tolist()]
        return {_TAG: "ndarray", "dtype": obj.dtype.str,
                "shape": list(obj.shape), "data": flat}
    if isinstance(obj, np.generic):
        return {_TAG: "scalar", "dtype": obj.dtype.str,
                "v": encode(obj.item())}
    if isinstance(obj, PurePath):
        return {_TAG: "path", "v": str(obj)}
    if isinstance(obj, tuple):
        return {_TAG: "tuple", "v": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, str)]
        if bad:
            raise TypeError(f"codec: dict keys must be str, got {bad[:3]!r}")
        if _TAG in obj:
            raise TypeError(f"codec: a dict may not use the key {_TAG!r}")
        return {k: encode(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        if not cls.__module__.startswith(_PACKAGE + ".") and cls.__module__ != _PACKAGE:
            raise TypeError(f"codec: dataclass {cls.__module__}.{cls.__qualname__} "
                            f"is not one of {_PACKAGE}'s")
        return {_TAG: "dataclass", "cls": f"{cls.__module__}:{cls.__qualname__}",
                "fields": {f.name: encode(getattr(obj, f.name))
                           for f in dataclasses.fields(obj) if f.init}}
    raise TypeError(f"codec: {type(obj).__name__} is not supported")


def decode(tree):
    """The inverse of `encode`."""
    if isinstance(tree, list):
        return [decode(x) for x in tree]
    if not isinstance(tree, dict):
        return tree
    tag = tree.get(_TAG)
    if tag is None:
        return {k: decode(v) for k, v in tree.items()}
    if tag == "float":
        return float(tree["v"])
    if tag == "ndarray":
        data = [decode(x) for x in tree["data"]]
        return np.array(data, dtype=np.dtype(tree["dtype"])).reshape(tree["shape"])
    if tag == "scalar":
        return np.dtype(tree["dtype"]).type(decode(tree["v"]))
    if tag == "path":
        return Path(tree["v"])
    if tag == "tuple":
        return tuple(decode(x) for x in tree["v"])
    if tag == "dataclass":
        module, _, qualname = tree["cls"].partition(":")
        if not (module == _PACKAGE or module.startswith(_PACKAGE + ".")):
            raise ValueError(f"codec: refusing to build {tree['cls']!r}")
        obj = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            raise ValueError(f"codec: {tree['cls']!r} is not a dataclass")
        return obj(**{k: decode(v) for k, v in tree["fields"].items()})
    raise ValueError(f"codec: unknown tag {tag!r}")


def dumps(obj) -> str:
    return json.dumps(encode(obj), allow_nan=False)


def loads(text: str):
    return decode(json.loads(text))


__all__ = ["decode", "dumps", "encode", "loads"]
