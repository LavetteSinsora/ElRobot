"""A child process in ANOTHER interpreter: P0's spawn, without pickle.

    <python> -m g1d_deploy.runtime.processes.policy_host      (the call arrives on stdin)

Every other child is a `multiprocessing` spawn of P0's own interpreter, and
receives its arguments pickled. P5 runs under the policy's interpreter
(`[policy] python`; `docs/deploying_a_policy/policy_contract.md`, "The
adapter's own Python"), whose
numpy may be another major version — a pickle names numpy's private modules,
so it cannot cross. Here the call — `{"dotted", "args", "kwargs"}` — comes as
`config_loading/json_transport.py` JSON on stdin, and the function is imported
and called exactly as `runtime/launcher/children.py::_child` would: SIGINT ignored first (the operator's Ctrl-C is
P0's alone).

numpy + stdlib only: part of the minimal core.
"""

from __future__ import annotations

import importlib
import signal
import sys


def main() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from g1d_deploy.config_loading import json_transport as codec

    call = codec.loads(sys.stdin.read())
    # Nothing reads stdin after this; a policy that did would block on a pipe
    # P0 has already closed, rather than on the operator's terminal.
    module, _, func = call["dotted"].rpartition(".")
    getattr(importlib.import_module(module), func)(
        *call.get("args", ()), **call.get("kwargs", {}))


if __name__ == "__main__":
    main()
