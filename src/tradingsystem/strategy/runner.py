"""Sandbox child process. Started as: python -I -m tradingsystem.strategy.runner

stdin:  4-byte big-endian header length, JSON header {code, params, incremental_from}, Arrow IPC stream of bars.
stdout: Arrow IPC stream of the signals frame (reset_index).
stderr: on failure, a final JSON line {"reason": ..., "detail": ...}; exit code 3.
"""
from __future__ import annotations

import builtins
import io
import json
import struct
import sys
import traceback

ALLOWED = {"pandas", "numpy", "math"}

_SAFE_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "pow", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "print", "hash",
    "True", "False", "None", "NotImplemented", "Ellipsis",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError", "ZeroDivisionError", "ArithmeticError",
    "RuntimeError", "StopIteration", "AttributeError", "LookupError",
]


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of {name!r} is not allowed in strategies")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _fail(reason: str, detail: str) -> None:
    sys.stderr.write("\n" + json.dumps({"reason": reason, "detail": detail}) + "\n")
    sys.stderr.flush()
    sys.exit(3)


def main() -> None:
    raw_in = sys.stdin.buffer
    raw_out = sys.stdout.buffer
    sys.stdout = sys.stderr  # strategy prints go to stderr, never into the protocol stream

    import socket

    import numpy as np  # noqa: F401  (preload before restrictions)
    import pandas as pd
    import pyarrow as pa

    def _no_network(*a, **k):
        raise PermissionError("network access is not allowed in strategies")

    socket.socket = _no_network  # type: ignore[assignment]
    socket.create_connection = _no_network  # type: ignore[assignment]

    (n,) = struct.unpack(">I", raw_in.read(4))
    header = json.loads(raw_in.read(n))
    if sys.platform != "win32":  # POSIX: hard address-space cap on ourselves before strategy code runs
        import resource
        cap = int(header["max_rss_mb"]) * 1024 * 1024 * 2
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    table =pa.ipc.open_stream(io.BytesIO(raw_in.read())).read_all()
    bars = table.to_pandas().set_index(["date", "symbol"]).sort_index()

    from tradingsystem.strategy.spec import Params

    safe = {k: getattr(builtins, k) for k in _SAFE_BUILTINS}
    safe["__import__"] = _restricted_import
    env = {"__builtins__": safe, "__name__": "strategy"}
    try:
        exec(compile(header["code"], "strategy.py", "exec"), env)  # noqa: S102 - this is the sandbox
        signals = env["signals"]
        params = Params(header["params"])
        k = header.get("incremental_from")
        if k is None:
            out = signals(bars, params)
        else:
            dates = bars.index.get_level_values("date").unique()
            parts = []
            for t in range(k, len(dates)):
                sub = bars[bars.index.get_level_values("date") <= dates[t]]
                res = signals(sub, params)
                if not isinstance(res, pd.DataFrame):
                    out = res
                    break
                parts.append(res[res.index.get_level_values("date") == dates[t]])
            else:
                out = pd.concat(parts) if parts else bars.iloc[0:0][[]]
    except MemoryError:
        _fail("memory", "strategy ran out of memory")
    except BaseException:  # noqa: BLE001 - report every strategy failure to the parent
        _fail("crash", traceback.format_exc(limit=5))

    if not isinstance(out, pd.DataFrame):
        _fail("bad_output", f"signals() must return a DataFrame, got {type(out).__name__}")
    try:
        frame = out.reset_index()
        tbl = pa.Table.from_pandas(frame, preserve_index=False)
    except Exception as e:  # noqa: BLE001
        _fail("bad_output", f"signals() output cannot be serialised: {e}")
    with pa.ipc.new_stream(raw_out, tbl.schema) as w:
        w.write_table(tbl)
    raw_out.flush()


if __name__ == "__main__":
    main()
