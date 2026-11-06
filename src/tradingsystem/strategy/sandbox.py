"""Parent side of the strategy sandbox: run signals() in a low-privilege subprocess."""
from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import psutil
import pyarrow as pa

from tradingsystem.strategy.loader import LoadedStrategy
from tradingsystem.strategy.spec import Params
from tradingsystem.strategy.static_check import static_check

Reason = Literal["static", "timeout", "memory", "crash", "bad_output"]


class SandboxError(Exception):
    def __init__(self, reason: Reason, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SandboxConfig:
    timeout_s: float = 120
    max_rss_mb: int = 2048


DEFAULT_SANDBOX = SandboxConfig()


def _encode(code: str, params: Params, bars: pd.DataFrame, incremental_from: int | None, max_rss_mb: int) -> bytes:
    header = json.dumps({"code": code, "params": dict(params), "incremental_from": incremental_from,
                         "max_rss_mb": max_rss_mb}).encode()
    tbl = pa.Table.from_pandas(bars.reset_index(), preserve_index=False)
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, tbl.schema) as w:
        w.write_table(tbl)
    return struct.pack(">I", len(header)) + header + sink.getvalue()


def _tree_rss(ps: psutil.Process) -> int:
    """RSS of the child and its descendants (a Windows venv python.exe is a launcher with a real child)."""
    total = 0
    try:
        procs = [ps, *ps.children(recursive=True)]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def _kill_tree(ps: psutil.Process) -> None:
    try:
        procs = [*ps.children(recursive=True), ps]
    except psutil.Error:
        procs = [ps]
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass


def run_signals(strategy: LoadedStrategy, params: Params, bars: pd.DataFrame, *,
                incremental_from: int | None = None, cfg: SandboxConfig = DEFAULT_SANDBOX,
                _skip_static: bool = False) -> pd.DataFrame:
    """Run signals() in a subprocess. Limits: timeout + RSS watchdog everywhere; RLIMIT_AS on POSIX
    (set by the child itself). Low-privilege user and no-network container come with Docker (Plan 4)."""
    if not _skip_static:
        violations = static_check(strategy.code)
        if violations:
            raise SandboxError("static", "; ".join(violations))

    payload = _encode(strategy.code, params, bars, incremental_from, cfg.max_rss_mb)
    with tempfile.TemporaryDirectory(prefix="ts-sandbox-") as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "tradingsystem.strategy.runner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
        )
        result: dict = {}

        def talk():
            result["out"], result["err"] = proc.communicate(payload)

        t = threading.Thread(target=talk, daemon=True)
        t.start()
        killed: Reason | None = None
        started = time.monotonic()
        ps = psutil.Process(proc.pid)
        while t.is_alive():
            t.join(0.05)
            if not t.is_alive():
                break
            if time.monotonic() - started > cfg.timeout_s:
                killed = "timeout"
            elif _tree_rss(ps) > cfg.max_rss_mb * 1024 * 1024:
                killed = "memory"
            if killed:
                _kill_tree(ps)
                t.join(10)
                break

    if killed == "timeout":
        raise SandboxError("timeout", f"signals() exceeded {cfg.timeout_s}s")
    if killed == "memory":
        raise SandboxError("memory", f"signals() exceeded {cfg.max_rss_mb} MB")
    if proc.returncode != 0:
        err = result.get("err", b"").decode("utf-8", "replace")
        last = err.strip().splitlines()[-1] if err.strip() else ""
        try:
            info = json.loads(last)
            raise SandboxError(info["reason"], info["detail"])
        except (json.JSONDecodeError, KeyError, TypeError):
            raise SandboxError("crash", err[-2000:] or f"exit code {proc.returncode}") from None

    out = pa.ipc.open_stream(io.BytesIO(result["out"])).read_all().to_pandas()
    if {"date", "symbol"} <= set(out.columns):
        out = out.set_index(["date", "symbol"])
    return out
