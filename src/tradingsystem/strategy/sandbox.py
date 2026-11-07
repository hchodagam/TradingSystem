"""Parent side of the strategy sandbox: run signals() in a low-privilege subprocess.

One child process runs any number of param sets (process start-up dominates: importing pandas alone
can take seconds), so callers that need many runs should use run_signals_many.
Limits: timeout + RSS watchdog over the child's process tree everywhere; RLIMIT_AS on POSIX (set by
the child itself). The low-privilege user and no-network container come with Docker (Plan 4).
"""
from __future__ import annotations

import io
import json
import os
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
from tradingsystem.strategy.params import Params
from tradingsystem.strategy.static_check import static_check

Reason = Literal["static", "timeout", "memory", "crash", "bad_output"]


class SandboxError(Exception):
    def __init__(self, reason: Reason, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SandboxConfig:
    timeout_s: float = 120  # per param set
    max_rss_mb: int = 2048


DEFAULT_SANDBOX = SandboxConfig()


def _encode(code: str, params_list: list[Params], bars: pd.DataFrame, incremental_from: int | None,
            max_rss_mb: int) -> bytes:
    header = json.dumps({"code": code, "params_list": [dict(p) for p in params_list],
                         "incremental_from": incremental_from, "max_rss_mb": max_rss_mb}).encode()
    tbl = pa.Table.from_pandas(bars.reset_index(), preserve_index=False)
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, tbl.schema) as w:
        w.write_table(tbl)
    return struct.pack(">I", len(header)) + header + sink.getvalue()


def _child_env() -> dict[str, str]:
    """Minimal environment: nothing from .env (Dhan token) or the user's shell reaches strategy code."""
    keep = ("SYSTEMROOT", "SystemRoot", "PATH", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONHASHSEED"] = "0"
    return env


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


def run_signals_many(strategy: LoadedStrategy, params_list: list[Params], bars: pd.DataFrame, *,
                     incremental_from: int | None = None, cfg: SandboxConfig = DEFAULT_SANDBOX,
                     _skip_static: bool = False) -> list[pd.DataFrame]:
    if not params_list:
        return []
    if not _skip_static:
        violations = static_check(strategy.code)
        if violations:
            raise SandboxError("static", "; ".join(violations))

    payload = _encode(strategy.code, params_list, bars, incremental_from, cfg.max_rss_mb)
    timeout = cfg.timeout_s * len(params_list)
    with tempfile.TemporaryDirectory(prefix="ts-sandbox-") as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "tradingsystem.strategy.runner"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=_child_env(),
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
            if time.monotonic() - started > timeout:
                killed = "timeout"
            elif _tree_rss(ps) > cfg.max_rss_mb * 1024 * 1024:
                killed = "memory"
            if killed:
                _kill_tree(ps)
                t.join(10)
                break

    if killed == "timeout":
        raise SandboxError("timeout", f"signals() exceeded {cfg.timeout_s}s per param set")
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

    buf, pos, outs = result["out"], 0, []
    for _ in params_list:
        (n,) = struct.unpack(">Q", buf[pos:pos + 8])
        out = pa.ipc.open_stream(io.BytesIO(buf[pos + 8:pos + 8 + n])).read_all().to_pandas()
        pos += 8 + n
        if {"date", "symbol"} <= set(out.columns):
            out = out.set_index(["date", "symbol"])
        outs.append(out)
    return outs


def run_signals(strategy: LoadedStrategy, params: Params, bars: pd.DataFrame, *,
                incremental_from: int | None = None, cfg: SandboxConfig = DEFAULT_SANDBOX,
                _skip_static: bool = False) -> pd.DataFrame:
    return run_signals_many(strategy, [params], bars, incremental_from=incremental_from, cfg=cfg,
                            _skip_static=_skip_static)[0]
