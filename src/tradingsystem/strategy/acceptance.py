"""Code acceptance pipeline: schema -> lint -> smoke (static + run + output check) -> look-ahead.
(The Critic review and the retry loop with the Code agent arrive in Plan 2.)"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from tradingsystem.strategy.loader import LoadedStrategy, StrategyFormatError, load_strategy
from tradingsystem.strategy.sandbox import SandboxError, run_signals
from tradingsystem.strategy.spec import Params

Stage = Literal["schema", "lint", "smoke", "lookahead", "passed"]
MAX_REPORTED = 20


@dataclass(frozen=True)
class AcceptanceResult:
    ok: bool
    stage: Stage
    errors: list[str] = field(default_factory=list)
    strategy: LoadedStrategy | None = None


def check_output(out: pd.DataFrame, bars: pd.DataFrame) -> list[str]:
    errs: list[str] = []
    missing = [c for c in ("entry", "exit", "stop") if c not in out.columns]
    if missing:
        errs.append(f"missing column(s): {', '.join(missing)} (need entry, exit, stop)")
    if not out.index.equals(bars.index):
        errs.append(f"index differs from the bars index ({len(out)} rows vs {len(bars)}); "
                    "return one row per input (date, symbol)")
    for col in ("entry", "exit"):
        if col in out.columns:
            if out[col].isna().any():
                errs.append(f"NaN in {col}: flags must be True/False on every row")
            elif out[col].dtype != bool:
                errs.append(f"{col} must be bool, got {out[col].dtype}")
    if "stop" in out.columns:
        if not pd.api.types.is_float_dtype(out["stop"]):
            errs.append(f"stop must be float (NaN for none), got {out['stop'].dtype}")
        elif (out["stop"] <= 0).any():
            errs.append("stop must be > 0 where set")
    return errs


def lookahead_check(s: LoadedStrategy, params: Params, bars: pd.DataFrame, window: int = 250,
                    batch: pd.DataFrame | None = None) -> list[str]:
    """Re-run signals() day by day over the last `window` dates; output must equal the batch run.
    Pass `batch` (signals on the same bars and params) to save a sandbox start."""
    dates = bars.index.get_level_values("date").unique()
    k = max(len(dates) - window, 1)
    if batch is None:
        batch = run_signals(s, params, bars)
    inc = run_signals(s, params, bars, incremental_from=k)
    expected = batch[batch.index.get_level_values("date") >= dates[k]]
    if not inc.index.equals(expected.index):
        return ["incremental run returned different rows from the batch run"]
    diffs: list[str] = []
    for col in ("entry", "exit"):
        bad = inc[col].to_numpy() != expected[col].to_numpy()
        for d, sym in inc.index[bad][:MAX_REPORTED]:
            diffs.append(f"{col} differs on {pd.Timestamp(d).date()} {sym}: uses data after that day")
    a, b = inc["stop"].to_numpy(dtype=float), expected["stop"].to_numpy(dtype=float)
    bad = ~np.isclose(a, b, equal_nan=True)
    for d, sym in inc.index[bad][:MAX_REPORTED]:
        diffs.append(f"stop differs on {pd.Timestamp(d).date()} {sym}: uses data after that day")
    return diffs[:MAX_REPORTED]


def _lint(path: Path) -> list[str]:
    r = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", "--isolated", "--select", "E,F,B",
         "--line-length", "120", "--output-format", "concise", str(path)],
        check=False, capture_output=True, text=True,
    )
    if r.returncode == 0:
        return []
    return [ln for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("Found")] or [r.stderr.strip()]


def accept(path: Path, bars: pd.DataFrame) -> AcceptanceResult:
    try:
        s = load_strategy(path)
    except StrategyFormatError as e:
        return AcceptanceResult(False, "schema", [str(e)])
    lint = _lint(Path(path))
    if lint:
        return AcceptanceResult(False, "lint", lint, s)
    params = s.spec.defaults()
    try:
        out = run_signals(s, params, bars)
    except SandboxError as e:
        return AcceptanceResult(False, "smoke", [f"{e.reason}: {e.detail}"], s)
    errs = check_output(out, bars)
    if errs:
        return AcceptanceResult(False, "smoke", errs, s)
    try:
        diffs = lookahead_check(s, params, bars, batch=out)
    except SandboxError as e:
        return AcceptanceResult(False, "lookahead", [f"{e.reason}: {e.detail}"], s)
    if diffs:
        return AcceptanceResult(False, "lookahead", diffs, s)
    return AcceptanceResult(True, "passed", [], s)
