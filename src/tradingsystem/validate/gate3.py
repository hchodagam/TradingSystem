"""Gate 3: the locked holdout, run once per strategy hash. Failure is final: no re-tuning."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingsystem.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from tradingsystem.config import AppConfig
from tradingsystem.data.checks import ValidationReport
from tradingsystem.strategy.acceptance import check_output
from tradingsystem.strategy.loader import LoadedStrategy
from tradingsystem.strategy.sandbox import SandboxError, run_signals
from tradingsystem.validate import CheckResult, GateResult
from tradingsystem.validate.gate2 import BacktestContext
from tradingsystem.validate.metrics import max_drawdown, returns, sharpe


class HoldoutAlreadyUsed(Exception):
    pass


class HoldoutDataInvalid(Exception):
    def __init__(self, report):
        super().__init__(f"holdout data failed validation ({len(report.issues)} issues); the holdout was NOT used")
        self.report = report


class HoldoutLedger:
    """Append-only JSON list of holdout runs (moves to Postgres in Plan 3). The entry is written once the
    holdout data has loaded and validated, before the strategy runs: a strategy crash counts as the run,
    a data or infrastructure failure does not."""

    def __init__(self, path: Path = Path("state/holdout_runs.json")):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def used(self, hash: str) -> bool:
        return any(e["hash"] == hash for e in self.entries())

    def claim(self, hash: str, name: str) -> None:
        if self.used(hash):
            raise HoldoutAlreadyUsed(f"holdout already run for {name} ({hash[:12]}); failure is final")
        self._write([*self.entries(), {"hash": hash, "name": name,
                                       "started": datetime.now(UTC).isoformat(), "result": None}])

    def record(self, hash: str, result: GateResult) -> None:
        entries = self.entries()
        for e in entries:
            if e["hash"] == hash and e["result"] is None:
                e["result"] = {"passed": result.passed,
                               "checks": [{"name": c.name, "value": c.value, "threshold": c.threshold,
                                           "passed": c.passed} for c in result.checks]}
        self._write(entries)


def holdout_backtest(strategy: LoadedStrategy, bars: pd.DataFrame, ctx: BacktestContext,
                     cfg: AppConfig) -> BacktestResult:
    """Signals on research history + holdout (warm-up), trading only from holdout_start."""
    sig = run_signals(strategy, strategy.spec.defaults(), bars)
    errs = check_output(sig, bars)
    if errs:
        raise SandboxError("bad_output", "; ".join(errs))
    last =bars.index.get_level_values("date").max().date()
    bt = BacktestConfig(cfg.gates.holdout_start, last, cfg.risk.initial_capital)
    return run_backtest(sig, bars, ctx.universe, ctx.sectors, strategy.spec, cfg, bt)


def gate3(strategy: LoadedStrategy, store, ctx: BacktestContext, cfg: AppConfig, research_sharpe: float,
          research_passed_gates_1_2: bool, ledger: HoldoutLedger,
          validate: Callable[[pd.DataFrame], ValidationReport] | None = None) -> GateResult:
    """`validate` checks the holdout bars (the CLI passes validate_bars over [holdout_start, last])."""
    if not research_passed_gates_1_2:
        raise ValueError("the holdout runs only after Gates 1 and 2 pass")
    if ledger.used(strategy.hash):
        raise HoldoutAlreadyUsed(f"holdout already run for {strategy.spec.name} ({strategy.hash[:12]}); "
                                 "failure is final")
    bars = store.holdout_bars(sorted(ctx.universe.all_symbols()))
    if validate is not None:
        report = validate(bars)
        if not report.ok:
            raise HoldoutDataInvalid(report)
    ledger.claim(strategy.hash, strategy.spec.name)
    r = holdout_backtest(strategy, bars, ctx, cfg)
    s = sharpe(returns(r.equity), cfg.gates.risk_free_rate)
    ratio = s / research_sharpe if research_sharpe > 0 else 0.0
    dd = max_drawdown(r.equity)
    g3, g1 = cfg.gates.gate3, cfg.gates.gate1
    result = GateResult.of("gate3", [
        CheckResult("sharpe_ratio_vs_research", ratio, g3.min_sharpe_ratio_vs_research,
                    ratio >= g3.min_sharpe_ratio_vs_research),
        CheckResult("max_drawdown", dd, g1.max_drawdown, dd <= g1.max_drawdown),
    ])
    ledger.record(strategy.hash, result)
    return result
