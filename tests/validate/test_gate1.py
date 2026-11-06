from pathlib import Path

import numpy as np
import pandas as pd

from tradingsystem.backtest.engine import TRADE_COLUMNS, BacktestResult
from tradingsystem.config import load_config
from tradingsystem.validate.gate1 import gate1

GATES = load_config(Path(__file__).parents[2] / "config").gates
DAYS = pd.bdate_range("2020-01-01", periods=500)


def result(n_trades: int, drift: float = 0.002) -> BacktestResult:
    rng = np.random.default_rng(0)
    equity = pd.Series(100_000 * np.cumprod(1 + rng.normal(drift, 0.005, len(DAYS))), index=DAYS)
    pnl = [100.0] * (n_trades * 6 // 10) + [-50.0] * (n_trades - n_trades * 6 // 10)
    trades = pd.DataFrame({c: None for c in TRADE_COLUMNS}, index=range(n_trades))
    trades["pnl_net"] = pnl
    return BacktestResult(equity=equity, trades=trades, kill_switch_trips=0)


def flat_benchmark():
    rng = np.random.default_rng(1)
    return pd.Series(10_000 * np.cumprod(1 + rng.normal(0, 0.01, len(DAYS))), index=DAYS)


def check(g, name):
    return next(c for c in g.checks if c.name == name)


def test_passes_when_all_checks_pass():
    g = gate1(result(100), flat_benchmark(), GATES)
    assert g.passed and g.gate == "gate1"
    assert [c.name for c in g.checks] == ["sharpe", "max_drawdown", "trades", "profit_factor", "beats_benchmark"]


def test_fails_on_trades_alone():
    g = gate1(result(99), flat_benchmark(), GATES)
    assert not g.passed
    assert [c.name for c in g.checks if not c.passed] == ["trades"]
    assert check(g, "trades").value == 99 and check(g, "trades").threshold == 100


def test_fails_when_benchmark_better():
    strong = pd.Series(10_000 * np.cumprod(np.full(len(DAYS), 1.003) + np.tile([0.0001, -0.0001], 250)),
                       index=DAYS)
    g = gate1(result(100), strong, GATES)
    assert not check(g, "beats_benchmark").passed


def test_benchmark_aligned_to_strategy_dates():
    bench = pd.concat([pd.Series([1.0, 1e6], index=pd.to_datetime(["2010-01-01", "2010-01-02"])), flat_benchmark()])
    assert gate1(result(100), bench, GATES).passed
