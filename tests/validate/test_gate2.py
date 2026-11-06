from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic import make_bars

from tradingsystem.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from tradingsystem.config import load_config
from tradingsystem.strategy.loader import load_strategy
from tradingsystem.strategy.sandbox import run_signals
from tradingsystem.universe import Universe
from tradingsystem.validate.gate1 import gate1
from tradingsystem.validate.gate2 import BacktestContext, gate2, mc_drawdown_pct95, thresholds, trade_returns

ROOT = Path(__file__).parents[2]
FIX = ROOT / "tests" / "fixtures" / "strategies"
BASE = load_config(ROOT / "config")
# small grid keeps the test fast; thresholds are the real ones
CFG = BASE.model_copy(update={"gates": BASE.gates.model_copy(update={
    "gate2": BASE.gates.gate2.model_copy(update={"grid_values_per_param": 3, "grid_max_combos": 10})})})
SYMS = [f"S{i:02d}" for i in range(20)]
START, END = date(2012, 1, 1), date(2022, 12, 31)


def ctx_for(bars):
    return BacktestContext(Universe([(date(2011, 1, 1), s, "add") for s in SYMS]), {s: s for s in SYMS},
                           bars["close"].unstack().mean(axis=1))


@pytest.fixture(scope="module")
def noise_bars():
    return make_bars(SYMS, START, END, seed=1)


@pytest.fixture(scope="module")
def edge_bars():
    return make_bars(SYMS, START, END, seed=7, drift=-0.002, planted_edge=True)


def test_random_entry_fails_gate2(noise_bars):
    s = load_strategy(FIX / "random_entry.py")
    g = gate2(s, noise_bars, ctx_for(noise_bars), CFG, family_trials=1, family_sr_var=None)
    assert not g.passed
    assert not next(c for c in g.checks if c.name == "cost_stress").passed


def test_planted_edge_passes_gate1_and_gate2(edge_bars):
    s = load_strategy(FIX / "dip_buyer.py")
    ctx = ctx_for(edge_bars)
    sig = run_signals(s, s.spec.defaults(), edge_bars)
    r = run_backtest(sig, edge_bars, ctx.universe, ctx.sectors, s.spec, CFG,
                     BacktestConfig(START, END, CFG.risk.initial_capital))
    g1 = gate1(r, ctx.benchmark, CFG.gates)
    assert g1.passed, g1.checks
    g2 = gate2(s, edge_bars, ctx, CFG, family_trials=1, family_sr_var=None)
    assert g2.passed, [c for c in g2.checks if not c.passed]
    assert [c.name for c in g2.checks] == ["wf_oos_ratio", "wf_losing_years", "sensitivity", "mc_dd95",
                                           "cost_stress", "dsr"]


def test_overfit_strategy_uses_stricter_thresholds(tmp_path):
    params = ", ".join(f'"p{i}": {{"default": 1, "range": [0, 2]}}' for i in range(6))
    code = (FIX / "random_entry.py").read_text().replace(
        '"params": {"pct": {"default": 0.05, "range": [0.02, 0.10]}}', '"params": {' + params + "}")
    p = tmp_path / "six.py"
    p.write_text(code)
    assert thresholds(load_strategy(p), CFG) == (0.80, 0.99)
    assert thresholds(load_strategy(FIX / "random_entry.py"), CFG) == (0.70, 0.95)


def test_mc_drawdown_known_case():
    # all orderings identical: three -1% trades compound to 1 - 0.99^3
    assert mc_drawdown_pct95(np.array([-0.01] * 3), 50, 0) == pytest.approx(1 - 0.99**3)


def test_mc_drawdown_compounding_never_exceeds_100pct():
    # ten doublings and one -50% trade: whatever the order the worst drawdown is exactly 50%
    assert mc_drawdown_pct95(np.array([1.0] * 10 + [-0.5]), 200, 0) == pytest.approx(0.5)


def test_mc_no_trades_is_zero():
    assert mc_drawdown_pct95(np.array([]), 50, 0) == 0.0


def test_trade_returns_relative_to_equity_before_entry():
    days = pd.bdate_range("2024-01-01", periods=4)
    equity = pd.Series([100_000.0, 100_000.0, 200_000.0, 210_000.0], index=days)
    trades = pd.DataFrame({"entry_date": [days[1], days[3]], "pnl_net": [1_000.0, 2_000.0]})
    r = BacktestResult(equity=equity, trades=trades, kill_switch_trips=0)
    assert list(trade_returns(r, 100_000.0)) == pytest.approx([0.01, 0.01])
