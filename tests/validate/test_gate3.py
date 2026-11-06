from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from synthetic import make_bars

from tradingsystem.config import load_config
from tradingsystem.strategy.loader import load_strategy
from tradingsystem.universe import Universe
from tradingsystem.validate.gate2 import BacktestContext
from tradingsystem.validate.gate3 import HoldoutAlreadyUsed, HoldoutLedger, gate3

ROOT = Path(__file__).parents[2]
FIX = ROOT / "tests" / "fixtures" / "strategies"
CFG = load_config(ROOT / "config")
SYMS = [f"S{i:02d}" for i in range(8)]


class FakeStore:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail
        self.bars = make_bars(SYMS, date(2021, 1, 1), date(2024, 6, 28), seed=7, drift=-0.002, planted_edge=True)

    def holdout_bars(self, symbols):
        self.calls += 1
        if self.fail:
            raise RuntimeError("disk on fire")
        return self.bars


@pytest.fixture
def ctx():
    return BacktestContext(Universe([(date(2011, 1, 1), s, "add") for s in SYMS]), {s: s for s in SYMS},
                           pd.Series(dtype=float))


@pytest.fixture
def ledger(tmp_path):
    return HoldoutLedger(tmp_path / "state" / "holdout_runs.json")


@pytest.fixture(scope="module")
def strategy():
    return load_strategy(FIX / "dip_buyer.py")


def test_runs_once_then_refuses(strategy, ctx, ledger):
    store = FakeStore()
    g = gate3(strategy, store, ctx, CFG, research_sharpe=2.0, research_passed_gates_1_2=True, ledger=ledger)
    assert g.gate == "gate3" and [c.name for c in g.checks] == ["sharpe_ratio_vs_research", "max_drawdown"]
    assert ledger.used(strategy.hash)
    with pytest.raises(HoldoutAlreadyUsed):
        gate3(strategy, store, ctx, CFG, research_sharpe=2.0, research_passed_gates_1_2=True, ledger=ledger)
    assert store.calls == 1


def test_ledger_persists_across_instances(strategy, ctx, ledger):
    gate3(strategy, FakeStore(), ctx, CFG, 2.0, True, ledger)
    assert HoldoutLedger(ledger.path).used(strategy.hash)


def test_refuses_when_gates_1_2_not_passed(strategy, ctx, ledger):
    store = FakeStore()
    with pytest.raises(ValueError):
        gate3(strategy, store, ctx, CFG, 2.0, False, ledger)
    assert store.calls == 0 and not ledger.used(strategy.hash)


def test_crash_during_run_still_uses_the_holdout(strategy, ctx, ledger):
    with pytest.raises(RuntimeError):
        gate3(strategy, FakeStore(fail=True), ctx, CFG, 2.0, True, ledger)
    assert ledger.used(strategy.hash)


def test_changed_code_gets_its_own_entry(strategy, ctx, ledger, tmp_path):
    gate3(strategy, FakeStore(), ctx, CFG, 2.0, True, ledger)
    p = tmp_path / "dip2.py"
    p.write_text(strategy.code + "\n# tweak\n")
    other = load_strategy(p)
    gate3(other, FakeStore(), ctx, CFG, 2.0, True, ledger)
    hashes = [e["hash"] for e in ledger.entries()]
    assert hashes == [strategy.hash, other.hash]
    assert all(e["result"] is not None for e in ledger.entries())


def test_trades_never_before_holdout_start(strategy, ctx, ledger):
    from tradingsystem.validate.gate3 import holdout_backtest
    r = holdout_backtest(strategy, FakeStore().bars, ctx, CFG)
    assert not r.trades.empty
    assert r.trades.entry_date.min() >= pd.Timestamp(CFG.gates.holdout_start)
    assert r.equity.index.min() >= pd.Timestamp(CFG.gates.holdout_start)
