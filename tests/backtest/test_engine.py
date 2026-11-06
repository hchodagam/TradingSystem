"""Golden tests on hand-built bars. Slippage 0.1%; charges from config/costs.yaml."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingsystem.backtest.engine import BacktestConfig, run_backtest
from tradingsystem.calendar import trading_days
from tradingsystem.config import load_config
from tradingsystem.strategy.spec import StrategySpec
from tradingsystem.universe import Universe

_BASE = load_config(Path(__file__).parents[2] / "config")
CFG = _BASE.model_copy(update={"risk": _BASE.risk.model_copy(update={"atr_len": 2})})  # 12-day fixture
DAYS = list(trading_days(date(2024, 1, 1), date(2024, 1, 31))[:12])
SECTORS = {"A": "x", "B": "y"}
UNI = Universe([(date(2023, 1, 1), "A", "add"), (date(2023, 1, 1), "B", "add")])


def spec(max_hold=50):
    return StrategySpec.model_validate({
        "name": "t", "family": "t", "hypothesis": "t", "universe": "nifty100", "timeframe": "1D",
        "params": {"n": {"default": 1, "range": [0, 2]}}, "max_holding_days": max_hold})


def make_bars(symbols=("A", "B")):
    idx = pd.MultiIndex.from_product([DAYS, list(symbols)], names=["date", "symbol"])
    # high/low 101/99 -> ATR(2) = 2, so a NaN stop sizes at 2xATR = 4 per share: 250 by risk, 200 by cap
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6}, index=idx)


def set_bar(bars, day, sym, **vals):
    for k, v in vals.items():
        bars.loc[(DAYS[day], sym), k] = float(v)


def make_signals(bars, entries=(), exits=(), stops=None):
    s = pd.DataFrame({"entry": False, "exit": False, "stop": np.nan}, index=bars.index)
    for d, sym in entries:
        s.loc[(DAYS[d], sym), "entry"] = True
    for d, sym in exits:
        s.loc[(DAYS[d], sym), "exit"] = True
    for (d, sym), v in (stops or {}).items():
        s.loc[(DAYS[d], sym), "stop"] = v
    return s


def run(bars, sigs, cfg=CFG, max_hold=50, universe=UNI, capital=100_000.0, **bt):
    b = BacktestConfig(start=DAYS[0].date(), end=DAYS[-1].date(), initial_capital=capital, **bt)
    return run_backtest(sigs, bars, universe, SECTORS, spec(max_hold), cfg, b)


def test_signal_fills_next_open_with_slippage():
    bars = make_bars()
    r = run(bars, make_signals(bars, entries=[(1, "A")], stops={(1, "A"): 90.0}))
    t = r.trades.iloc[0]
    assert t.symbol == "A" and t.entry_date == DAYS[2]
    assert t.entry_price == pytest.approx(100.1)
    assert t.qty == 100  # 1% of 1L / (100 - 90)


def test_charges_deducted_from_cash():
    bars = make_bars()
    r = run(bars, make_signals(bars, entries=[(1, "A")], exits=[(4, "A")], stops={(1, "A"): 90.0}))
    t = r.trades.iloc[0]
    # buy 100 @ 100.1: STT 10, exch 0.30, SEBI 0.01, stamp 2, GST 0.06 = 12.37
    # sell 100 @ 99.9: STT 10, exch 0.30, SEBI 0.01, DP 12.5, GST 2.31 = 25.12
    assert t.exit_reason == "signal" and t.exit_date == DAYS[5]
    assert t.charges == pytest.approx(37.49)
    assert t.pnl_net == pytest.approx(-20.0 - 37.49)
    assert r.equity.iloc[-1] == pytest.approx(100_000 + r.trades.pnl_net.sum())


def test_gap_through_stop_fills_at_open():
    bars = make_bars()
    set_bar(bars, 3, "A", open=85, high=86, low=80, close=82)
    r = run(bars, make_signals(bars, entries=[(1, "A")], stops={(1, "A"): 90.0}))
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_date == DAYS[3]
    assert t.exit_price == pytest.approx(85 * 0.999)


def test_intraday_stop_fills_at_stop():
    bars = make_bars()
    set_bar(bars, 3, "A", open=95, high=96, low=88, close=92)
    t = run(bars, make_signals(bars, entries=[(1, "A")], stops={(1, "A"): 90.0})).trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_price == pytest.approx(90 * 0.999)


def test_stop_on_entry_day_intraday():
    bars = make_bars()
    set_bar(bars, 2, "A", open=100, high=100, low=89, close=95)
    t = run(bars, make_signals(bars, entries=[(1, "A")], stops={(1, "A"): 90.0})).trades.iloc[0]
    assert t.entry_date == DAYS[2] and t.exit_date == DAYS[2] and t.exit_reason == "stop"


def test_trailing_stop_from_later_signal_rows():
    bars = make_bars()
    set_bar(bars, 5, "A", open=99, high=99.5, low=96, close=97)
    sigs = make_signals(bars, entries=[(1, "A")], stops={(1, "A"): 90.0, (3, "A"): 97.5})
    t = run(bars, sigs).trades.iloc[0]
    assert t.exit_reason == "stop" and t.exit_date == DAYS[5]
    assert t.exit_price == pytest.approx(97.5 * 0.999)


def test_opened_below_stop_entry_skipped():
    bars = make_bars()
    set_bar(bars, 2, "A", open=89, high=95, low=88, close=94)
    r = run(bars, make_signals(bars, entries=[(1, "A")], stops={(1, "A"): 90.0}))
    assert r.trades.empty
    assert list(r.skipped.reason) == ["opened_below_stop"]


def test_max_holding_exit():
    bars = make_bars()
    t = run(bars, make_signals(bars, entries=[(1, "A")]), max_hold=3).trades.iloc[0]
    # entered day 2; closes on days 2, 3, 4 -> 3 days held -> exit at day 5 open
    assert t.exit_reason == "max_holding" and t.exit_date == DAYS[5]


def test_end_of_test_closes_at_last_close():
    bars = make_bars()
    set_bar(bars, 11, "A", open=100, high=110, low=100, close=110)
    t = run(bars, make_signals(bars, entries=[(1, "A")])).trades.iloc[0]
    assert t.exit_reason == "end_of_test" and t.exit_date == DAYS[11]
    assert t.exit_price == pytest.approx(110 * 0.999)


def test_fill_respects_position_cap_at_fill_price():
    bars = make_bars()
    set_bar(bars, 2, "A", open=105, high=106, low=104, close=105)
    t = run(bars, make_signals(bars, entries=[(1, "A")], stops={(1, "A"): 99.5})).trades.iloc[0]
    # sized on close 100: by risk 2000, by cap 200 -> 200; at fill 105.105 the 20% cap allows 190
    assert t.qty == 190


def test_insufficient_cash_reduces_qty():
    bars = make_bars()
    set_bar(bars, 2, "A", open=102, high=103, low=101, close=102)
    risk = CFG.risk.model_copy(update={"max_position_pct": 1.0, "risk_per_trade_pct": 0.5})
    r = run(bars, make_signals(bars, entries=[(1, "A")]), cfg=CFG.model_copy(update={"risk": risk}))
    t = r.trades.iloc[0]
    assert 900 < t.qty < 1000
    assert t.qty * t.entry_price <= 100_000


def test_missing_bar_marks_last_close_then_exits():
    bars = make_bars().drop([(DAYS[d], "A") for d in range(3, 12)])
    set_bar(bars, 2, "A", close=104)
    r = run(bars, make_signals(bars, entries=[(1, "A")]))
    t = r.trades.iloc[0]
    # missing on days 3..7 -> 5th missing close is day 7
    assert t.exit_reason == "data_missing" and t.exit_date == DAYS[7]
    assert t.exit_price == pytest.approx(104 * 0.999)
    assert r.equity.loc[DAYS[5]] == pytest.approx(r.equity.loc[DAYS[3]])


def test_non_member_entry_ignored():
    bars = make_bars()
    only_a = Universe([(date(2023, 1, 1), "A", "add")])
    r = run(bars, make_signals(bars, entries=[(1, "B")]), universe=only_a)
    assert r.trades.empty


def test_start_before_first_universe_snapshot():
    # I4: no members before the first snapshot -> no entries, no crash
    bars = make_bars()
    late = Universe([(DAYS[5].date(), "A", "add"), (DAYS[5].date(), "B", "add")])
    r = run(bars, make_signals(bars, entries=[(1, "A"), (6, "B")]), universe=late)
    assert list(r.trades.symbol) == ["B"]


def test_upper_circuit_entry_skipped():
    bars = make_bars()
    set_bar(bars, 2, "A", open=110, high=110, low=110, close=110)
    r = run(bars, make_signals(bars, entries=[(1, "A")]))
    assert r.trades.empty and list(r.skipped.reason) == ["upper_circuit"]


def test_kill_switch_trip_counted_not_enforced():
    bars = make_bars()
    for d in range(3, 12):
        set_bar(bars, d, "A", open=50, high=51, low=49, close=50)
    # 200 A @ 100.1 then A halves: equity ~ 79,967 cash + 10,000 = 89,967 -> 10.03% below the peak
    r = run(bars, make_signals(bars, entries=[(1, "A"), (6, "B")]))
    assert r.kill_switch_trips == 1
    assert "B" in set(r.trades.symbol)  # entries still allowed in a backtest


def test_deterministic():
    bars = make_bars()
    sigs = make_signals(bars, entries=[(1, "A"), (2, "B")], exits=[(6, "A")])
    a, b = run(bars, sigs), run(bars, sigs)
    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.trades, b.trades)


def test_slippage_mult_doubles_slippage():
    bars = make_bars()
    t = run(bars, make_signals(bars, entries=[(1, "A")]), slippage_mult=2.0).trades.iloc[0]
    assert t.entry_price == pytest.approx(100.2)


def test_equity_indexed_by_trading_days():
    bars = make_bars()
    r = run(bars, make_signals(bars))
    assert list(r.equity.index) == DAYS and (r.equity == 100_000).all()
