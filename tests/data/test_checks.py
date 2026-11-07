from datetime import date

import pandas as pd
import pytest

from tradingsystem.calendar import trading_days
from tradingsystem.data.checks import validate_bars
from tradingsystem.universe import Universe

START, END = date(2024, 1, 1), date(2024, 1, 12)
U = Universe([(date(2023, 1, 1), "A", "add"), (date(2023, 1, 1), "B", "add")])
NO_ACTIONS = pd.DataFrame({"symbol": [], "ex_date": pd.to_datetime([]), "factor": []})


def clean_bars():
    days = trading_days(START, END)
    idx = pd.MultiIndex.from_product([days, ["A", "B"]], names=["date", "symbol"])
    return pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000.0}, index=idx)


def kinds(report):
    return {i.kind for i in report.issues}


def run(bars, actions=NO_ACTIONS):
    return validate_bars(bars, U, START, END, actions, abnormal_move_pct=0.20)


def test_clean_is_ok():
    r = run(clean_bars())
    assert r.ok is True and r.issues == []


def test_missing_symbol():
    b = clean_bars().drop((pd.Timestamp("2024-01-05"), "B"))
    r = run(b)
    assert not r.ok and kinds(r) == {"missing_symbol"}
    assert r.issues[0].symbol == "B" and r.issues[0].date == pd.Timestamp("2024-01-05")


def test_missing_day():
    b = clean_bars().drop(pd.Timestamp("2024-01-08"), level="date")
    assert kinds(run(b)) == {"missing_day"}


def test_ohlc_invalid():
    b = clean_bars()
    b.loc[(pd.Timestamp("2024-01-03"), "A"), "high"] = 99.0
    assert kinds(run(b)) == {"ohlc_invalid"}


def test_non_positive():
    b = clean_bars()
    b.loc[(pd.Timestamp("2024-01-03"), "A"), ["open", "high", "low", "close"]] = 0.0
    assert "non_positive" in kinds(run(b))


def test_abnormal_move():
    b = clean_bars()
    b.loc[(pd.Timestamp("2024-01-04"), "A"), ["open", "high", "low", "close"]] = [130.0, 130.0, 125.0, 127.0]
    r = run(b)
    assert "abnormal_move" in kinds(r)


def test_move_on_ex_date_not_flagged():
    b = clean_bars()
    dates, syms = b.index.get_level_values("date"), b.index.get_level_values("symbol")
    after = (dates >= pd.Timestamp("2024-01-04")) & (syms == "A")
    b.loc[after, ["open", "high", "low", "close"]] = [75.0, 77.0, 74.0, 75.0]
    actions = pd.DataFrame({"symbol": ["A"], "ex_date": [pd.Timestamp("2024-01-04")], "factor": [1.0]})
    assert "abnormal_move" not in kinds(run(b, actions))


def test_acknowledged_exceptions_are_suppressed(tmp_path):
    # I7: the owner can acknowledge a genuine move or suspension in reference/data_exceptions.csv
    from tradingsystem.data.checks import load_exceptions
    b = clean_bars().drop((pd.Timestamp("2024-01-05"), "B"))
    b.loc[(pd.Timestamp("2024-01-04"), "A"), ["open", "high", "low", "close"]] = [130.0, 130.0, 125.0, 127.0]
    p = tmp_path / "data_exceptions.csv"
    p.write_text("symbol,date,kind,note\nB,2024-01-05,missing_symbol,suspended\nA,2024-01-04,abnormal_move,results\n")
    r = validate_bars(b, U, START, END, NO_ACTIONS, abnormal_move_pct=0.20, exceptions=load_exceptions(p))
    # A's next-day move back down is still flagged: only the acknowledged (symbol, date, kind) is suppressed
    assert {(i.kind, i.symbol, i.date) for i in r.issues} == {
        ("abnormal_move", "A", pd.Timestamp("2024-01-05"))}


def test_exception_with_star_date_covers_every_date(tmp_path):
    # a delisted former member with no data at all is acknowledged once, not once per day
    from tradingsystem.data.checks import load_exceptions
    b = clean_bars().drop("B", level="symbol")
    p = tmp_path / "data_exceptions.csv"
    p.write_text("symbol,date,kind,note\nB,*,missing_symbol,delisted; no Dhan data\n")
    r = validate_bars(b, U, START, END, NO_ACTIONS, abnormal_move_pct=0.20, exceptions=load_exceptions(p))
    assert r.ok, r.issues


def test_missing_day_can_be_acknowledged_with_blank_symbol(tmp_path):
    from tradingsystem.data.checks import load_exceptions
    b = clean_bars().drop(pd.Timestamp("2024-01-08"), level="date")
    p = tmp_path / "data_exceptions.csv"
    p.write_text("symbol,date,kind,note\n,2024-01-08,missing_day,exchange holiday missing from the calendar\n")
    assert validate_bars(b, U, START, END, NO_ACTIONS, abnormal_move_pct=0.20, exceptions=load_exceptions(p)).ok


def test_data_symbols_skips_acknowledged_no_data_symbols():
    from tradingsystem.data.checks import ALL_DATES, data_symbols
    exc = {("B", ALL_DATES, "missing_symbol"), ("A", pd.Timestamp("2024-01-03"), "missing_symbol")}
    assert data_symbols(U, exc) == ["A"]
    assert data_symbols(U, set()) == ["A", "B"]


def test_issue_detail_is_human_readable():
    b = clean_bars().drop((pd.Timestamp("2024-01-05"), "B"))
    assert "B" in run(b).issues[0].detail


@pytest.mark.parametrize("col", ["open", "close"])
def test_nan_price_is_invalid(col):
    b = clean_bars()
    b.loc[(pd.Timestamp("2024-01-03"), "A"), col] = float("nan")
    assert not run(b).ok
