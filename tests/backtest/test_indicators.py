import numpy as np
import pandas as pd
import pytest

from tradingsystem.backtest.indicators import atr


def test_atr_wilder_matches_hand_calc():
    idx = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=5), ["A"]], names=["date", "symbol"])
    bars = pd.DataFrame({
        "open": [10, 11, 12, 11, 13], "high": [11, 12, 13, 12, 15], "low": [9, 10, 11, 9, 12],
        "close": [10, 11, 12, 10, 14], "volume": 1.0}, index=idx, dtype=float)
    # TR: 2 (h-l), max(2,|12-10|,|10-10|)=2, 2, max(3,0,3)=3, max(3,|15-10|,|12-10|)=5
    # ATR3: day3 = mean(2,2,2)=2; day4 = (2*2+3)/3=7/3; day5 = (7/3*2+5)/3 = 29/9
    a = atr(bars, 3).xs("A", level="symbol")
    assert np.isnan(a.iloc[0]) and np.isnan(a.iloc[1])
    assert a.iloc[2] == pytest.approx(2.0)
    assert a.iloc[3] == pytest.approx(7 / 3)
    assert a.iloc[4] == pytest.approx(29 / 9)


def test_atr_per_symbol_independent():
    idx = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=3), ["A", "B"]], names=["date", "symbol"])
    bars = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 1.0, "close": 1.5, "volume": 1.0}, index=idx)
    bars.loc[(slice(None), "B"), "high"] = 5.0
    a = atr(bars, 2)
    assert a.loc[(pd.Timestamp("2024-01-02"), "A")] == pytest.approx(1.0)
    assert a.loc[(pd.Timestamp("2024-01-02"), "B")] == pytest.approx(4.0)
    assert a.index.equals(bars.index)
