from datetime import date

import pandas as pd

from tradingsystem.calendar import trading_days


def test_weekend_excluded():
    assert pd.Timestamp("2024-01-06") not in trading_days(date(2024, 1, 1), date(2024, 1, 31))


def test_republic_day_holiday():
    assert pd.Timestamp("2024-01-26") not in trading_days(date(2024, 1, 1), date(2024, 1, 31))


def test_regular_day_included_and_bounds_inclusive():
    days = trading_days(date(2024, 1, 2), date(2024, 1, 3))
    assert list(days) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_naive_timestamps():
    assert trading_days(date(2024, 1, 1), date(2024, 1, 10)).tz is None
