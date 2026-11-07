import pandas as pd

from tradingsystem.data.adjust import adjust


def _bars():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]], names=["date", "symbol"])
    df = pd.DataFrame({"open": 1000.0, "high": 1000.0, "low": 1000.0, "close": 1000.0, "volume": 10.0}, index=idx)
    df.loc[(pd.Timestamp("2024-01-03"), "A"), ["open", "high", "low", "close"]] = 200.0
    return df


def test_split_adjustment():
    actions = pd.DataFrame({"symbol": ["A"], "ex_date": [pd.Timestamp("2024-01-03")], "factor": [5.0]})
    out = adjust(_bars(), actions)
    a = out.xs("A", level="symbol")
    assert (a["close"] == 200.0).all()
    assert list(a["volume"]) == [50.0, 50.0, 10.0]
    b = out.xs("B", level="symbol")
    assert (b["close"] == 1000.0).all() and (b["volume"] == 10.0).all()


def test_compounding_actions():
    actions = pd.DataFrame({"symbol": ["A", "A"], "ex_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                            "factor": [2.0, 5.0]})
    a = adjust(_bars(), actions).xs("A", level="symbol")
    assert list(a["close"]) == [100.0, 200.0, 200.0]


def test_no_actions_is_identity():
    bars = _bars()
    empty = pd.DataFrame({"symbol": [], "ex_date": pd.to_datetime([]), "factor": []})
    pd.testing.assert_frame_equal(adjust(bars, empty), bars)


def test_input_not_mutated():
    bars = _bars()
    before = bars.copy()
    adjust(bars, pd.DataFrame({"symbol": ["A"], "ex_date": [pd.Timestamp("2024-01-03")], "factor": [5.0]}))
    pd.testing.assert_frame_equal(bars, before)
