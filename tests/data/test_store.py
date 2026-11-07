from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tradingsystem.calendar import trading_days
from tradingsystem.config import load_config
from tradingsystem.data.store import BarStore, UnknownSymbolsError

BASE = load_config(Path(__file__).parents[2] / "config")


class FakeClient:
    def __init__(self):
        self.calls = []

    def daily(self, symbol, start, end, index=False):
        self.calls.append((symbol, start, end, index))
        days = trading_days(start, end)
        return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
                            index=pd.DatetimeIndex(days, name="date"))


@pytest.fixture
def cfg(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir()
    (ref / "corporate_actions.csv").write_text("symbol,ex_date,factor\n")
    data = BASE.data.model_copy(update={"cache_dir": tmp_path / "cache", "reference_dir": ref})
    gates = BASE.gates.model_copy(update={"research_start": date(2022, 11, 1), "research_end": date(2022, 12, 31),
                                          "holdout_start": date(2023, 1, 1)})
    return BASE.model_copy(update={"data": data, "gates": gates})


def test_store_fetches_only_missing_dates(cfg):
    client = FakeClient()
    store = BarStore(cfg, client)
    store.refresh(["A"], date(2023, 1, 31), include_benchmark=False)
    assert len(client.calls) == 1 and client.calls[0][1] == date(2022, 11, 1)
    store.refresh(["A"], date(2023, 1, 31), include_benchmark=False)
    assert len(client.calls) == 1
    store.refresh(["A"], date(2023, 2, 3), include_benchmark=False)
    assert len(client.calls) == 2 and client.calls[1][1] == date(2023, 2, 1)


def test_weekend_end_does_not_refetch(cfg):
    client = FakeClient()
    store = BarStore(cfg, client)
    store.refresh(["A"], date(2023, 2, 3), include_benchmark=False)  # Friday
    store.refresh(["A"], date(2023, 2, 5), include_benchmark=False)  # Sunday
    assert len(client.calls) == 1


def test_research_bars_never_include_holdout(cfg):
    store = BarStore(cfg, FakeClient())
    store.refresh(["A", "B"], date(2023, 3, 31), include_benchmark=False)
    bars = store.research_bars(["A", "B"])
    dates = bars.index.get_level_values("date")
    assert dates.max() <= pd.Timestamp(cfg.gates.research_end)
    assert dates.min() >= pd.Timestamp(cfg.gates.research_start)
    assert bars.index.names == ["date", "symbol"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.is_monotonic_increasing


def test_holdout_bars_extend_past_research_end(cfg):
    store = BarStore(cfg, FakeClient())
    store.refresh(["A"], date(2023, 3, 31), include_benchmark=False)
    assert store.holdout_bars(["A"]).index.get_level_values("date").max() > pd.Timestamp(cfg.gates.research_end)


def test_benchmark_series(cfg):
    client = FakeClient()
    store = BarStore(cfg, client)
    store.refresh([], date(2023, 1, 31))
    assert client.calls[0][3] is True
    s = store.benchmark(date(2022, 12, 1), date(2022, 12, 31))
    assert isinstance(s, pd.Series) and s.index.min() >= pd.Timestamp("2022-12-01")


def test_refresh_reports_all_unknown_symbols(cfg):
    # I5: fetch what we can, then name every symbol Dhan does not know
    class Partial(FakeClient):
        def daily(self, symbol, start, end, index=False):
            if symbol in ("OLD1", "OLD2"):
                raise KeyError(symbol)
            return super().daily(symbol, start, end, index)

    client = Partial()
    store = BarStore(cfg, client)
    with pytest.raises(UnknownSymbolsError) as e:
        store.refresh(["A", "OLD1", "OLD2"], date(2023, 1, 31), include_benchmark=False)
    assert e.value.symbols == ["OLD1", "OLD2"]
    assert len(store.research_bars(["A"])) > 0


def test_bar_fixes_drop_listed_bars(cfg):
    store = BarStore(cfg, FakeClient())
    store.refresh(["A", "B"], date(2022, 12, 31), include_benchmark=False)
    (Path(cfg.data.reference_dir) / "bar_fixes.csv").write_text(
        "symbol,date,action,note\nA,2022-11-02,drop,stray bar at wrong price basis\n")
    bars = store.research_bars(["A", "B"])
    assert (pd.Timestamp("2022-11-02"), "A") not in bars.index
    assert (pd.Timestamp("2022-11-02"), "B") in bars.index
    assert (pd.Timestamp("2022-11-03"), "A") in bars.index


def test_bar_fixes_unknown_action_rejected(cfg):
    store = BarStore(cfg, FakeClient())
    store.refresh(["A"], date(2022, 12, 31), include_benchmark=False)
    (Path(cfg.data.reference_dir) / "bar_fixes.csv").write_text("symbol,date,action,note\nA,2022-11-02,shift,x\n")
    with pytest.raises(ValueError, match="action"):
        store.research_bars(["A"])


def test_missing_symbol_raises(cfg):
    with pytest.raises(FileNotFoundError):
        BarStore(cfg, None).research_bars(["ZZZ"])
