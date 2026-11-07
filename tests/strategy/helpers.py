"""Build small strategy files and bars for strategy tests."""
from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

from synthetic import make_bars

from tradingsystem.strategy.loader import LoadedStrategy, load_strategy

HEADER = '''import pandas as pd
import numpy as np

SPEC = {
    "name": "t",
    "family": "test",
    "hypothesis": "test strategy",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {"n": {"default": 5, "range": [2, 10]}},
    "max_holding_days": 5,
}

'''

CLEAN_BODY = '''
def signals(bars, p):
    close = bars["close"]
    ma = close.groupby(level="symbol").transform(lambda s: s.rolling(p.n).mean())
    out = pd.DataFrame(index=bars.index)
    out["entry"] = (close > ma).fillna(False).astype(bool)
    out["exit"] = (close < ma).fillna(False).astype(bool)
    out["stop"] = np.nan
    return out
'''


def strategy_from(tmp_path: Path, body: str, name: str = "s.py") -> LoadedStrategy:
    p = tmp_path / name
    p.write_text(HEADER + textwrap.dedent(body), encoding="utf-8")
    return load_strategy(p)


def small_bars(n_symbols: int = 3, years: int = 1, seed: int = 0):
    return make_bars([f"S{i}" for i in range(n_symbols)], date(2020, 1, 1), date(2020 + years - 1, 12, 31), seed=seed)
