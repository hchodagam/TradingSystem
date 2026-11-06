"""Synthetic daily bars: geometric random walk on NSE trading days (tests and `ts evaluate --synthetic-seed`)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from tradingsystem.calendar import trading_days


def make_bars(symbols: list[str], start: date, end: date, seed: int, drift: float = 0.0, vol: float = 0.02,
              planted_edge: bool = False) -> pd.DataFrame:
    """Bars frame. With planted_edge, after any day closing < -3% the next 5 days drift +1%/day."""
    rng = np.random.default_rng(seed)
    days = trading_days(start, end)
    frames = []
    for sym in symbols:
        n = len(days)
        shocks = rng.normal(drift, vol, n)
        rets = np.empty(n)
        boost = 0
        for i in range(n):
            r = shocks[i] + (0.01 if boost > 0 else 0.0)
            boost = max(boost - 1, 0)
            rets[i] = r
            if planted_edge and r < -0.03:
                boost = 5
        close = 100.0 * np.exp(np.cumsum(np.log1p(rets)))
        prev = np.concatenate([[100.0], close[:-1]])
        open_ = prev * (1 + rng.normal(0, vol / 4, n))
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
        vol_ = rng.integers(100_000, 1_000_000, n).astype(float)
        frames.append(pd.DataFrame({"date": days, "symbol": sym, "open": open_, "high": hi, "low": lo,
                                    "close": close, "volume": vol_}))
    return pd.concat(frames, ignore_index=True).set_index(["date", "symbol"]).sort_index()


class SyntheticStore:
    """In-memory stand-in for BarStore with the same research/holdout clamping."""

    def __init__(self, bars: pd.DataFrame, research_start: date, research_end: date):
        self.bars = bars
        self.lo, self.hi = pd.Timestamp(research_start), pd.Timestamp(research_end)

    def _between(self, lo, hi) -> pd.DataFrame:
        d = self.bars.index.get_level_values("date")
        return self.bars[(d >= lo) & (d <= hi)]

    def refresh(self, symbols, end, include_benchmark=True) -> None:
        pass

    def research_bars(self, symbols) -> pd.DataFrame:
        return self._between(self.lo, self.hi)

    def holdout_bars(self, symbols) -> pd.DataFrame:
        return self._between(self.lo, pd.Timestamp.max)

    def benchmark(self, start, end) -> pd.Series:
        b = self.bars["close"].unstack().mean(axis=1)
        return b[(b.index >= pd.Timestamp(start)) & (b.index <= pd.Timestamp(end))]
