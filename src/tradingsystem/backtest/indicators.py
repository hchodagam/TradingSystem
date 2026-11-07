"""Indicators the engine needs (strategies compute their own)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(tr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(tr), np.nan)
    if len(tr) < n:
        return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def atr(bars: pd.DataFrame, n: int) -> pd.Series:
    """Wilder ATR per symbol, indexed like bars. The first true range is high - low."""
    parts = []
    for _sym, g in bars.groupby(level="symbol", sort=False):
        h, lo, c = g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()
        prev = np.concatenate([[np.nan], c[:-1]])
        tr = np.nanmax(np.vstack([h - lo, np.abs(h - prev), np.abs(lo - prev)]), axis=0)
        parts.append(pd.Series(_wilder(tr, n), index=g.index))
    if not parts:
        return pd.Series(dtype=float, index=bars.index)
    return pd.concat(parts).reindex(bars.index)
