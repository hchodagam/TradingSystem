"""Split/bonus adjustment. factor = new shares per old share (1:5 split -> 5, 1:1 bonus -> 2).
Dividends are not adjusted."""
from __future__ import annotations

import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]


def adjust(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    if actions is None or actions.empty:
        return out
    dates = out.index.get_level_values("date")
    syms = out.index.get_level_values("symbol")
    for row in actions.itertuples(index=False):
        mask = (syms == row.symbol) & (dates < pd.Timestamp(row.ex_date))
        if mask.any():
            out.loc[mask, PRICE_COLS] = out.loc[mask, PRICE_COLS] / float(row.factor)
            out.loc[mask, "volume"] = out.loc[mask, "volume"] * float(row.factor)
    return out


def load_actions(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["factor"] = df["factor"].astype(float)
    return df
