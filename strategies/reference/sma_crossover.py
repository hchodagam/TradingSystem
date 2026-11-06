import numpy as np
import pandas as pd

SPEC = {
    "name": "sma_crossover",
    "family": "trend_following",
    "hypothesis": "When the 50-day average crosses above the 200-day average a stock has entered a durable "
                  "uptrend that tends to persist; the opposite cross marks its end.",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {
        "fast": {"default": 50, "range": [20, 100]},
        "slow": {"default": 200, "range": [100, 300]},
    },
    "max_holding_days": 250,
}


def signals(bars, p):
    close = bars["close"]
    by = close.groupby(level="symbol")
    fast = by.transform(lambda s: s.rolling(int(p.fast)).mean())
    slow = by.transform(lambda s: s.rolling(int(p.slow)).mean())
    above = (fast > slow).astype(bool)
    prev = above.groupby(level="symbol").shift(1).fillna(False).astype(bool)
    valid = (fast.notna() & slow.notna()).astype(bool)
    out = pd.DataFrame(index=bars.index)
    out["entry"] = (above & ~prev & valid).astype(bool)
    out["exit"] = (~above & prev & valid).astype(bool)
    out["stop"] = np.nan
    return out
