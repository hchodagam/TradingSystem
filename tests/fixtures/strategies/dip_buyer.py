import numpy as np
import pandas as pd

SPEC = {
    "name": "dip_buyer",
    "family": "test_dip",
    "hypothesis": "After a sharp one-day drop the stock rebounds over the following days "
                  "(matches the edge planted by tests/synthetic.py)",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {
        "drop": {"default": 0.035, "range": [0.03, 0.06]},
        "hold": {"default": 4, "range": [3, 5]},
    },
    "max_holding_days": 4,
}


def signals(bars, p):
    close = bars["close"]
    ret = close.groupby(level="symbol").pct_change()
    entry = (ret < -p.drop).fillna(False).astype(bool)
    held = entry.groupby(level="symbol").transform(lambda s: s.rolling(int(p.hold), min_periods=1).sum())
    out = pd.DataFrame(index=bars.index)
    out["entry"] = entry
    out["exit"] = (held.fillna(0) == 0).astype(bool) & ~entry
    out["stop"] = np.nan
    return out
