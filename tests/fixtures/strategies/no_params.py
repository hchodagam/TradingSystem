import numpy as np
import pandas as pd

SPEC = {
    "name": "no_params",
    "family": "test",
    "hypothesis": "Parameter-free variant of the dip buyer",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {},
    "max_holding_days": 4,
}


def signals(bars, p):
    ret = bars["close"].groupby(level="symbol").pct_change()
    out = pd.DataFrame(index=bars.index)
    out["entry"] = (ret < -0.035).fillna(False).astype(bool)
    out["exit"] = False
    out["stop"] = np.nan
    return out
