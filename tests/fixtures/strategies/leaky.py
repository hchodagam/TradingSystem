import numpy as np
import pandas as pd

SPEC = {
    "name": "leaky",
    "family": "test",
    "hypothesis": "Buys when tomorrow closes higher (deliberate look-ahead)",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {"n": {"default": 1, "range": [1, 3]}},
    "max_holding_days": 1,
}


def signals(bars, p):
    close = bars["close"]
    tomorrow = close.groupby(level="symbol").shift(-1)
    out = pd.DataFrame(index=bars.index)
    out["entry"] = (tomorrow > close).astype(bool)
    out["exit"] = (tomorrow < close).astype(bool)
    out["stop"] = np.nan
    return out
