import numpy as np
import pandas as pd

SPEC = {
    "name": "leaky_param",
    "family": "test",
    "hypothesis": "Clean at the default lag 0; looks lag days ahead otherwise (look-ahead hidden in the grid)",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {"lag": {"default": 0, "range": [0, 10]}},
    "max_holding_days": 5,
}


def signals(bars, p):
    close = bars["close"]
    if int(p.lag) == 0:
        ref = close.groupby(level="symbol").shift(1)
        up = close > ref
    else:
        ref = close.groupby(level="symbol").shift(-int(p.lag))
        up = ref > close
    out = pd.DataFrame(index=bars.index)
    out["entry"] = up.fillna(False).astype(bool)
    out["exit"] = False
    out["stop"] = np.nan
    return out
