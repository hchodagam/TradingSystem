import numpy as np
import pandas as pd

SPEC = {
    "name": "bad_output_param",
    "family": "test",
    "hypothesis": "Well-formed at the default; drops the stop column when n > 5 (grid-only defect)",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {"n": {"default": 3, "range": [1, 10]}},
    "max_holding_days": 5,
}


def signals(bars, p):
    close = bars["close"]
    ma = close.groupby(level="symbol").transform(lambda s: s.rolling(int(p.n)).mean())
    out = pd.DataFrame(index=bars.index)
    out["entry"] = (close > ma).astype(bool)
    out["exit"] = (close < ma).astype(bool)
    out["stop"] = np.nan
    if p.n > 5:
        out = out.drop(columns=["stop"])
    return out
