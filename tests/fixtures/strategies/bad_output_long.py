import numpy as np
import pandas as pd

SPEC = {
    "name": "bad_output_long",
    "family": "test",
    "hypothesis": "Well-formed on short history, drops the stop column on long history",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {"n": {"default": 3, "range": [1, 10]}},
    "max_holding_days": 5,
}


def signals(bars, p):
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)
    out["entry"] = False
    out["exit"] = False
    out["stop"] = np.nan
    if len(close.index.get_level_values("date").unique()) > 600:
        out = out.drop(columns=["stop"])
    return out
