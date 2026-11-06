import numpy as np
import pandas as pd

SPEC = {
    "name": "random_entry",
    "family": "test_random",
    "hypothesis": "Entries at random carry no edge (control strategy; must fail the gates)",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {"pct": {"default": 0.05, "range": [0.02, 0.10]}},
    "max_holding_days": 10,
}


def signals(bars, p):
    # deterministic per (date, symbol): no look-ahead, same answer batch or incremental
    h = pd.util.hash_pandas_object(bars.index.to_frame(index=False), index=False).to_numpy()
    u = (h % np.uint64(10_000)).astype(float) / 10_000.0
    out = pd.DataFrame(index=bars.index)
    out["entry"] = u < p.pct
    out["exit"] = False
    out["stop"] = np.nan
    return out
