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
    day = bars.index.get_level_values("date").to_numpy().astype("datetime64[D]").astype(np.int64)
    sym = np.asarray(bars.index.codes[1], dtype=np.int64)
    u = ((day * 2654435761 + sym * 40503) % 10007).astype(float) / 10007.0
    out = pd.DataFrame(index=bars.index)
    out["entry"] = u < p.pct
    out["exit"] = False
    out["stop"] = np.nan
    return out
