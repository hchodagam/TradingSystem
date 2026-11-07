"""Performance metrics on daily data."""
from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS = 252


def sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe (sqrt 252), sample std; rf is an annual rate. 0.0 when undefined."""
    r = pd.Series(daily_returns, dtype=float).dropna() - rf / TRADING_DAYS
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if not sd or not math.isfinite(sd) or sd < 1e-15:
        return 0.0
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS))


def per_period_sharpe(daily_returns: pd.Series) -> float:
    return sharpe(daily_returns) / math.sqrt(TRADING_DAYS)


def returns(equity: pd.Series) -> pd.Series:
    return pd.Series(equity, dtype=float).pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    e = pd.Series(equity, dtype=float)
    if e.empty:
        return 0.0
    return float((1 - e / e.cummax()).max())


def profit_factor(pnl: pd.Series) -> float:
    p = pd.Series(pnl, dtype=float)
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    if losses == 0:
        return math.inf if wins > 0 else 0.0
    return float(wins / losses)


def total_return(equity: pd.Series) -> float:
    e = pd.Series(equity, dtype=float)
    return float(e.iloc[-1] / e.iloc[0] - 1) if len(e) > 1 else 0.0
