"""Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)."""
from __future__ import annotations

import math

from scipy.stats import norm

EULER_GAMMA = 0.5772156649


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_var: float) -> float:
    """Probability the true Sharpe exceeds the best expected among n_trials unskilled trials.
    sr is per-period (not annualised); kurt is Pearson kurtosis (normal = 3)."""
    if n_obs < 2:
        return 0.0
    n = max(int(n_trials), 1)
    if n == 1 or sr_var <= 0:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(sr_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                   + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e)))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return 0.0
    return float(norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)))
