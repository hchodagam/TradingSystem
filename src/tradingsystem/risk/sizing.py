"""Position sizing and the upper-circuit skip rule."""
from __future__ import annotations

import math

from tradingsystem.config import RiskConfig

_EPS = 1e-9


def size_position(entry: float, stop: float, atr: float, equity: float, cfg: RiskConfig) -> int:
    """Shares = floor(risk% x equity / (entry - stop)), capped at max_position_pct of equity.
    A missing or unusable stop (NaN, or >= entry) is replaced by entry - atr_stop_mult x ATR, for sizing only.
    Returns 0 when the skip rules hit: < 1 share or position value < min_position_value."""
    if not (equity > 0 and entry > 0):
        return 0
    if not (math.isfinite(stop) and stop < entry):
        if not (math.isfinite(atr) and atr > 0):
            return 0
        stop = entry - cfg.atr_stop_mult * atr
    per_share = entry - stop
    if per_share <= 0:
        return 0
    by_risk = math.floor(cfg.risk_per_trade_pct * equity / per_share + _EPS)
    by_cap = math.floor(cfg.max_position_pct * equity / entry + _EPS)
    qty = min(by_risk, by_cap)
    if qty < 1 or qty * entry < cfg.min_position_value:
        return 0
    return qty


def is_upper_circuit(open_: float, prev_close: float, cfg: RiskConfig) -> bool:
    """Proxy: Dhan daily bars carry no circuit band, so a gap-up of upper_circuit_gap_pct or more counts."""
    return open_ >= prev_close * (1 + cfg.upper_circuit_gap_pct) * (1 - _EPS)
