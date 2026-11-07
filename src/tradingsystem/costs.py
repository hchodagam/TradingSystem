"""Indian equity delivery cost model: STT, exchange, SEBI, stamp, DP, GST; plus slippage."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from tradingsystem.config import CostRates

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Fill:
    symbol: str
    date: date
    side: Side
    qty: int
    price: float


@dataclass(frozen=True)
class ChargeBreakdown:
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp: float = 0.0
    dp: float = 0.0
    gst: float = 0.0
    total: float = 0.0


def _round(x: float, places: int) -> float:
    q = Decimal(1).scaleb(-places)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))  # float(): numpy repr is np.float64(..)


def day_charges(fills: list[Fill], rates: CostRates) -> ChargeBreakdown:
    """Charges for a set of fills. Rounded per (symbol, date, side) group, as a contract note does."""
    turnover: dict[tuple, float] = defaultdict(float)
    for f in fills:
        turnover[(f.symbol, f.date, f.side)] += f.qty * f.price

    r = rates.rounding
    tot = defaultdict(float)
    for (_sym, _d, side), t in turnover.items():
        brokerage = rates.brokerage_delivery
        stt = t * rates.stt_pct
        exch = t * rates.exchange_txn_pct
        sebi = t * rates.sebi_pct
        stamp = t * rates.stamp_buy_pct if side == "BUY" else 0.0
        dp = rates.dp_per_sell_scrip_day if side == "SELL" else 0.0
        gst = (brokerage + exch + sebi + dp) * rates.gst_pct
        tot["stt"] += _round(stt, r.stt)
        tot["exchange"] += _round(exch, r.other)
        tot["sebi"] += _round(sebi, r.other)
        tot["stamp"] += _round(stamp, r.stamp)
        tot["dp"] += _round(dp, r.other)
        tot["gst"] += _round(gst, r.other)
        tot["brokerage"] += _round(brokerage, r.other)

    parts = {k: round(tot[k], 2) for k in ("stt", "exchange", "sebi", "stamp", "dp", "gst")}
    total = round(sum(parts.values()) + tot["brokerage"], 2)
    return ChargeBreakdown(**parts, total=total)


def apply_slippage(price: float, side: Side, rates: CostRates, mult: float = 1.0) -> float:
    s = rates.slippage_pct * mult
    return price * (1 + s) if side == "BUY" else price * (1 - s)
