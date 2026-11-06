"""Portfolio caps on new entries: max open positions, max per sector, no pyramiding."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from tradingsystem.config import RiskConfig


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    qty: int
    stop: float
    signal_date: pd.Timestamp


@dataclass
class Holding:
    symbol: str
    qty: int
    entry_price: float
    stop: float
    entry_date: pd.Timestamp


def admit(intents: list[OrderIntent], holdings: dict[str, Holding], sectors: dict[str, str],
          cfg: RiskConfig) -> tuple[list[OrderIntent], list[tuple[OrderIntent, str]]]:
    """Process intents in the given order. Returns (accepted, [(rejected, reason)])."""
    held = set(holdings)
    per_sector = Counter(sectors.get(s) for s in held)
    accepted: list[OrderIntent] = []
    rejected: list[tuple[OrderIntent, str]] = []
    for it in intents:
        sector = sectors.get(it.symbol)
        if it.symbol in held:
            rejected.append((it, "already_held"))
        elif sector is None:
            rejected.append((it, "unknown_sector"))
        elif len(held) >= cfg.max_open_positions:
            rejected.append((it, "max_positions"))
        elif per_sector[sector] >= cfg.max_per_sector:
            rejected.append((it, "max_per_sector"))
        else:
            accepted.append(it)
            held.add(it.symbol)
            per_sector[sector] += 1
    return accepted, rejected
