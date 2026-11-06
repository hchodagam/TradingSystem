"""Point-in-time index membership and sector map."""
from __future__ import annotations

import bisect
import csv
from datetime import date, datetime
from pathlib import Path


class Universe:
    """Built from a change log: rows of (date, symbol, add|remove). The first date is the initial snapshot."""

    def __init__(self, changes: list[tuple[date, str, str]]):
        if not changes:
            raise ValueError("universe change log is empty")
        changes = sorted(changes, key=lambda c: (c[0], c[2] != "remove"))  # removes before adds on a day
        self._dates: list[date] = []
        self._snapshots: list[frozenset[str]] = []
        current: set[str] = set()
        for d, sym, action in changes:
            if action == "add":
                current.add(sym)
            elif action == "remove":
                current.discard(sym)
            else:
                raise ValueError(f"bad action {action!r} for {sym} on {d}")
            if self._dates and self._dates[-1] == d:
                self._snapshots[-1] = frozenset(current)
            else:
                self._dates.append(d)
                self._snapshots.append(frozenset(current))
        self._all = frozenset(s for _, s, _ in changes)

    @classmethod
    def from_csv(cls, path: Path) -> Universe:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [(date.fromisoformat(r["date"]), r["symbol"].strip(), r["action"].strip()) for r in csv.DictReader(f)]
        return cls(rows)

    def members(self, d: date) -> frozenset[str]:
        if isinstance(d, datetime):  # includes pd.Timestamp
            d = d.date()
        i =bisect.bisect_right(self._dates, d) - 1
        if i < 0:
            raise ValueError(f"{d} is before the first universe snapshot {self._dates[0]}")
        return self._snapshots[i]

    def all_symbols(self) -> frozenset[str]:
        return self._all

    @property
    def first_date(self) -> date:
        return self._dates[0]


def load_sectors(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["symbol"].strip(): r["sector"].strip() for r in csv.DictReader(f)}
