"""Bar validation: missing symbols/days, OHLC sanity, non-positive prices, abnormal moves."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from tradingsystem.calendar import trading_days
from tradingsystem.universe import Universe

IssueKind = Literal["missing_symbol", "missing_day", "ohlc_invalid", "abnormal_move", "non_positive"]


@dataclass(frozen=True)
class Issue:
    kind: IssueKind
    symbol: str | None
    date: pd.Timestamp
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    issues: list[Issue] = field(default_factory=list)


Exceptions = set[tuple[str, pd.Timestamp, str]]


def load_exceptions(path) -> Exceptions:
    """reference/data_exceptions.csv: symbol,date,kind[,note] - issues the owner has checked and accepted."""
    from pathlib import Path
    if not Path(path).exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {(r.symbol.strip(), pd.Timestamp(r.date), r.kind.strip()) for r in df.itertuples(index=False)}


def validate_bars(bars: pd.DataFrame, universe: Universe, start: date, end: date, actions: pd.DataFrame,
                  abnormal_move_pct: float, exceptions: Exceptions | None = None) -> ValidationReport:
    report = _validate(bars, universe, start, end, actions, abnormal_move_pct)
    if not exceptions:
        return report
    kept = [i for i in report.issues if (i.symbol, i.date, i.kind) not in exceptions]
    return ValidationReport(ok=not kept, issues=kept)


def _validate(bars: pd.DataFrame, universe: Universe, start: date, end: date, actions: pd.DataFrame,
              abnormal_move_pct: float) -> ValidationReport:
    issues: list[Issue] = []
    present = bars.index.get_level_values("date")
    by_day = {d: set(g) for d, g in bars.reset_index().groupby("date")["symbol"]}

    for d in trading_days(start, end):
        syms = by_day.get(d)
        if not syms:
            issues.append(Issue("missing_day", None, d, f"no bars at all on {d.date()}"))
            continue
        members = universe.members(d) if d.date() >= universe.first_date else frozenset()
        for s in sorted(members - syms):
            issues.append(Issue("missing_symbol", s, d, f"{s} is an index member but has no bar on {d.date()}"))

    in_range = bars[(present >= pd.Timestamp(start)) & (present <= pd.Timestamp(end))]
    o, h, lo, c = (in_range[k] for k in ("open", "high", "low", "close"))
    prices = in_range[["open", "high", "low", "close"]]
    nonpos = (prices <= 0).any(axis=1)
    invalid = ~((lo <= np.minimum(o, c)) & (np.maximum(o, c) <= h)) | prices.isna().any(axis=1)
    for (d, s) in in_range.index[nonpos]:
        issues.append(Issue("non_positive", s, d, f"{s} has a price <= 0 on {d.date()}"))
    for (d, s) in in_range.index[invalid & ~nonpos]:
        r = in_range.loc[(d, s)]
        issues.append(Issue("ohlc_invalid", s, d,
                            f"{s} on {d.date()}: O={r.open} H={r.high} L={r.low} C={r.close} is not low<=open,close<=high"))

    ex = set()
    if actions is not None and len(actions):
        ex = {(r.symbol, pd.Timestamp(r.ex_date)) for r in actions.itertuples(index=False)}
    prev = bars["close"].groupby(level="symbol").shift(1)
    move = (bars["close"] / prev - 1).abs()
    flagged = move[(move > abnormal_move_pct) & (present >= pd.Timestamp(start)) & (present <= pd.Timestamp(end))]
    for (d, s), m in flagged.items():
        if (s, d) not in ex:
            issues.append(Issue("abnormal_move", s, d, f"{s} moved {m:.1%} on {d.date()} with no corporate action"))

    return ValidationReport(ok=not issues, issues=issues)
