"""Probe Dhan history depth and split adjustment. Needs DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env.

    uv run python scripts/probe_dhan.py [SYMBOL ...]

Prints the earliest date Dhan returns per symbol (sets research_start in config/gates.yaml), and for each
symbol in reference/corporate_actions.csv the closes around its ex_date (shows whether Dhan pre-adjusts).
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

from dotenv import load_dotenv

from tradingsystem.config import load_config
from tradingsystem.data.adjust import load_actions
from tradingsystem.data.dhan import DhanClient


def main(argv: list[str]) -> None:
    load_dotenv()
    cfg = load_config()
    client = DhanClient(cfg.data, os.environ["DHAN_CLIENT_ID"], os.environ["DHAN_ACCESS_TOKEN"])
    symbols = argv or ["RELIANCE", "HDFCBANK", "INFY"]
    today = date.today()
    print("Earliest available daily bar:")
    for s in symbols:
        df = client.daily(s, date(2000, 1, 1), today)
        print(f"  {s:<12} {df.index.min().date() if len(df) else 'none'}   ({len(df)} bars)")
    b = client.daily(cfg.data.benchmark_symbol, date(2000, 1, 1), today, index=True)
    print(f"  {cfg.data.benchmark_symbol:<12} {b.index.min().date() if len(b) else 'none'}   ({len(b)} bars)")

    actions = load_actions(cfg.data.reference_dir / "corporate_actions.csv")
    if actions.empty:
        print("\nreference/corporate_actions.csv is empty: add one known split to check adjustment.")
    for a in actions.itertuples(index=False):
        ex = a.ex_date.date()
        df = client.daily(a.symbol, ex - timedelta(days=10), ex + timedelta(days=3))
        before = df[df.index < a.ex_date]["close"].iloc[-1]
        after = df[df.index >= a.ex_date]["close"].iloc[0]
        verdict = "ADJUSTED by Dhan" if abs(before / after - 1) < 0.5 else "RAW (not adjusted)"
        print(f"\n{a.symbol} ex {ex} factor {a.factor}: close before {before:.2f}, on ex {after:.2f} -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
