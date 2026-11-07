"""ts evaluate <strategy.py>: data -> validation -> acceptance -> backtest -> Gate 1 -> Gate 2 [-> Gate 3].

Exit codes: 0 every gate run passed, 1 a gate failed, 2 data or acceptance error.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tradingsystem.backtest.engine import BacktestConfig, run_backtest
from tradingsystem.config import AppConfig, load_config
from tradingsystem.strategy.acceptance import accept, check_output
from tradingsystem.strategy.sandbox import SandboxError, run_signals
from tradingsystem.universe import Universe, load_sectors
from tradingsystem.validate import GateResult
from tradingsystem.validate.gate1 import gate1
from tradingsystem.validate.gate2 import BacktestContext, gate2
from tradingsystem.validate.gate3 import HoldoutAlreadyUsed, HoldoutDataInvalid, HoldoutLedger, gate3

SYNTH_SYMBOLS = [f"S{i:02d}" for i in range(20)]
SYNTH_END = date(2024, 6, 28)


class _Stop(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _print_gate(name: str, g: GateResult) -> None:
    print(f"\n{name} {'PASS' if g.passed else 'FAIL'}")
    print(f"  {'check':<26}{'value':>12}{'threshold':>12}  result")
    for c in g.checks:
        print(f"  {c.name:<26}{c.value:>12.4g}{c.threshold:>12.4g}  {'PASS' if c.passed else 'FAIL'}")
    for n in g.notes:
        print(f"  note: {n}")


def _issues(report, limit: int = 20) -> str:
    lines = "\n".join(f"  {i.kind}: {i.detail}" for i in report.issues[:limit])
    more = f"\n  ... and {len(report.issues) - limit} more" if len(report.issues) > limit else ""
    return lines + more + ("\n  (acknowledge checked issues in reference/data_exceptions.csv: "
                           "symbol,date,kind,note)")


class _SymbolStore:
    """BarStore restricted to the symbols that have data (see data_symbols); same read interface."""

    def __init__(self, store, symbols: list[str]):
        self._store, self._symbols = store, symbols

    def research_bars(self, symbols):
        return self._store.research_bars([s for s in symbols if s in set(self._symbols)])

    def holdout_bars(self, symbols):
        return self._store.holdout_bars([s for s in symbols if s in set(self._symbols)])

    def benchmark(self, start, end):
        return self._store.benchmark(start, end)


def _setup(cfg: AppConfig, args) -> tuple:
    if args.synthetic_seed is not None:
        from tradingsystem.synthetic import SyntheticStore, make_bars
        bars = make_bars(SYNTH_SYMBOLS, cfg.gates.research_start, SYNTH_END, seed=args.synthetic_seed,
                         drift=-0.002 if args.synthetic_edge else 0.0, planted_edge=args.synthetic_edge)
        uni = Universe([(cfg.gates.research_start, s, "add") for s in SYNTH_SYMBOLS])
        return SyntheticStore(bars, cfg.gates.research_start, cfg.gates.research_end), uni, \
            {s: s for s in SYNTH_SYMBOLS}, None

    from dotenv import load_dotenv

    from tradingsystem.data.checks import data_symbols, load_exceptions, validate_bars
    from tradingsystem.data.dhan import DhanAuthError, DhanClient, load_security_overrides
    from tradingsystem.data.store import BarStore, UnknownSymbolsError

    ref = Path(cfg.data.reference_dir)
    try:
        uni = Universe.from_csv(ref / "nifty100_changes.csv")
    except ValueError as e:
        raise _Stop(2, f"{ref / 'nifty100_changes.csv'}: {e} (fill it from NSE index change notices)") from e
    sectors = load_sectors(ref / "sectors.csv")
    load_dotenv()
    try:
        client = DhanClient(cfg.data, os.environ["DHAN_CLIENT_ID"], os.environ["DHAN_ACCESS_TOKEN"],
                            security_overrides=load_security_overrides(ref / "security_ids.csv"))
    except KeyError as e:
        raise _Stop(2, f"{e.args[0]} missing from .env") from e
    store = BarStore(cfg, client)
    exceptions = load_exceptions(ref / "data_exceptions.csv")
    symbols = data_symbols(uni, exceptions)
    try:
        store.refresh(symbols, date.today())
    except DhanAuthError as e:
        raise _Stop(2, f"Dhan: {e}") from e
    except UnknownSymbolsError as e:
        raise _Stop(2, str(e)) from e
    from tradingsystem.data.adjust import load_actions
    actions_path = ref / "corporate_actions.csv"
    actions = load_actions(actions_path) if actions_path.exists() else None
    start = max(cfg.gates.research_start, uni.first_date)
    report = validate_bars(store.research_bars(symbols), uni, start, cfg.gates.research_end, actions,
                           cfg.data.abnormal_move_pct, exceptions)
    if not report.ok:
        raise _Stop(2, f"research data failed validation ({len(report.issues)} issues):\n{_issues(report)}")

    def validate_holdout(bars: pd.DataFrame):
        last = bars.index.get_level_values("date").max().date()
        return validate_bars(bars, uni, cfg.gates.holdout_start, last, actions, cfg.data.abnormal_move_pct,
                             exceptions)

    return _SymbolStore(store, symbols), uni, sectors, validate_holdout


def evaluate(args) -> int:
    cfg = load_config(Path(args.config))
    path = Path(args.strategy)
    if not path.exists():
        raise _Stop(2, f"{path}: file not found")
    store, uni, sectors, validate_holdout = _setup(cfg, args)
    bars = store.research_bars(uni.all_symbols())

    smoke_from = pd.Timestamp(cfg.gates.research_end - timedelta(days=730))
    smoke = bars[bars.index.get_level_values("date") >= smoke_from]
    acc = accept(path, smoke)
    if not acc.ok:
        raise _Stop(2, f"{path.name}: rejected at {acc.stage}\n" + "\n".join(f"  {e}" for e in acc.errors))
    s = acc.strategy
    print(f"{s.spec.name}  hash {s.hash[:12]}  ({len(s.spec.params)} params"
          f"{', OVERFIT RISK' if s.spec.overfit_risk else ''})")

    ctx = BacktestContext(uni, sectors, store.benchmark(cfg.gates.research_start, cfg.gates.research_end))
    try:
        sig = run_signals(s, s.spec.defaults(), bars)
    except SandboxError as e:
        raise _Stop(2, f"signals failed on the research period: {e}") from e
    errs = check_output(sig, bars)
    if errs:
        raise _Stop(2, "signals() output is malformed on the full research period:\n"
                    + "\n".join(f"  {e}" for e in errs))
    bt =BacktestConfig(cfg.gates.research_start, cfg.gates.research_end, cfg.risk.initial_capital)
    result = run_backtest(sig, bars, uni, sectors, s.spec, cfg, bt)
    print(f"research backtest: {len(result.trades)} trades, final equity {result.equity.iloc[-1]:,.0f}, "
          f"kill-switch trips {result.kill_switch_trips}")
    g1 = gate1(result, ctx.benchmark, cfg.gates)
    _print_gate("Gate 1", g1)
    if not g1.passed:
        return 1
    g2 = gate2(s, bars, ctx, cfg, family_trials=args.family_trials, family_sr_var=None)
    _print_gate("Gate 2", g2)
    if not g2.passed:
        return 1
    if not args.holdout:
        print("\nGates 1-2 passed. Re-run with --holdout to spend the single holdout run.")
        return 0
    research_sharpe = next(c.value for c in g1.checks if c.name == "sharpe")
    ledger = HoldoutLedger(Path(args.state_dir) / "holdout_runs.json")
    try:
        g3 = gate3(s, store, ctx, cfg, research_sharpe, True, ledger, validate=validate_holdout)
    except HoldoutAlreadyUsed as e:
        raise _Stop(2, str(e)) from e
    except HoldoutDataInvalid as e:
        raise _Stop(2, f"{e}:\n{_issues(e.report)}") from e
    _print_gate("Gate 3", g3)
    return 0 if g3.passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ts")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="run a strategy file through acceptance, backtest and gates")
    e.add_argument("strategy")
    e.add_argument("--holdout", action="store_true", help="also run Gate 3 (once per strategy hash)")
    e.add_argument("--family-trials", type=int, default=1, help="trials so far in this strategy family (DSR)")
    e.add_argument("--config", default="config")
    e.add_argument("--state-dir", default="state")
    e.add_argument("--synthetic-seed", type=int, default=None, help=argparse.SUPPRESS)
    e.add_argument("--synthetic-edge", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    try:
        return evaluate(args)
    except _Stop as stop:
        print(str(stop))
        return stop.code
    except Exception as e:  # noqa: BLE001 - exit 1 is reserved for "a gate failed"
        print(f"error: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
