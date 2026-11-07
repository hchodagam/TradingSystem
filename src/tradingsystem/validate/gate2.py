"""Gate 2: robustness. Look-ahead and output checks on every param set used, then walk-forward,
parameter sensitivity, Monte Carlo, cost stress and the deflated Sharpe."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from tradingsystem.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from tradingsystem.config import AppConfig
from tradingsystem.strategy.acceptance import check_output, compare_incremental, incremental_start
from tradingsystem.strategy.loader import LoadedStrategy
from tradingsystem.strategy.sandbox import SandboxError, run_signals_many
from tradingsystem.strategy.spec import Params, StrategySpec
from tradingsystem.universe import Universe
from tradingsystem.validate import CheckResult, GateResult
from tradingsystem.validate.dsr import deflated_sharpe
from tradingsystem.validate.grid import neighbours, param_grid
from tradingsystem.validate.metrics import per_period_sharpe, returns, sharpe, total_return


@dataclass(frozen=True)
class BacktestContext:
    universe: Universe
    sectors: dict[str, str]
    benchmark: pd.Series


def _key(p: Params) -> tuple:
    return tuple(sorted(p.items()))


def _fmt(p: Params) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(p.items())) or "(no params)"


def _compact(sig: pd.DataFrame) -> pd.DataFrame:
    """Only rows the engine acts on: keeps hundreds of param sets in memory on real data."""
    return sig[sig["entry"] | sig["exit"] | sig["stop"].notna()]


class _Runner:
    """Signals per param set (validated, compacted, computed `chunk_size` sets per sandbox process) and
    backtests per (params, slippage)."""

    def __init__(self, strategy: LoadedStrategy, bars: pd.DataFrame, ctx: BacktestContext, cfg: AppConfig,
                 start: date, end: date, chunk_size: int = 16):
        self.s, self.bars, self.ctx, self.cfg = strategy, bars, ctx, cfg
        self.start, self.end = start, end
        self.chunk_size = max(1, chunk_size)
        self._sig: dict[tuple, pd.DataFrame] = {}
        self._bt: dict[tuple, BacktestResult] = {}
        self.invalid: dict[tuple, str] = {}

    def _store(self, p: Params, out: pd.DataFrame) -> None:
        errs = check_output(out, self.bars)
        if errs:
            self.invalid[_key(p)] = f"{_fmt(p)}: {'; '.join(errs)}"
        else:
            self._sig[_key(p)] = _compact(out)

    def prefetch(self, params_list: list[Params]) -> None:
        todo = list({_key(p): p for p in params_list
                     if _key(p) not in self._sig and _key(p) not in self.invalid}.values())
        for i in range(0, len(todo), self.chunk_size):
            chunk = todo[i:i + self.chunk_size]
            try:
                outs = run_signals_many(self.s, chunk, self.bars)
            except SandboxError:
                if len(chunk) == 1:
                    raise
                for p in chunk:  # find which param set fails
                    try:
                        self._store(p, run_signals_many(self.s, [p], self.bars)[0])
                    except SandboxError as e:
                        self.invalid[_key(p)] = f"{_fmt(p)}: {e.reason}: {e.detail[-300:]}"
                continue
            for p, out in zip(chunk, outs, strict=True):
                self._store(p, out)

    def result(self, p: Params, slippage_mult: float = 1.0) -> BacktestResult:
        k = (_key(p), slippage_mult)
        if k not in self._bt:
            if _key(p) not in self._sig:
                self.prefetch([p])
            bt = BacktestConfig(self.start, self.end, self.cfg.risk.initial_capital, slippage_mult=slippage_mult)
            self._bt[k] = run_backtest(self._sig[_key(p)], self.bars, self.ctx.universe, self.ctx.sectors,
                                       self.s.spec, self.cfg, bt)
        return self._bt[k]


def range_ends(spec: StrategySpec) -> list[Params]:
    """Each param at its range low and high, others at defaults (deduped, excluding the defaults)."""
    base = spec.defaults()
    seen, out = {_key(base)}, []
    for k, p in spec.params.items():
        for v in p.range:
            q = Params(base)
            q[k] = v
            if _key(q) not in seen:
                seen.add(_key(q))
                out.append(q)
    return out


def lookahead_params(strategy: LoadedStrategy, bars: pd.DataFrame, params: list[Params],
                     window: int) -> list[str]:
    """Look-ahead check (batch vs day-by-day) for extra param sets, in two sandbox processes."""
    if not params:
        return []
    k = incremental_start(bars, window)
    batch = run_signals_many(strategy, params, bars)
    inc = run_signals_many(strategy, params, bars, incremental_from=k)
    notes = []
    for p, b, i in zip(params, batch, inc, strict=True):
        if check_output(b, bars):
            continue  # malformed output: reported by the signals_valid check
        diffs = compare_incremental(b, i, bars, k)
        if diffs:
            notes.append(f"look-ahead at {_fmt(p)}: {diffs[0]}")
    return notes


def _windows(start: date, end: date, fit: int, test: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp,
                                                                         pd.Timestamp]]:
    out = []
    y = start.year + fit
    while y + test - 1 <= end.year:
        out.append((pd.Timestamp(y - fit, 1, 1), pd.Timestamp(y - 1, 12, 31),
                    pd.Timestamp(y, 1, 1), pd.Timestamp(y + test - 1, 12, 31)))
        y += test
    return out


def _slice(r: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> pd.Series:
    return r[(r.index >= a) & (r.index <= b)]


def trade_returns(result: BacktestResult, initial_capital: float) -> np.ndarray:
    """Each trade's net P&L as a fraction of portfolio equity at the close before its entry."""
    if result.trades.empty:
        return np.array([])
    eq = result.equity
    pos = eq.index.searchsorted(pd.DatetimeIndex(result.trades["entry_date"]), side="left") - 1
    base = np.where(pos >= 0, eq.to_numpy()[np.clip(pos, 0, None)], initial_capital)
    return result.trades["pnl_net"].to_numpy(dtype=float) / base


def mc_drawdown_pct95(trade_rets: np.ndarray, runs: int, seed: int) -> float:
    """95th-percentile max drawdown over `runs` reshuffles of the trade order, compounding trade returns
    (shuffling rupee P&L would mis-scale a compounding account)."""
    if len(trade_rets) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    dds = np.empty(runs)
    for i in range(runs):
        path = np.concatenate([[1.0], np.cumprod(1 + rng.permutation(trade_rets))])
        peak = np.maximum.accumulate(path)
        dds[i] = np.max(1 - path / peak)
    return float(np.percentile(dds, 95))


def thresholds(strategy: LoadedStrategy, cfg: AppConfig) -> tuple[float, float]:
    """(sensitivity min profitable fraction, DSR min); stricter when the strategy has > 5 params."""
    g = cfg.gates.gate2
    if len(strategy.spec.params) > g.overfit_param_count:
        return g.overfit_sens_min_profitable, g.overfit_dsr_min
    return g.sens_min_profitable, g.dsr_min


def gate2(strategy: LoadedStrategy, bars: pd.DataFrame, ctx: BacktestContext, cfg: AppConfig,
          family_trials: int, family_sr_var: float | None) -> GateResult:
    g, gates = cfg.gates.gate2, cfg.gates
    start, end = gates.research_start, gates.research_end
    sens_min, dsr_min = thresholds(strategy, cfg)
    defaults = strategy.spec.defaults()
    grid = param_grid(strategy.spec, g.grid_values_per_param, g.grid_max_combos, g.mc_seed)
    neigh = neighbours(strategy.spec, g.sens_step_pct)
    checks: list[CheckResult] = []

    # 1. look-ahead at each param's range ends (acceptance only checked the defaults)
    leaks = lookahead_params(strategy, bars, range_ends(strategy.spec), g.lookahead_param_window)
    checks.append(CheckResult("lookahead_params", len(leaks), 0, not leaks))
    if leaks:
        return GateResult.of("gate2", checks, leaks)

    # 2. every param set the gate uses must produce well-formed output
    run = _Runner(strategy, bars, ctx, cfg, start, end, chunk_size=g.sandbox_chunk_size)
    run.prefetch([defaults, *grid, *neigh])
    bad = list(run.invalid.values())
    checks.append(CheckResult("signals_valid", len(bad), 0, not bad))
    if bad:
        return GateResult.of("gate2", checks, bad)
    base = run.result(defaults)
    notes: list[str] = []

    # walk-forward: fit = best in-sample Sharpe on the grid; returns sliced from one full-period run per point
    grid_rets = {_key(p): returns(run.result(p).equity) for p in grid}
    is_best, oos_parts, losing = [], [], 0
    for fa, fb, ta, tb in _windows(start, end, g.wf_fit_years, g.wf_test_years):
        scored = [(sharpe(_slice(r, fa, fb), gates.risk_free_rate), k) for k, r in grid_rets.items()]
        best_sr, best_k = max(scored, key=lambda t: (t[0], t[1]))
        is_best.append(best_sr)
        oos = _slice(grid_rets[best_k], ta, tb)
        oos_parts.append(oos)
        if float(np.prod(1 + oos.to_numpy()) - 1) < 0:
            losing += 1
    oos_all = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    mean_is = float(np.mean(is_best)) if is_best else 0.0
    ratio = sharpe(oos_all, gates.risk_free_rate) / mean_is if mean_is > 0 else 0.0

    # sensitivity (not applicable when there is nothing to perturb)
    if neigh:
        frac = float(np.mean([total_return(run.result(p).equity) > 0 for p in neigh]))
        sens = CheckResult("sensitivity", frac, sens_min, frac >= sens_min)
    else:
        sens = CheckResult("sensitivity", math.nan, sens_min, True)
        notes.append("sensitivity: not applicable (no params, or ranges too narrow to perturb)")

    dd95 = mc_drawdown_pct95(trade_returns(base, cfg.risk.initial_capital), g.mc_runs, g.mc_seed)
    stressed = total_return(run.result(defaults, g.cost_stress_slippage_mult).equity)

    # deflated Sharpe
    r = returns(base.equity)
    sr = per_period_sharpe(r)
    if family_sr_var is None or family_trials < 2:
        grid_srs = [per_period_sharpe(x) for x in grid_rets.values()]
        sr_var = float(np.var(grid_srs, ddof=1)) if len(grid_srs) > 1 else 0.0
    else:
        sr_var = family_sr_var
    sk = float(skew(r)) if len(r) > 2 else 0.0
    ku = float(kurtosis(r, fisher=False)) if len(r) > 3 else 3.0
    dsr = deflated_sharpe(sr, len(r), sk, ku, max(family_trials, 1), sr_var)

    checks += [
        CheckResult("wf_oos_ratio", ratio, g.wf_min_oos_ratio, ratio >= g.wf_min_oos_ratio),
        CheckResult("wf_losing_years", losing, g.wf_max_losing_years, losing <= g.wf_max_losing_years),
        sens,
        CheckResult("mc_dd95", dd95, g.mc_dd_pct95_max, dd95 <= g.mc_dd_pct95_max),
        CheckResult("cost_stress", stressed, 0.0, stressed > 0),
        CheckResult("dsr", dsr, dsr_min, dsr >= dsr_min),
    ]
    return GateResult.of("gate2", checks, notes)
