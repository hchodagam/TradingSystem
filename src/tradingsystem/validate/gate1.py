"""Gate 1: research-period backtest, net of costs."""
from __future__ import annotations

import pandas as pd

from tradingsystem.backtest.engine import BacktestResult
from tradingsystem.config import GatesConfig
from tradingsystem.validate import CheckResult, GateResult
from tradingsystem.validate.metrics import max_drawdown, profit_factor, returns, sharpe


def gate1(result: BacktestResult, benchmark_close: pd.Series, cfg: GatesConfig) -> GateResult:
    g = cfg.gate1
    rf = cfg.risk_free_rate
    s = sharpe(returns(result.equity), rf)
    dd = max_drawdown(result.equity)
    n = len(result.trades)
    pf = profit_factor(result.trades["pnl_net"]) if n else 0.0
    bench = benchmark_close.reindex(result.equity.index).ffill().dropna()
    bs = sharpe(returns(bench), rf)
    return GateResult.of("gate1", [
        CheckResult("sharpe", s, g.min_sharpe, s >= g.min_sharpe),
        CheckResult("max_drawdown", dd, g.max_drawdown, dd <= g.max_drawdown),
        CheckResult("trades", n, g.min_trades, n >= g.min_trades),
        CheckResult("profit_factor", pf, g.min_profit_factor, pf >= g.min_profit_factor),
        CheckResult("beats_benchmark", s - bs, 0.0, s > bs),
    ])
