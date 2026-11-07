"""Event-driven daily backtest: signals at close t -> orders filled at open t+1.

Per trading day t:
 1. open: fill queued exits, then queued entries (skip on upper circuit / open at or below stop;
    shrink qty to the 20% cap at the fill price and to available cash; re-apply skip rules)
 2. stops: open <= stop -> exit at open; else low <= stop -> exit at stop (slippage applied)
 3. close: mark to market, kill-switch, then queue exits (signal, max holding, data missing) and
    new entries (members only, sized on close_t as the entry estimate, admitted by risk caps)
At the end every holding is closed at the last close (reason end_of_test).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from tradingsystem.backtest.indicators import atr
from tradingsystem.config import AppConfig
from tradingsystem.costs import Fill, apply_slippage, day_charges
from tradingsystem.risk.admission import Holding, OrderIntent, admit
from tradingsystem.risk.killswitch import KillSwitch
from tradingsystem.risk.sizing import is_upper_circuit, size_position
from tradingsystem.strategy.spec import StrategySpec
from tradingsystem.universe import Universe

TRADE_COLUMNS = ["symbol", "entry_date", "entry_price", "exit_date", "exit_price", "qty", "charges", "pnl_net",
                 "exit_reason"]


@dataclass(frozen=True)
class BacktestConfig:
    start: date
    end: date
    initial_capital: float
    slippage_mult: float = 1.0
    enforce_kill_switch: bool = False  # nobody can resume() in a backtest; trips are reported instead


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    kill_switch_trips: int
    skipped: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "symbol", "reason"]))


@dataclass
class _Pos:
    h: Holding
    entry_charges: float
    last_close: float
    days_held: int = 0
    missing: int = 0


def _by_date(df: pd.DataFrame, cols: list[str]) -> dict[pd.Timestamp, dict[str, tuple]]:
    # plain-list iteration: iterating pandas (Arrow-backed) index values directly is ~50x slower
    codes, uniq = pd.factorize(df.index.get_level_values("date"))
    keys = list(uniq)
    out: dict[pd.Timestamp, dict[str, tuple]] = {k: {} for k in keys}
    syms = df.index.get_level_values("symbol").to_numpy(dtype=object).tolist()
    vals = df[cols].to_numpy(dtype=float).tolist()
    for c, s, v in zip(codes.tolist(), syms, vals, strict=True):
        out[keys[c]][s] = tuple(v)
    return out


def run_backtest(signals: pd.DataFrame, bars: pd.DataFrame, universe: Universe, sectors: dict[str, str],
                 spec: StrategySpec, cfg: AppConfig, bt: BacktestConfig) -> BacktestResult:
    risk, rates = cfg.risk, cfg.costs
    lo, hi = pd.Timestamp(bt.start), pd.Timestamp(bt.end)
    all_dates = bars.index.get_level_values("date").unique().sort_values()
    days = [d for d in all_dates if lo <= d <= hi]

    ohlc = _by_date(bars, ["open", "high", "low", "close"])
    atr_s = atr(bars, risk.atr_len)
    atr_by = _by_date(atr_s.to_frame("atr"), ["atr"])
    active = signals[signals["entry"] | signals["exit"] | signals["stop"].notna()]
    sig_by = _by_date(active, ["entry", "exit", "stop"])

    cash = float(bt.initial_capital)
    pos: dict[str, _Pos] = {}
    prev_close: dict[str, float] = {}
    pending_entries: list[OrderIntent] = []
    pending_exits: dict[str, str] = {}
    trades: list[dict] = []
    skipped: list[tuple] = []
    equity_pts: list[float] = []
    ks = KillSwitch(risk, enforce=bt.enforce_kill_switch)
    last_equity = cash

    def charges_for(sym: str, d: pd.Timestamp, side: str, qty: int, price: float) -> float:
        return day_charges([Fill(sym, d.date(), side, qty, price)], rates).total

    def close_position(sym: str, d: pd.Timestamp, raw_price: float, reason: str) -> None:
        nonlocal cash
        p = pos.pop(sym)
        price = apply_slippage(raw_price, "SELL", rates, bt.slippage_mult)
        ch = charges_for(sym, d, "SELL", p.h.qty, price)
        cash += p.h.qty * price - ch
        gross = (price - p.h.entry_price) * p.h.qty
        trades.append({"symbol": sym, "entry_date": p.h.entry_date, "entry_price": p.h.entry_price,
                       "exit_date": d, "exit_price": price, "qty": p.h.qty, "charges": round(p.entry_charges + ch, 2),
                       "pnl_net": gross - p.entry_charges - ch, "exit_reason": reason})

    for d in days:
        today = ohlc.get(d, {})

        # 1. open: exits first, then entries
        for sym, reason in sorted(pending_exits.items()):
            if sym in pos and sym in today:
                close_position(sym, d, today[sym][0], reason)
        pending_exits = {s: r for s, r in pending_exits.items() if s in pos and s not in today}

        for it in pending_entries:
            bar = today.get(it.symbol)
            if bar is None:
                skipped.append((d, it.symbol, "no_bar"))
                continue
            o = bar[0]
            if is_upper_circuit(o, prev_close.get(it.symbol, o), risk):
                skipped.append((d, it.symbol, "upper_circuit"))
                continue
            if math.isfinite(it.stop) and o <= it.stop:
                skipped.append((d, it.symbol, "opened_below_stop"))
                continue
            price = apply_slippage(o, "BUY", rates, bt.slippage_mult)
            qty = min(it.qty, math.floor(risk.max_position_pct * last_equity / price + 1e-9))
            qty = min(qty, math.floor(cash / price))
            ch = charges_for(it.symbol, d, "BUY", qty, price) if qty > 0 else 0.0
            while qty > 0 and qty * price + ch > cash:
                qty -= 1
                ch = charges_for(it.symbol, d, "BUY", qty, price) if qty > 0 else 0.0
            if qty < 1 or qty * price < risk.min_position_value:
                skipped.append((d, it.symbol, "insufficient_cash"))
                continue
            cash -= qty * price + ch
            pos[it.symbol] = _Pos(Holding(it.symbol, qty, price, it.stop, d), ch, last_close=o)
        pending_entries = []

        # 2. stops
        for sym in sorted(pos):
            p, bar = pos[sym], today.get(sym)
            if bar is None or not math.isfinite(p.h.stop):
                continue
            o, _h, low, _c = bar
            if o <= p.h.stop and p.h.entry_date != d:
                close_position(sym, d, o, "stop")
            elif low <= p.h.stop:
                close_position(sym, d, p.h.stop, "stop")

        # 3. close: mark, kill-switch, queue orders
        for sym, bar in today.items():
            prev_close[sym] = bar[3]
        for sym in sorted(pos):
            p = pos[sym]
            p.days_held += 1
            if sym in today:
                p.last_close = today[sym][3]
                p.missing = 0
            else:
                p.missing += 1
        for sym in sorted(pos):
            if pos[sym].missing >= risk.data_missing_exit_days:
                close_position(sym, d, pos[sym].last_close, "data_missing")
                pending_exits.pop(sym, None)

        equity = cash + sum(p.h.qty * p.last_close for p in pos.values())
        ks.update(equity)
        last_equity = equity

        sigs = sig_by.get(d, {})
        for sym in sorted(pos):
            s = sigs.get(sym)
            if s is not None and math.isfinite(s[2]):
                pos[sym].h.stop = float(s[2])  # a held symbol's latest stop wins (trailing stops)
            if sym in pending_exits:
                continue
            if s is not None and s[1]:
                pending_exits[sym] = "signal"
            elif pos[sym].days_held >= spec.max_holding_days:
                pending_exits[sym] = "max_holding"

        if ks.allows_entries():
            members = universe.members(d) if d.date() >= universe.first_date else frozenset()
            intents = []
            for sym in sorted(sigs):
                entry, _x, stop = sigs[sym]
                if not entry or sym in pos or sym not in members or sym not in today:
                    continue
                a = atr_by.get(d, {}).get(sym, (math.nan,))[0]
                qty = size_position(today[sym][3], float(stop), float(a), equity, risk)
                if qty == 0:
                    skipped.append((d, sym, "size"))
                    continue
                intents.append(OrderIntent(sym, qty, float(stop), d))
            accepted, rejected = admit(intents, {s: p.h for s, p in pos.items()}, sectors, risk)
            skipped.extend((d, it.symbol, why) for it, why in rejected)
            pending_entries = accepted
        equity_pts.append(equity)

    if days:
        last = days[-1]
        for sym in sorted(pos):
            close_position(sym, last, pos[sym].last_close, "end_of_test")
        equity_pts[-1] = cash

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    skipped_df = pd.DataFrame(skipped, columns=["date", "symbol", "reason"])
    equity_s = pd.Series(equity_pts, index=pd.DatetimeIndex(days, name="date"), name="equity", dtype=float)
    return BacktestResult(equity=equity_s, trades=trades_df, kill_switch_trips=ks.trips, skipped=skipped_df)
