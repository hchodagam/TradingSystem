"""Parquet cache of raw Dhan bars, with research/holdout clamping and adjustment on read."""
from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tradingsystem.calendar import trading_days
from tradingsystem.config import AppConfig
from tradingsystem.data import BAR_COLUMNS
from tradingsystem.data.adjust import adjust, load_actions


class UnknownSymbolsError(KeyError):
    def __init__(self, symbols: list[str]):
        super().__init__(", ".join(symbols))
        self.symbols = symbols

    def __str__(self) -> str:
        return (f"Dhan does not know {len(self.symbols)} symbol(s): {', '.join(self.symbols)}. Delisted or renamed "
                "members need an entry in reference/security_ids.csv (symbol,security_id).")


def _fname(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9&_-]", "_", symbol) + ".parquet"


class BarStore:
    def __init__(self, cfg: AppConfig, client=None):
        self.cfg = cfg
        self.client = client
        self.cache = Path(cfg.data.cache_dir)

    # --- cache ------------------------------------------------------------------------------
    def _path(self, symbol: str, index: bool = False) -> Path:
        return self.cache / ("index" if index else "equity") / _fname(symbol)

    def _read_raw(self, symbol: str, index: bool = False) -> pd.DataFrame:
        p = self._path(symbol, index)
        if not p.exists():
            raise FileNotFoundError(f"no cached bars for {symbol} at {p}; run refresh first")
        return pd.read_parquet(p)

    def _refresh_one(self, symbol: str, end: date, index: bool) -> None:
        p = self._path(symbol, index)
        start = self.cfg.gates.research_start
        old = None
        if p.exists():
            old = pd.read_parquet(p)
            if len(old):
                start = old.index.max().date() + timedelta(days=1)
        if start > end or len(trading_days(start, end)) == 0:
            return
        new = self.client.daily(symbol, start, end, index=index)[BAR_COLUMNS].astype("float64")
        df = new if old is None else pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index.name = "date"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)

    def refresh(self, symbols, end: date, include_benchmark: bool = True) -> None:
        if self.client is None:
            raise RuntimeError("BarStore has no Dhan client; cannot refresh")
        unknown = []
        for s in sorted(set(symbols)):
            try:
                self._refresh_one(s, end, index=False)
            except KeyError:
                unknown.append(s)
        if unknown:
            raise UnknownSymbolsError(unknown)
        if include_benchmark:
            self._refresh_one(self.cfg.data.benchmark_symbol, end, index=True)

    # --- reads ------------------------------------------------------------------------------
    def _actions(self) -> pd.DataFrame:
        p = Path(self.cfg.data.reference_dir) / "corporate_actions.csv"
        return load_actions(p) if p.exists() else pd.DataFrame(columns=["symbol", "ex_date", "factor"])

    def _bar_fixes(self) -> set[tuple[pd.Timestamp, str]]:
        """reference/bar_fixes.csv: symbol,date,action,note - owner-reviewed bad bars. Only action 'drop'."""
        p = Path(self.cfg.data.reference_dir) / "bar_fixes.csv"
        if not p.exists():
            return set()
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        bad = df[df["action"].str.strip() != "drop"]
        if len(bad):
            raise ValueError(f"bar_fixes.csv: unsupported action(s) {sorted(set(bad['action']))}; only 'drop'")
        return {(pd.Timestamp(r.date), r.symbol.strip()) for r in df.itertuples(index=False)}

    def _bars(self, symbols, start: date, end: date | None) -> pd.DataFrame:
        frames = []
        for s in sorted(set(symbols)):
            df = self._read_raw(s).copy()
            df["symbol"] = s
            frames.append(df.reset_index())
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS, index=pd.MultiIndex.from_tuples([], names=["date", "symbol"]))
        out = pd.concat(frames, ignore_index=True)
        out = out[out["date"] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out["date"] <= pd.Timestamp(end)]
        out = out.set_index(["date", "symbol"]).sort_index()[BAR_COLUMNS]
        drops = self._bar_fixes()
        if drops:
            out = out[~out.index.isin(list(drops))]
        if not self.cfg.data.prices_adjusted_by_dhan:
            out = adjust(out, self._actions())
        return out

    def research_bars(self, symbols) -> pd.DataFrame:
        g = self.cfg.gates
        return self._bars(symbols, g.research_start, g.research_end)

    def holdout_bars(self, symbols) -> pd.DataFrame:
        """Research history (for warm-up) plus the holdout. Only Gate 3 may call this."""
        return self._bars(symbols, self.cfg.gates.research_start, None)

    def benchmark(self, start: date, end: date) -> pd.Series:
        df = self._read_raw(self.cfg.data.benchmark_symbol, index=True)
        s = df["close"]
        return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
