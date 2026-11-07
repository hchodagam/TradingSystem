"""Dhan v2 historical daily data client."""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import httpx
import pandas as pd

from tradingsystem.config import DataConfig
from tradingsystem.data import BAR_COLUMNS


class DhanAuthError(RuntimeError):
    pass


_RETRYABLE = {429, 500, 502, 503, 504}


class DhanClient:
    def __init__(self, cfg: DataConfig, client_id: str, token: str, http: httpx.Client | None = None,
                 scrip_master: pd.DataFrame | None = None, sleep: Callable[[float], None] = time.sleep,
                 security_overrides: dict[str, str] | None = None):
        self.cfg = cfg
        # symbol -> security id for historical members absent from today's scrip master (delisted/renamed)
        self.overrides = dict(security_overrides or {})
        self.client_id = client_id
        self.token = token
        self.http = http or httpx.Client(base_url=cfg.dhan_base_url, timeout=30.0)
        self._scrip = scrip_master
        self._sleep = sleep

    # --- security ids -------------------------------------------------------------------------
    def _scrip_master(self) -> pd.DataFrame:
        if self._scrip is None:
            path = Path(self.cfg.cache_dir) / "scrip_master.csv"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                r = httpx.get(self.cfg.scrip_master_url, timeout=120.0)
                r.raise_for_status()
                path.write_bytes(r.content)
            self._scrip = pd.read_csv(path, dtype=str, keep_default_na=False)
        return self._scrip

    def security_id(self, symbol: str, index: bool = False) -> str:
        if not index and symbol in self.overrides:
            return self.overrides[symbol]
        m = self._scrip_master()
        rows = m[(m["SEM_EXM_EXCH_ID"] == "NSE") & (m["SEM_TRADING_SYMBOL"] == symbol)
                 & (m["SEM_INSTRUMENT_NAME"] == ("INDEX" if index else "EQUITY"))]
        if not index and "SEM_SERIES" in rows and (rows["SEM_SERIES"] == "EQ").any():
            rows = rows[rows["SEM_SERIES"] == "EQ"]
        if rows.empty:
            raise KeyError(f"{symbol!r} not in Dhan scrip master ({'INDEX' if index else 'EQUITY'})")
        return str(rows["SEM_SMST_SECURITY_ID"].iloc[0])

    # --- daily bars ---------------------------------------------------------------------------
    def _post(self, body: dict) -> dict:
        headers = {"access-token": self.token, "client-id": self.client_id, "Content-Type": "application/json"}
        attempt = 0
        while True:
            try:
                r = self.http.post("/charts/historical", json=body, headers=headers)
                if r.status_code == 401:
                    raise DhanAuthError("access token expired or invalid")
                if r.status_code == 400 and _error_code(r) == "DH-907":
                    return {}  # "no data present" for this range (e.g. before listing): an empty chunk
                if r.status_code in _RETRYABLE and attempt < self.cfg.retries:
                    raise _Retry()
                r.raise_for_status()
                return r.json()
            except (_Retry, httpx.TransportError):
                if attempt >= self.cfg.retries:
                    raise
                self._sleep(self.cfg.backoff_seconds * (2 ** attempt))
                attempt += 1

    def daily(self, symbol: str, start: date, end: date, index: bool = False) -> pd.DataFrame:
        """Unadjusted daily bars for [start, end], indexed by tz-naive date."""
        sid = self.security_id(symbol, index=index)
        frames = []
        lo = start
        stop = end + timedelta(days=1)  # Dhan's toDate is exclusive
        while lo < stop:
            hi = min(lo + timedelta(days=self.cfg.max_days_per_request), stop)
            data = self._post({
                "securityId": sid, "exchangeSegment": "IDX_I" if index else "NSE_EQ",
                "instrument": "INDEX" if index else "EQUITY", "expiryCode": 0, "oi": False,
                "fromDate": lo.isoformat(), "toDate": hi.isoformat(),
            })
            frames.append(_to_frame(data))
            lo = hi
        df = pd.concat(frames) if frames else _to_frame({})
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


class _Retry(Exception):
    pass


def _error_code(r: httpx.Response) -> str | None:
    try:
        return r.json().get("errorCode")
    except ValueError:
        return None


def load_security_overrides(path: Path) -> dict[str, str]:
    """reference/security_ids.csv: symbol,security_id (optional file)."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {r.symbol.strip(): r.security_id.strip() for r in df.itertuples(index=False) if r.symbol.strip()}


def _to_frame(data: dict) -> pd.DataFrame:
    ts = data.get("timestamp") or []
    dates = (pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True)
             .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize())
    df = pd.DataFrame({c: pd.Series(data.get(c) or [], dtype="float64") for c in BAR_COLUMNS})
    df.index = pd.DatetimeIndex(dates, name="date")
    return df
