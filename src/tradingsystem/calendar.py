"""NSE trading days. Uses exchange_calendars' XBOM (BSE), which shares NSE's equity holidays."""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=1)
def _cal():
    return xcals.get_calendar("XBOM", start="2005-01-01")


def trading_days(start: date, end: date) -> pd.DatetimeIndex:
    """Trading sessions in [start, end], inclusive, as tz-naive midnight timestamps."""
    cal = _cal()
    lo = max(pd.Timestamp(start), cal.first_session)
    hi = min(pd.Timestamp(end), cal.last_session)
    if lo > hi:
        return pd.DatetimeIndex([])
    sessions = cal.sessions_in_range(lo, hi)
    return pd.DatetimeIndex(sessions.tz_localize(None) if sessions.tz else sessions, name="date")
