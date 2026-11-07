import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest

from tradingsystem.config import load_config
from tradingsystem.data.dhan import DhanAuthError, DhanClient

CFG = load_config(Path(__file__).parents[2] / "config").data
IST = timezone(timedelta(hours=5, minutes=30))
SCRIP = pd.DataFrame({
    "SEM_EXM_EXCH_ID": ["NSE", "NSE", "NSE"],
    "SEM_INSTRUMENT_NAME": ["EQUITY", "EQUITY", "INDEX"],
    "SEM_TRADING_SYMBOL": ["INFY", "INFY", "NIFTY 100"],
    "SEM_SERIES": ["EQ", "BE", ""],
    "SEM_SMST_SECURITY_ID": ["1594", "9999", "17"],
})


def _epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp())


def _payload(days: list[date]) -> dict:
    n = len(days)
    return {"open": [100.0 + i for i in range(n)], "high": [110.0 + i for i in range(n)],
            "low": [90.0 + i for i in range(n)], "close": [105.0 + i for i in range(n)],
            "volume": [1000 + i for i in range(n)], "timestamp": [_epoch(d) for d in days]}


def _client(handler, **kw) -> DhanClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=CFG.dhan_base_url)
    return DhanClient(CFG, "cid", "tok", http=http, scrip_master=SCRIP, sleep=lambda s: None, **kw)


def test_security_id_prefers_eq_series():
    c = _client(lambda r: httpx.Response(500))
    assert c.security_id("INFY") == "1594"
    assert c.security_id("NIFTY 100", index=True) == "17"


def test_daily_parses_arrays():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        assert req.headers["access-token"] == "tok"
        return httpx.Response(200, json=_payload([date(2024, 1, 2), date(2024, 1, 3)]))

    df = _client(handler).daily("INFY", date(2024, 1, 1), date(2024, 1, 5))
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df.loc["2024-01-03", "close"] == 106.0
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    body = seen[0]
    assert body["securityId"] == "1594" and body["exchangeSegment"] == "NSE_EQ" and body["instrument"] == "EQUITY"


def test_index_uses_idx_segment():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_payload([]))

    _client(handler).daily("NIFTY 100", date(2024, 1, 1), date(2024, 1, 5), index=True)
    assert seen[0]["exchangeSegment"] == "IDX_I" and seen[0]["instrument"] == "INDEX"


def test_chunks_long_range():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_payload([]))

    _client(handler).daily("INFY", date(2012, 1, 1), date(2022, 12, 31))
    assert len(seen) == 3
    assert seen[0]["fromDate"] == "2012-01-01"
    for a, b in zip(seen, seen[1:], strict=False):
        assert a["toDate"] == b["fromDate"]  # toDate is exclusive: chunks are contiguous


def test_rows_outside_range_dropped():
    def handler(req):
        return httpx.Response(200, json=_payload([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 8)]))

    df = _client(handler).daily("INFY", date(2024, 1, 2), date(2024, 1, 5))
    assert list(df.index) == [pd.Timestamp("2024-01-02")]


def test_retries_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500)
        return httpx.Response(200, json=_payload([date(2024, 1, 2)]))

    df = _client(handler).daily("INFY", date(2024, 1, 1), date(2024, 1, 5))
    assert len(calls) == 3 and len(df) == 1


def test_gives_up_after_retries():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).daily("INFY", date(2024, 1, 1), date(2024, 1, 5))
    assert len(calls) == CFG.retries + 1


def test_no_data_chunk_before_listing_is_empty_not_an_error():
    # Dhan answers a range with no data (e.g. before listing) with 400 DH-907
    def handler(req):
        body = json.loads(req.content)
        if body["fromDate"] < "2020-01-01":
            return httpx.Response(400, json={"errorType": "Data_Error", "errorCode": "DH-907",
                                             "errorMessage": "no data present"})
        return httpx.Response(200, json=_payload([date(2022, 1, 3)]))

    df = _client(handler).daily("INFY", date(2012, 1, 1), date(2022, 12, 31))
    assert list(df.index) == [pd.Timestamp("2022-01-03")]


def test_other_400_still_raises():
    def handler(req):
        return httpx.Response(400, json={"errorType": "Input_Exception", "errorCode": "DH-905", "errorMessage": "x"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).daily("INFY", date(2024, 1, 1), date(2024, 1, 5))


def test_401_raises_auth_error_without_retry():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401)

    with pytest.raises(DhanAuthError):
        _client(handler).daily("INFY", date(2024, 1, 1), date(2024, 1, 5))
    assert len(calls) == 1


def test_security_id_override_for_delisted_symbol():
    # I5: historical members missing from today's scrip master are mapped in a reference file
    c = _client(lambda r: httpx.Response(500), security_overrides={"OLDCO": "4242"})
    assert c.security_id("OLDCO") == "4242"


def test_unknown_symbol_raises():
    with pytest.raises(KeyError):
        _client(lambda r: httpx.Response(500)).security_id("NOPE")
