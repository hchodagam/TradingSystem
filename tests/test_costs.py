import json
from datetime import date
from pathlib import Path

import pytest

from tradingsystem.config import load_config
from tradingsystem.costs import Fill, apply_slippage, day_charges

ROOT = Path(__file__).parent.parent
rates = load_config(ROOT / "config").costs
D = date(2024, 3, 1)
NOTES = sorted((Path(__file__).parent / "fixtures" / "contract_notes").glob("*.json"))


def test_buy_delivery():
    # turnover 1,00,000: STT 100, exch 2.97, SEBI 0.10, stamp 15, GST 0.18*(2.97+0.10)=0.5526
    c = day_charges([Fill("INFY", D, "BUY", 100, 1000.0)], rates)
    assert (c.stt, c.exchange, c.sebi, c.stamp, c.dp, c.gst) == (100, 2.97, 0.10, 15, 0, 0.55)
    assert c.total == pytest.approx(118.62)


def test_sell_delivery_with_dp():
    # turnover 1,10,000: STT 110, exch 3.267, SEBI 0.11, DP 12.5, GST 0.18*(3.267+0.11+12.5)=2.85786
    c = day_charges([Fill("INFY", D, "SELL", 100, 1100.0)], rates)
    assert (c.stt, c.exchange, c.sebi, c.stamp, c.dp, c.gst) == (110, 3.27, 0.11, 0, 12.5, 2.86)
    assert c.total == pytest.approx(128.74)


def test_dp_once_per_scrip_per_day():
    c = day_charges([Fill("INFY", D, "SELL", 50, 1100.0), Fill("INFY", D, "SELL", 50, 1100.0)], rates)
    assert c.dp == 12.5
    assert c.stt == 110


def test_dp_per_scrip():
    c = day_charges([Fill("INFY", D, "SELL", 10, 1000.0), Fill("TCS", D, "SELL", 10, 1000.0)], rates)
    assert c.dp == 25.0


def test_empty_fills_cost_nothing():
    assert day_charges([], rates).total == 0


def test_slippage():
    assert apply_slippage(100.0, "BUY", rates) == pytest.approx(100.1)
    assert apply_slippage(100.0, "SELL", rates) == pytest.approx(99.9)
    assert apply_slippage(100.0, "BUY", rates, mult=2.0) == pytest.approx(100.2)


@pytest.mark.parametrize("note", NOTES, ids=[n.stem for n in NOTES])
def test_matches_real_contract_note(note):
    data = json.loads(note.read_text())
    fills = [Fill(f["symbol"], date.fromisoformat(f["date"]), f["side"], f["qty"], f["price"]) for f in data["fills"]]
    c = day_charges(fills, rates)
    for k, v in data["expected"].items():
        assert getattr(c, k) == pytest.approx(v, abs=0.01), k
