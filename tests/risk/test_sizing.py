from pathlib import Path

import pytest

from tradingsystem.config import load_config
from tradingsystem.risk.sizing import is_upper_circuit, size_position

cfg = load_config(Path(__file__).parents[2] / "config").risk


def test_risk_sizing_capped_by_position_pct():
    # 1% of 1L = 1000 risk / 20 per share = 50 shares = 25,000 > 20% cap (20,000) -> 40
    assert size_position(500, 480, 10, 100_000, cfg) == 40


def test_risk_sizing_exact():
    # 1000 / 10 = 100 shares = 20,000 = exactly the cap
    assert size_position(200, 190, 5, 100_000, cfg) == 100


def test_nan_stop_uses_2_atr():
    assert size_position(200, float("nan"), 5, 100_000, cfg) == 100


def test_skip_below_min_value():
    # 1000 / 1000 = 1 share worth 4,000 < 5,000
    assert size_position(4000, 3000, 50, 100_000, cfg) == 0


def test_skip_below_one_share():
    assert size_position(9000, 7000, 50, 100_000, cfg) == 0


def test_stop_at_or_above_entry_falls_back_to_atr():
    assert size_position(200, 205, 5, 100_000, cfg) == 100


def test_no_stop_and_no_atr_skips():
    assert size_position(200, float("nan"), float("nan"), 100_000, cfg) == 0


def test_zero_equity_skips():
    assert size_position(200, 190, 5, 0, cfg) == 0


@pytest.mark.parametrize("open_,prev,expected", [(110.0, 100.0, True), (109.99, 100.0, False)])
def test_upper_circuit_proxy(open_, prev, expected):
    assert is_upper_circuit(open_, prev, cfg) is expected
