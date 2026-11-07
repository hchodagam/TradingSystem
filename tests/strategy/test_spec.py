import pytest
from pydantic import ValidationError

from tradingsystem.strategy.spec import Params, StrategySpec

EXAMPLE = {
    "name": "rsi_pullback_in_uptrend",
    "family": "mean_reversion_in_trend",
    "hypothesis": "Strong stocks that dip briefly tend to resume trend",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {
        "rsi_len": {"default": 3, "range": [2, 5]},
        "rsi_entry": {"default": 15, "range": [5, 25]},
        "trend_ma": {"default": 200, "range": [100, 250]},
    },
    "max_holding_days": 10,
}


def spec(**over):
    return StrategySpec.model_validate({**EXAMPLE, **over})


def test_design_doc_example_validates():
    s = spec()
    assert s.params["rsi_len"].is_int
    assert not s.overfit_risk


def test_default_outside_range_fails():
    with pytest.raises(ValidationError):
        spec(params={"x": {"default": 9, "range": [1, 5]}})


def test_lo_not_below_hi_fails():
    with pytest.raises(ValidationError):
        spec(params={"x": {"default": 5, "range": [5, 5]}})


def test_blank_hypothesis_fails():
    with pytest.raises(ValidationError):
        spec(hypothesis="   ")


@pytest.mark.parametrize("name", ["Bad", "1abc", "has-dash", ""])
def test_bad_name_fails(name):
    with pytest.raises(ValidationError):
        spec(name=name)


def test_wrong_universe_or_timeframe_fails():
    with pytest.raises(ValidationError):
        spec(universe="sp500")
    with pytest.raises(ValidationError):
        spec(timeframe="1H")


def test_six_params_is_overfit_risk():
    params = {f"p{i}": {"default": 1, "range": [0, 2]} for i in range(6)}
    assert spec(params=params).overfit_risk


def test_float_param():
    p = spec(params={"x": {"default": 0.5, "range": [0.1, 1]}}).params["x"]
    assert not p.is_int


def test_defaults_attribute_access():
    p = spec().defaults()
    assert isinstance(p, Params) and p.rsi_len == 3 and p["trend_ma"] == 200


def test_params_missing_attribute_raises():
    with pytest.raises(AttributeError):
        _ = Params(a=1).b


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        spec(author="me")
