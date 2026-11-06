import pytest
from scipy.stats import norm

from tradingsystem.strategy.spec import StrategySpec
from tradingsystem.validate.dsr import deflated_sharpe
from tradingsystem.validate.grid import param_grid


def test_dsr_single_trial_equals_psr():
    sr, n, skew, kurt = 0.1, 500, 0.0, 3.0
    psr = norm.cdf(sr * (n - 1) ** 0.5 / (1 - skew * sr + (kurt - 1) / 4 * sr**2) ** 0.5)
    assert deflated_sharpe(sr, n, skew, kurt, n_trials=1, sr_var=0.01) == pytest.approx(psr)


def test_dsr_drops_with_more_trials():
    args = dict(sr=0.1, n_obs=500, skew=0.0, kurt=3.0, sr_var=0.001)
    assert deflated_sharpe(n_trials=100, **args) < deflated_sharpe(n_trials=10, **args)


def test_dsr_zero_variance_is_psr():
    a = deflated_sharpe(0.1, 500, 0.0, 3.0, n_trials=50, sr_var=0.0)
    b = deflated_sharpe(0.1, 500, 0.0, 3.0, n_trials=1, sr_var=0.0)
    assert a == pytest.approx(b)


def spec(params):
    return StrategySpec.model_validate({
        "name": "g", "family": "g", "hypothesis": "g", "universe": "nifty100", "timeframe": "1D",
        "params": params, "max_holding_days": 5})


def test_grid_int_values_rounded_and_deduped():
    g = param_grid(spec({"n": {"default": 3, "range": [2, 4]}}), n_per_param=5, max_combos=100, seed=0)
    assert sorted(p.n for p in g) == [2, 3, 4]


def test_grid_includes_defaults_and_respects_cap():
    s = spec({f"p{i}": {"default": 0.5, "range": [0.0, 1.0]} for i in range(4)})
    g = param_grid(s, n_per_param=5, max_combos=50, seed=1)
    assert len(g) == 50
    assert dict(g[0]) == dict(s.defaults())
    assert len({tuple(sorted(p.items())) for p in g}) == 50


def test_grid_is_seeded():
    s = spec({f"p{i}": {"default": 0.5, "range": [0.0, 1.0]} for i in range(4)})
    assert param_grid(s, 5, 20, seed=3) == param_grid(s, 5, 20, seed=3)
