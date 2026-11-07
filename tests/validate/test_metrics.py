import math

import pandas as pd
import pytest

from tradingsystem.validate.metrics import max_drawdown, profit_factor, sharpe


def test_sharpe_hand_calc():
    # mean 0.005, sample std sqrt(5e-4/3) = 0.0129099 -> 0.005/0.0129099*sqrt(252) = 6.1482
    assert sharpe(pd.Series([0.01, -0.01, 0.02, 0.0])) == pytest.approx(6.1482, abs=1e-4)


def test_sharpe_risk_free_rate():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    assert sharpe(r, rf=0.0252) == pytest.approx((0.005 - 0.0001) / 0.0129099 * math.sqrt(252), abs=1e-3)


def test_sharpe_zero_std_is_zero():
    assert sharpe(pd.Series([0.0, 0.0, 0.0])) == 0.0
    assert sharpe(pd.Series([0.01])) == 0.0


def test_max_drawdown():
    assert max_drawdown(pd.Series([100.0, 120.0, 90.0, 130.0])) == pytest.approx(0.25)
    assert max_drawdown(pd.Series([100.0, 110.0])) == 0.0


def test_profit_factor():
    assert profit_factor(pd.Series([100.0, -50.0, 50.0])) == pytest.approx(3.0)
    assert profit_factor(pd.Series([100.0])) == math.inf
    assert profit_factor(pd.Series([-1.0])) == 0.0
    assert profit_factor(pd.Series([], dtype=float)) == 0.0
