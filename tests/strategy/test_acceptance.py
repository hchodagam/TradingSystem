from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import HEADER, small_bars

from tradingsystem.strategy.acceptance import accept, check_output

ROOT = Path(__file__).parents[2]
REF = ROOT / "strategies" / "reference" / "sma_crossover.py"
FIX = ROOT / "tests" / "fixtures" / "strategies"

BAD = {
    "nan_entry": ('out["entry"] = np.where(close > ma, True, np.nan)', "NaN in entry"),
    "index": ("out = out.iloc[1:]", "index"),
    "int_flags": ('out["entry"] = out["entry"].astype(int)', "bool"),
    "missing_stop": ('out = out.drop(columns=["stop"])', "stop"),
    "negative_stop": ('out["stop"] = -1.0', "stop"),
}

BODY = '''
def signals(bars, p):
    close = bars["close"]
    ma = close.groupby(level="symbol").transform(lambda s: s.rolling(p.n).mean())
    out = pd.DataFrame(index=bars.index)
    out["entry"] = (close > ma).astype(bool)
    out["exit"] = (close < ma).astype(bool)
    out["stop"] = np.nan
    {mutation}
    return out
'''


@pytest.fixture(scope="module")
def bars():
    return small_bars(n_symbols=3, years=2)


def write(tmp_path, body, name="s.py"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_reference_strategy_accepted(bars):
    r = accept(REF, bars)
    assert r.errors == [] and r.stage == "passed" and r.ok


def test_reference_strategy_generates_signals(bars):
    long_bars = small_bars(n_symbols=3, years=4, seed=3)
    r = accept(REF, long_bars)
    assert r.ok


def test_leaky_strategy_caught(bars):
    r = accept(FIX / "leaky.py", bars)
    assert (r.ok, r.stage) == (False, "lookahead")
    assert r.errors


@pytest.mark.parametrize("name", list(BAD))
def test_bad_output_rejected(tmp_path, bars, name):
    mutation, msg = BAD[name]
    r = accept(write(tmp_path, BODY.replace("{mutation}", mutation)), bars)
    assert r.stage == "smoke" and not r.ok
    assert any(msg in e for e in r.errors), r.errors


def test_lint_failure_stops_pipeline(tmp_path, bars):
    r = accept(write(tmp_path, BODY.replace("{mutation}", "x = undefined_name")), bars)
    assert r.stage == "lint" and any("F821" in e for e in r.errors)


def test_schema_failure(tmp_path, bars):
    r = accept(write(tmp_path, "x = 1\n").with_name("s.py"), bars)
    assert r.stage == "schema" and r.strategy is None


def test_static_violation_is_smoke_stage(tmp_path, bars):
    r = accept(write(tmp_path, BODY.replace("{mutation}", "bars.to_csv('x')")), bars)
    assert r.stage == "smoke" and any("to_csv" in e for e in r.errors)


def test_check_output_clean(bars):
    out = pd.DataFrame({"entry": False, "exit": False, "stop": np.nan}, index=bars.index)
    assert check_output(out, bars) == []
