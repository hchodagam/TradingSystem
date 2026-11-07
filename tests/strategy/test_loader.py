import pytest

from tradingsystem.strategy.loader import StrategyFormatError, load_strategy

GOOD = '''import pandas as pd

SPEC = {
    "name": "demo",
    "family": "trend_following",
    "hypothesis": "Trends persist",
    "universe": "nifty100",
    "timeframe": "1D",
    "params": {"n": {"default": 10, "range": [5, 20]}},
    "max_holding_days": 5,
}


def signals(bars, p):
    return bars
'''


def write(tmp_path, text, name="s.py", newline="\n"):
    p = tmp_path / name
    p.write_bytes(text.replace("\n", newline).encode())
    return p


def test_loads_good_file(tmp_path):
    s = load_strategy(write(tmp_path, GOOD))
    assert s.spec.name == "demo" and len(s.hash) == 64 and "def signals" in s.code


def test_missing_spec(tmp_path):
    with pytest.raises(StrategyFormatError, match="SPEC"):
        load_strategy(write(tmp_path, GOOD.replace("SPEC =", "OTHER =")))


def test_missing_signals(tmp_path):
    with pytest.raises(StrategyFormatError, match="signals"):
        load_strategy(write(tmp_path, GOOD.replace("def signals(bars, p)", "def other(bars, p)")))


def test_wrong_signals_signature(tmp_path):
    with pytest.raises(StrategyFormatError, match="signals"):
        load_strategy(write(tmp_path, GOOD.replace("def signals(bars, p)", "def signals(bars)")))


def test_non_literal_spec(tmp_path):
    with pytest.raises(StrategyFormatError, match="literal"):
        load_strategy(write(tmp_path, GOOD.replace('"name": "demo"', '"name": make_name()')))


def test_invalid_spec_schema(tmp_path):
    with pytest.raises(StrategyFormatError, match="max_holding_days"):
        load_strategy(write(tmp_path, GOOD.replace('"max_holding_days": 5', '"max_holding_days": 0')))


def test_syntax_error(tmp_path):
    with pytest.raises(StrategyFormatError, match="syntax"):
        load_strategy(write(tmp_path, GOOD + "\ndef broken(:\n"))


def test_crlf_and_lf_same_hash(tmp_path):
    a = load_strategy(write(tmp_path, GOOD, "a.py", "\n"))
    b = load_strategy(write(tmp_path, GOOD, "b.py", "\r\n"))
    assert a.hash == b.hash


def test_hash_changes_with_code(tmp_path):
    a = load_strategy(write(tmp_path, GOOD, "a.py"))
    b = load_strategy(write(tmp_path, GOOD + "# x\n", "b.py"))
    assert a.hash != b.hash


def test_file_is_never_executed(tmp_path):
    s = load_strategy(write(tmp_path, "raise SystemExit('executed!')\n" + GOOD))
    assert s.spec.name == "demo"
