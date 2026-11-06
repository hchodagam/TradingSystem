import os

import pandas as pd
import pytest
from helpers import CLEAN_BODY, small_bars, strategy_from

from tradingsystem.strategy.sandbox import SandboxConfig, SandboxError, run_signals, run_signals_many
from tradingsystem.strategy.spec import Params


def test_many_params_one_process_matches_single_runs(tmp_path, bars):
    s = strategy_from(tmp_path, CLEAN_BODY)
    ps = [Params(n=2), Params(n=5), Params(n=9)]
    many = run_signals_many(s, ps, bars)
    assert len(many) == 3
    for p, out in zip(ps, many, strict=True):
        pd.testing.assert_frame_equal(out, run_signals(s, p, bars))
    assert not many[0].equals(many[2])


def test_many_keeps_each_runs_own_columns_and_dtypes(tmp_path, bars):
    # batching must not let one run's columns or dtypes leak into another's
    body = CLEAN_BODY.replace("    return out", '    if p.n > 5:\n        out = out.drop(columns=["stop"])\n'
                                             '        out["entry"] = out["entry"].astype(int)\n    return out')
    s = strategy_from(tmp_path, body)
    good, bad = run_signals_many(s, [Params(n=2), Params(n=9)], bars)
    assert list(good.columns) == ["entry", "exit", "stop"] and good["entry"].dtype == bool
    assert list(bad.columns) == ["entry", "exit"] and bad["entry"].dtype != bool


def test_child_does_not_import_pydantic():
    import subprocess
    import sys
    code = "import sys, tradingsystem.strategy.params; print('pydantic' in sys.modules)"
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


@pytest.fixture(scope="module")
def bars():
    return small_bars()


def test_runs_clean_strategy_and_writes_nothing(tmp_path, bars):
    s = strategy_from(tmp_path, CLEAN_BODY)
    before = set(os.listdir(tmp_path))
    out = run_signals(s, s.spec.defaults(), bars)
    assert out.index.equals(bars.index)
    assert list(out.columns) == ["entry", "exit", "stop"]
    assert out["entry"].dtype == bool
    assert set(os.listdir(tmp_path)) == before


def test_static_violation_raises_without_spawning(tmp_path, bars):
    s = strategy_from(tmp_path, "import os\n" + CLEAN_BODY)
    with pytest.raises(SandboxError) as e:
        run_signals(s, s.spec.defaults(), bars)
    assert e.value.reason == "static"


def test_timeout_kills_child(tmp_path, bars):
    s = strategy_from(tmp_path, "def signals(bars, p):\n    while True:\n        pass\n")
    with pytest.raises(SandboxError) as e:
        run_signals(s, s.spec.defaults(), bars, cfg=SandboxConfig(timeout_s=5))
    assert e.value.reason == "timeout"


def test_memory_limit(tmp_path, bars):
    # holds ~800 MB while working; a brief spike between 50 ms watchdog samples can slip through on
    # Windows (POSIX has a hard RLIMIT_AS in the child)
    s = strategy_from(tmp_path, "def signals(bars, p):\n    a = np.ones(100_000_000)\n    while True:\n"
                                "        a = a + 1\n")
    with pytest.raises(SandboxError) as e:
        run_signals(s, s.spec.defaults(), bars, cfg=SandboxConfig(max_rss_mb=400))
    assert e.value.reason == "memory"


def test_runtime_error_is_crash(tmp_path, bars):
    s = strategy_from(tmp_path, "def signals(bars, p):\n    return bars['nope']\n")
    with pytest.raises(SandboxError) as e:
        run_signals(s, s.spec.defaults(), bars)
    assert e.value.reason == "crash" and "nope" in e.value.detail


@pytest.mark.parametrize("body", [
    "def signals(bars, p):\n    import os\n    return os.listdir('.')\n",
    "def signals(bars, p):\n    open('pwned.txt', 'w').write('x')\n    return bars\n",
    "def signals(bars, p):\n    return eval('1')\n",
])
def test_runtime_layer_blocks_without_static_check(tmp_path, bars, body):
    s = strategy_from(tmp_path, body)
    with pytest.raises(SandboxError) as e:
        run_signals(s, s.spec.defaults(), bars, _skip_static=True)
    assert e.value.reason == "crash"
    assert not (tmp_path / "pwned.txt").exists()


def test_child_environment_has_no_secrets(tmp_path, bars, monkeypatch):
    # even if a strategy reaches os (here: static check skipped), the parent's secrets are not in its env
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "secret-token")
    body = CLEAN_BODY.replace(
        "    return out",
        '    env = pd.core.common.sys.modules["os"].environ\n'
        '    out["stop"] = float(len(env.get("DHAN_ACCESS_TOKEN", "")))\n'
        "    return out")
    s = strategy_from(tmp_path, body)
    out = run_signals(s, s.spec.defaults(), bars, _skip_static=True)
    assert (out["stop"] == 0.0).all()


def test_print_does_not_corrupt_output(tmp_path, bars):
    s = strategy_from(tmp_path, CLEAN_BODY.replace("    return out", "    print('hello')\n    return out"))
    out = run_signals(s, s.spec.defaults(), bars)
    assert len(out) == len(bars)


def test_non_dataframe_is_bad_output(tmp_path, bars):
    s = strategy_from(tmp_path, "def signals(bars, p):\n    return 42\n")
    with pytest.raises(SandboxError) as e:
        run_signals(s, s.spec.defaults(), bars)
    assert e.value.reason == "bad_output"


def test_incremental_returns_only_rows_from_k(tmp_path, bars):
    s = strategy_from(tmp_path, CLEAN_BODY)
    dates = bars.index.get_level_values("date").unique()
    k = len(dates) - 10
    out = run_signals(s, s.spec.defaults(), bars, incremental_from=k)
    got = out.index.get_level_values("date").unique()
    assert list(got) == list(dates[k:])
    batch = run_signals(s, s.spec.defaults(), bars)
    pd.testing.assert_frame_equal(out, batch.loc[out.index])
