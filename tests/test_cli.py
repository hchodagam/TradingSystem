"""CLI end-to-end on synthetic data (no Dhan). Small Gate 2 grid via a copied config dir."""
import shutil
from pathlib import Path

import pytest

from tradingsystem.cli import main

ROOT = Path(__file__).parent.parent
FIX = ROOT / "tests" / "fixtures" / "strategies"


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    shutil.copytree(ROOT / "config", d)
    g = (d / "gates.yaml").read_text()
    g = g.replace("grid_values_per_param: 5", "grid_values_per_param: 3").replace("grid_max_combos: 200",
                                                                                   "grid_max_combos: 10")
    (d / "gates.yaml").write_text(g)
    return d


def run(capsys, *args):
    code = main(list(args))
    return code, capsys.readouterr().out


def test_planted_edge_passes(capsys, config_dir, tmp_path):
    code, out = run(capsys, "evaluate", str(FIX / "dip_buyer.py"), "--config", str(config_dir),
                    "--synthetic-seed", "7", "--synthetic-edge", "--holdout", "--state-dir", str(tmp_path / "state"))
    assert "Gate 1 PASS" in out and "Gate 2 PASS" in out and "Gate 3" in out
    assert code in (0, 1)  # holdout outcome on synthetic data is not the point; gates 1-2 are


def test_random_entry_fails_with_exit_1(capsys, config_dir):
    code, out = run(capsys, "evaluate", str(FIX / "random_entry.py"), "--config", str(config_dir),
                    "--synthetic-seed", "1")
    assert code == 1 and "Gate 1 FAIL" in out and "Gate 2" not in out


def test_leaky_strategy_exits_2(capsys, config_dir):
    code, out = run(capsys, "evaluate", str(FIX / "leaky.py"), "--config", str(config_dir), "--synthetic-seed", "1")
    assert code == 2 and "lookahead" in out


def test_bad_output_on_full_history_exits_2(capsys, config_dir):
    # I2: output is checked on the full research run, not only the acceptance smoke run
    code, out = run(capsys, "evaluate", str(FIX / "bad_output_long.py"), "--config", str(config_dir),
                    "--synthetic-seed", "1")
    assert code == 2 and "stop" in out


@pytest.mark.parametrize("exc", [FileNotFoundError("x.parquet"), KeyError("OLDCO"), RuntimeError("boom")])
def test_unexpected_errors_exit_2_not_1(capsys, monkeypatch, exc):
    # I5: exit 1 means "a gate failed"; anything else is an error (2)
    import tradingsystem.cli as cli

    def explode(args):
        raise exc

    monkeypatch.setattr(cli, "evaluate", explode)
    assert cli.main(["evaluate", "x.py"]) == 2
    assert "error" in capsys.readouterr().out.lower()


def test_missing_file_exits_2(capsys, config_dir):
    code, out = run(capsys, "evaluate", "nope.py", "--config", str(config_dir), "--synthetic-seed", "1")
    assert code == 2 and "nope.py" in out
