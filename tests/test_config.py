import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tradingsystem.config import load_config

CONFIG = Path(__file__).parent.parent / "config"


def test_loads_all_sections():
    cfg = load_config(CONFIG)
    assert cfg.costs.stt_pct == 0.001
    assert cfg.risk.max_open_positions == 6
    assert cfg.gates.gate1.min_sharpe == 1.0
    assert cfg.gates.holdout_start > cfg.gates.research_end


def test_unknown_key_rejected(tmp_path):
    shutil.copytree(CONFIG, tmp_path, dirs_exist_ok=True)
    with open(tmp_path / "risk.yaml", "a") as f:
        f.write("\nfoo: 1\n")
    with pytest.raises(ValidationError):
        load_config(tmp_path)


def test_config_is_frozen():
    cfg = load_config(CONFIG)
    with pytest.raises(ValidationError):
        cfg.risk.max_open_positions = 7


def test_holdout_must_follow_research(tmp_path):
    shutil.copytree(CONFIG, tmp_path, dirs_exist_ok=True)
    text = (tmp_path / "gates.yaml").read_text().replace("holdout_start: 2023-01-01", "holdout_start: 2020-01-01")
    (tmp_path / "gates.yaml").write_text(text)
    with pytest.raises(ValidationError):
        load_config(tmp_path)
