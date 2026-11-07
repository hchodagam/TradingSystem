from pathlib import Path

import pandas as pd

from tradingsystem.config import load_config
from tradingsystem.risk.admission import Holding, OrderIntent, admit

cfg = load_config(Path(__file__).parents[2] / "config").risk
D = pd.Timestamp("2024-01-02")


def h(sym):
    return Holding(sym, 10, 100.0, 90.0, D)


def o(sym):
    return OrderIntent(sym, 10, 90.0, D)


SECTORS = {f"S{i}": f"sec{i}" for i in range(10)} | {"B1": "bank", "B2": "bank", "B3": "bank"}


def test_max_six_positions():
    holdings = {s: h(s) for s in ["S0", "S1", "S2", "S3", "S4"]}
    acc, rej = admit([o("S5"), o("S6"), o("S7")], holdings, SECTORS, cfg)
    assert [a.symbol for a in acc] == ["S5"]
    assert [(r.symbol, why) for r, why in rej] == [("S6", "max_positions"), ("S7", "max_positions")]


def test_max_two_per_sector():
    acc, rej = admit([o("B2"), o("B3")], {"B1": h("B1")}, SECTORS, cfg)
    assert [a.symbol for a in acc] == ["B2"] and rej[0][1] == "max_per_sector"


def test_already_held():
    acc, rej = admit([o("S1")], {"S1": h("S1")}, SECTORS, cfg)
    assert acc == [] and rej[0][1] == "already_held"


def test_unknown_sector():
    acc, rej = admit([o("ZZZ")], {}, SECTORS, cfg)
    assert acc == [] and rej[0][1] == "unknown_sector"


def test_duplicate_intent_same_day():
    acc, rej = admit([o("S1"), o("S1")], {}, SECTORS, cfg)
    assert len(acc) == 1 and rej[0][1] == "already_held"


def test_inputs_not_mutated():
    holdings = {"S0": h("S0")}
    admit([o("S1")], holdings, SECTORS, cfg)
    assert list(holdings) == ["S0"]
