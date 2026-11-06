from datetime import date

import pytest

from tradingsystem.universe import Universe, load_sectors


@pytest.fixture
def u(tmp_path):
    p = tmp_path / "changes.csv"
    p.write_text(
        "date,symbol,action\n"
        "2020-01-01,A,add\n2020-01-01,B,add\n"
        "2021-06-01,B,remove\n2021-06-01,C,add\n"
    )
    return Universe.from_csv(p)


def test_members_point_in_time(u):
    assert u.members(date(2021, 5, 31)) == {"A", "B"}
    assert u.members(date(2021, 6, 1)) == {"A", "C"}


def test_members_accepts_timestamp(u):
    import pandas as pd
    assert u.members(pd.Timestamp("2021-06-01")) == {"A", "C"}


def test_members_before_first_snapshot_raises(u):
    with pytest.raises(ValueError):
        u.members(date(2019, 12, 31))


def test_all_symbols(u):
    assert u.all_symbols() == {"A", "B", "C"}


def test_bad_action_rejected(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("date,symbol,action\n2020-01-01,A,added\n")
    with pytest.raises(ValueError):
        Universe.from_csv(p)


def test_load_sectors(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("symbol,sector\nINFY,IT\nHDFCBANK,Financials\n")
    assert load_sectors(p) == {"INFY": "IT", "HDFCBANK": "Financials"}
