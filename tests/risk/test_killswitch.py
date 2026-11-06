from pathlib import Path

from tradingsystem.config import load_config
from tradingsystem.risk.killswitch import KillSwitch

cfg = load_config(Path(__file__).parents[2] / "config").risk


def test_killswitch_trips_at_10pct():
    ks = KillSwitch(cfg, enforce=True)
    ks.update(100_000)
    ks.update(90_000)
    assert ks.tripped and ks.trips == 1 and not ks.allows_entries()


def test_killswitch_not_at_9_99pct():
    ks = KillSwitch(cfg, enforce=True)
    ks.update(100_000)
    ks.update(90_010)
    assert not ks.tripped and ks.allows_entries()


def test_killswitch_stays_tripped_on_recovery_until_resume():
    ks = KillSwitch(cfg, enforce=True)
    for e in (100_000, 89_000, 120_000):
        ks.update(e)
    assert ks.tripped
    ks.resume()
    assert not ks.tripped and ks.allows_entries()


def test_drawdown_measured_from_new_peak_after_resume():
    ks = KillSwitch(cfg, enforce=True)
    for e in (100_000, 89_000):
        ks.update(e)
    ks.resume()
    ks.update(85_000)  # 4.5% below the 89k level at resume: not a new trip
    assert not ks.tripped


def test_not_enforced_counts_trips_but_allows_entries():
    ks = KillSwitch(cfg, enforce=False)
    for e in (100_000, 89_000, 100_000, 120_000, 100_000):
        ks.update(e)
    assert ks.allows_entries() and ks.trips == 2
