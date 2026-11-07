# Plan 1 acceptance run

## Real data (Task 14, step 5): run 2026-11-07

`uv run ts evaluate strategies/reference/sma_crossover.py` on Dhan daily bars, 2012-01-01 to 2022-12-31. The universe is point-in-time Nifty 100 (202 symbols over the period; 188 have data). Result: **Gate 1 FAIL**, exit code 1.

Research backtest: 102 trades, final equity ₹3,29,992 from ₹1,00,000 (about 11.5% a year), 9 kill-switch trips (not enforced in backtests).

| check | value | threshold | result |
|---|---|---|---|
| sharpe | 0.750 | 1.0 | FAIL |
| max_drawdown | 0.258 | 0.20 | FAIL |
| trades | 102 | 100 | PASS |
| profit_factor | 2.107 | 1.3 | PASS |
| beats_benchmark | -0.106 (Nifty 100 buy-and-hold Sharpe ≈ 0.856) | 0 | FAIL |

**What this means:** the pipeline works on real data, but the spec's success criterion "the reference strategy passes the full pipeline" does not hold for a plain 50/200 crossover. It made money (profit factor 2.1), but with a lower Sharpe than simply holding the Nifty 100 and a drawdown above the 20% limit. The thresholds were not changed. The owner decides whether to keep the criterion, change the reference strategy, or accept "fails Gate 1 on its merits" as the expected result.

### Data the run used

- **Membership:** `reference/nifty100_changes.csv` was rebuilt backwards from today's list using 56 NSE Indices notices (2012–2026). The rebuild records each change with its notice in `scripts/build_nifty100_history.py`. It is internally consistent: 100 members at every step, apart from documented DVR and spin-off windows.
- **Delisted members:** 14 former members are delisted and have no Dhan data (HDFC Ltd, Cairn, Ranbaxy, ING Vysya, Mindtree…). They are acknowledged in `reference/data_exceptions.csv` and cannot be traded, which leaves a small survivorship bias.
- **Sectors:** NSE industry classification for 184 symbols; 18 delisted names were assigned by hand (marked `manual`).
- **Data cleaning:** suspicious bars were cross-checked against NSE bhavcopies.
  - 24 stray MOTHERSON bars are dropped (`reference/bar_fixes.csv`).
  - Two Dhan adjustment errors are fixed in `reference/corporate_actions.csv`: SUNDARMFIN 2012 and MOTHERSON 2024.
  - Verified genuine moves and data gaps are acknowledged in `reference/data_exceptions.csv`.
  - Research and holdout data both validate with 0 issues.
- **Contract notes:** still pending, so the cost model is checked only against hand-computed figures.

## Synthetic data (2026-11-07)

`uv run ts evaluate strategies/reference/sma_crossover.py --synthetic-seed 3` on a zero-drift random walk (no trends to catch), as expected:

| check | value | threshold | result |
|---|---|---|---|
| sharpe | -0.286 | 1.0 | FAIL |
| max_drawdown | 0.486 | 0.20 | FAIL |
| trades | 113 | 100 | PASS |
| profit_factor | 0.762 | 1.3 | FAIL |
| beats_benchmark | +0.240 | 0 | PASS |

`tests/test_cli.py` covers the other paths: the dip buyer on planted-edge data passes Gates 1 and 2, random entry fails Gate 1 (exit 1), and the look-ahead strategy is rejected at acceptance (exit 2).
