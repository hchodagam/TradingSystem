# Plan 1 acceptance run

## Real data (Task 14, step 5): not run yet

It needs these owner actions first:

1. Fill `reference/nifty100_changes.csv` (point-in-time membership) and `reference/sectors.csv`.
2. Put `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` in `.env`.
3. Run `uv run python scripts/probe_dhan.py`, then set `research_start` and `prices_adjusted_by_dhan` in config.
4. Add at least 2 real contract notes to `tests/fixtures/contract_notes/` and check `uv run pytest tests/test_costs.py`.

Then run:

```
uv run ts evaluate strategies/reference/sma_crossover.py
```

Paste the scorecard below. Whatever it scores, thresholds stay as they are.

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
