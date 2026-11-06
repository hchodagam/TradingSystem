# Strategy Engine — Plan 1 of 4: Deterministic Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic half of the engine: Dhan daily data with cache and validation, the Indian cost model, the strategy format with sandbox and look-ahead check, risk sizing, the event-driven backtest, and Gates 1–3. Done means a strategy file can be run through backtest and gates from the command line.

**Architecture:** One Python package `tradingsystem` with one sub-package per spec unit (`data`, `strategy`, `risk`, `backtest`, `validate`). Pure functions wherever possible; config comes from owner-edited YAML loaded into Pydantic models. Strategy code never runs in the parent process: the SPEC is read with `ast.literal_eval`, and `signals()` runs in a subprocess sandbox.

**Tech Stack:** Python 3.12, uv, pandas, numpy, pyarrow, pydantic v2, PyYAML, httpx, exchange_calendars, scipy (DSR), ruff, pytest.

**Spec:** `2026-11-06-strategy-engine-design.md` (repo root)

### Step 1 is split into 4 plans

| Plan | Scope | Depends on |
|---|---|---|
| **1 (this)** | Config, costs, data, strategy format + sandbox, risk, backtest, Gates 1–3 | — |
| 2 | `llm/` adapter + cost caps, 4 agents, model bake-off | 1 |
| 3 | Postgres, orchestrator + case state machine, `broker/` PaperBroker, `paper/` daily cycle, drift, replay, Gate 4, ideas library | 1, 2 |
| 4 | FastAPI + React dashboard (5 screens), single-user login, Docker Compose, VPS | 3 |

Plans 2–4 get written when this one is done, so they can build on the real interfaces.

## Global Constraints

- Python 3.12; dependencies managed with `uv`; all commands run as `uv run ...`.
- Market: Nifty 100 universe, daily bars (`timeframe: "1D"`), NSE trading days.
- Fills at **next-day open**; slippage **0.1%** on next-day open (buy pays up, sell receives down).
- STT 0.1% buy and sell; delivery brokerage ₹0; exchange txn charge, SEBI fee, stamp duty (buy only), GST on charges, DP charge per sell scrip per day. All rates live in `config/costs.yaml`.
- Risk: 1% of equity per trade; shares = floor(1% × equity ÷ (entry − stop)); no stop → 2×ATR(14) used for sizing only; max 20% of equity per position; max 6 open positions; max 2 per sector; skip if < 1 share, value < ₹5,000, or upper circuit at open; kill-switch at 10% drawdown from peak.
- Strategies emit signals only. They never size, cost or place orders. A strategy may have at most 5 params before it's flagged as overfit risk.
- Strategy sandbox import allowlist: `pandas`, `numpy`, `math`. The TA library stays an open item; it's deferred to Plan 2.
- Gate thresholds (`config/gates.yaml`) and risk limits (`config/risk.yaml`) are owner-edited only; code reads them, never writes them.
- Holdout data is never returned by research-period loaders, and the holdout runs **once** per strategy hash.
- Secrets (`DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN`) come from `.env` and are never committed.
- Development is test-first for every unit.

## Review Focus

1. **A held symbol has no bar** (suspended, delisted, renamed). The position is kept and marked at its last close. After 5 trading days with no bar it is exited at that last close with reason `data_missing`. Owned by Task 10.
2. **The price gaps through the stop at the open.** The exit fills at open × (1 − slippage), not at the stop price. Owned by Task 10.
3. **There isn't enough cash at the fill** (the open gaps up after sizing on yesterday's close). Quantity shrinks to what is affordable, then the skip rules are rechecked. Owned by Task 10.
4. **A strategy writes or reads files through pandas or numpy** (`df.to_csv`, `pd.read_csv`, `np.save`). The static check rejects it before it runs. Owned by Task 7.
5. **A strategy returns malformed output**: a NaN in `entry`, the wrong index, non-bool flags, or missing columns. The smoke test rejects it and names the defect. Owned by Task 8.

---

### Task 1: Project scaffold and config models

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `config/costs.yaml`, `config/risk.yaml`, `config/gates.yaml`, `config/data.yaml`, `src/tradingsystem/__init__.py`, `src/tradingsystem/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(config_dir: Path = Path("config")) -> AppConfig` with fields `costs: CostRates`, `risk: RiskConfig`, `gates: GatesConfig`, `data: DataConfig`. All models are `pydantic.BaseModel` with `model_config = ConfigDict(frozen=True, extra="forbid")`.

Config values (copy into the YAML files):
- `costs.yaml`: `brokerage_delivery: 0`, `stt_pct: 0.001`, `exchange_txn_pct: 0.0000297`, `sebi_pct: 0.000001`, `stamp_buy_pct: 0.00015`, `gst_pct: 0.18`, `dp_per_sell_scrip_day: 12.5`, `slippage_pct: 0.001`, `rounding: {stt: 0, stamp: 0, other: 2}`. These are starting values; Task 2's contract-note test is the authority.
- `risk.yaml`: `risk_per_trade_pct: 0.01`, `atr_len: 14`, `atr_stop_mult: 2.0`, `max_position_pct: 0.20`, `max_open_positions: 6`, `max_per_sector: 2`, `min_position_value: 5000`, `upper_circuit_gap_pct: 0.10`, `kill_switch_dd_pct: 0.10`, `data_missing_exit_days: 5`, `initial_capital: 100000`.
- `gates.yaml`: `research_start: 2012-01-01`, `research_end: 2022-12-31`, `holdout_start: 2023-01-01`, `risk_free_rate: 0.0`; gate1 `{min_sharpe: 1.0, max_drawdown: 0.20, min_trades: 100, min_profit_factor: 1.3}`; gate2 `{wf_fit_years: 3, wf_test_years: 1, wf_min_oos_ratio: 0.5, wf_max_losing_years: 1, sens_step_pct: 0.20, sens_min_profitable: 0.70, mc_runs: 1000, mc_seed: 42, mc_dd_pct95_max: 0.30, cost_stress_slippage_mult: 2.0, dsr_min: 0.95, grid_values_per_param: 5, grid_max_combos: 200, overfit_param_count: 5, overfit_dsr_min: 0.99, overfit_sens_min_profitable: 0.80}`; gate3 `{min_sharpe_ratio_vs_research: 0.5}`.
- `data.yaml`: `cache_dir: data/cache`, `reference_dir: reference`, `dhan_base_url: https://api.dhan.co/v2`, `scrip_master_url: https://images.dhan.co/api-data/api-scrip-master.csv`, `max_days_per_request: 1825`, `retries: 4`, `backoff_seconds: 2.0`, `prices_adjusted_by_dhan: false`, `benchmark_symbol: NIFTY 100`, `abnormal_move_pct: 0.20`.

The `overfit_*` values and `upper_circuit_gap_pct` are my defaults where the spec gives no number. The owner confirms them in review.

- [ ] **Step 1: Init repo and project.** Run `git init`, then `uv init --package --name tradingsystem --python 3.12`, then `uv add pandas numpy pyarrow pydantic pyyaml httpx exchange_calendars scipy python-dotenv psutil` and `uv add --dev pytest ruff`. `.gitignore` covers `.env`, `data/`, `state/`, `.venv/`, `__pycache__/`. `.env.example` lists `DHAN_CLIENT_ID=` and `DHAN_ACCESS_TOKEN=`.
- [ ] **Step 2: Write the failing tests**

```python
def test_loads_all_sections():
    cfg = load_config(Path("config"))
    assert cfg.costs.stt_pct == 0.001
    assert cfg.risk.max_open_positions == 6
    assert cfg.gates.gate1.min_sharpe == 1.0
    assert cfg.gates.holdout_start > cfg.gates.research_end

def test_unknown_key_rejected(tmp_path):  # copy config/, add `foo: 1` to risk.yaml
    with pytest.raises(ValidationError): load_config(tmp_path)

def test_config_is_frozen():
    with pytest.raises(ValidationError): load_config(Path("config")).risk.max_open_positions = 7
```

- [ ] **Step 3: Run** `uv run pytest tests/test_config.py -v`. Expected: FAIL (import error).
- [ ] **Step 4: Implement** the models and `load_config` in `src/tradingsystem/config.py`. Add a validator that `holdout_start > research_end`.
- [ ] **Step 5: Run** the same command. Expected: 3 passed.
- [ ] **Step 6: Commit** `git add -A; git commit -m "chore: scaffold project and config models"`

---

### Task 2: Cost model

**Files:**
- Create: `src/tradingsystem/costs.py`, `tests/fixtures/contract_notes/README.md`
- Test: `tests/test_costs.py`

**Interfaces:**
- Consumes: `CostRates` (Task 1).
- Produces:
  - `Fill` is a frozen dataclass with `symbol: str`, `date: date`, `side: Literal["BUY","SELL"]`, `qty: int`, `price: float`.
  - `ChargeBreakdown` is a frozen dataclass with `stt, exchange, sebi, stamp, dp, gst, total: float`.
  - `day_charges(fills: list[Fill], rates: CostRates) -> ChargeBreakdown` aggregates per (symbol, date, side). DP is charged once per (symbol, date) with any SELL. GST is 18% of (brokerage + exchange + SEBI + DP), computed on unrounded components. STT and stamp are rounded to the rupee and everything else to 2 dp; the rounding happens per (symbol, date, side) group.
  - `apply_slippage(price: float, side, rates) -> float`

- [ ] **Step 1: Write the failing tests** using these hand-computed values:

```python
def test_buy_delivery():  # 100 @ 1000, turnover 1,00,000
    c = day_charges([Fill("INFY", D, "BUY", 100, 1000.0)], rates)
    assert (c.stt, c.exchange, c.sebi, c.stamp, c.dp, c.gst) == (100, 2.97, 0.10, 15, 0, 0.55)
    assert c.total == pytest.approx(118.62)

def test_sell_delivery_with_dp():  # 100 @ 1100
    c = day_charges([Fill("INFY", D, "SELL", 100, 1100.0)], rates)
    assert (c.stt, c.exchange, c.sebi, c.stamp, c.dp, c.gst) == (110, 3.27, 0.11, 0, 12.5, 2.86)
    assert c.total == pytest.approx(128.74)

def test_dp_once_per_scrip_per_day():  # two SELL fills, same symbol and day -> dp == 12.5
def test_slippage(): assert apply_slippage(100.0, "BUY", rates) == pytest.approx(100.1)

@pytest.mark.parametrize("note", sorted(Path("tests/fixtures/contract_notes").glob("*.json")))
def test_matches_real_contract_note(note):  # {"fills":[...], "expected":{...}} -> exact to 0.01
```

- [ ] **Step 2: Run** `uv run pytest tests/test_costs.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** in `src/tradingsystem/costs.py`.
- [ ] **Step 4: Run** the same command. Expected: hand-computed tests pass; the parametrized test collects 0 cases.
- [ ] **Step 5: Owner action.** Transcribe at least 2 real Dhan delivery contract notes (one buy day, one sell day) into `tests/fixtures/contract_notes/*.json`, following the format in the README. Re-run; any mismatch is fixed by changing `config/costs.yaml` or the rounding rule, never the expected values.
- [ ] **Step 6: Commit** `git commit -am "feat: Indian delivery cost model"` (add the new files first).

---

### Task 3: Calendar, universe and sectors

**Files:**
- Create: `src/tradingsystem/calendar.py`, `src/tradingsystem/universe.py`, `reference/nifty100_changes.csv`, `reference/sectors.csv`, `reference/corporate_actions.csv`
- Test: `tests/test_calendar.py`, `tests/test_universe.py`

**Interfaces:**
- Produces:
  - `trading_days(start: date, end: date) -> pd.DatetimeIndex` uses `exchange_calendars` `"XBOM"`, which shares NSE's equity holidays.
  - `Universe.from_csv(path) -> Universe`. The CSV columns are `date,symbol,action`, where `action` is `add` or `remove` and the first date is the initial snapshot. `Universe.members(d: date) -> frozenset[str]` and `Universe.all_symbols() -> frozenset[str]`.
  - `load_sectors(path) -> dict[str, str]` (symbol → sector).

- [ ] **Step 1: Write the failing tests**

```python
def test_weekend_excluded(): assert pd.Timestamp("2024-01-06") not in trading_days(date(2024,1,1), date(2024,1,31))
def test_republic_day_holiday(): assert pd.Timestamp("2024-01-26") not in trading_days(date(2024,1,1), date(2024,1,31))
def test_members_point_in_time():  # tmp CSV: 2020-01-01 add A,B; 2021-06-01 remove B; 2021-06-01 add C
    assert u.members(date(2021,5,31)) == {"A","B"}; assert u.members(date(2021,6,1)) == {"A","C"}
def test_members_before_first_snapshot_raises(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/test_calendar.py tests/test_universe.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Owner action** (spec open item). Fill `reference/nifty100_changes.csv` from NSE index change notices back to `research_start`, and fill `reference/sectors.csv` from NSE's industry classification. Commit headers-only files until then.
- [ ] **Step 6: Commit** `git commit -m "feat: trading calendar and point-in-time universe"`

---

### Task 4: Dhan client, bar store and corporate-action adjustment

**Files:**
- Create: `src/tradingsystem/data/__init__.py`, `data/dhan.py`, `data/store.py`, `data/adjust.py`, `scripts/probe_dhan.py`
- Test: `tests/data/test_dhan.py`, `tests/data/test_store.py`, `tests/data/test_adjust.py`

**Interfaces:**
- Consumes: `DataConfig`, `GatesConfig` (Task 1); `Universe` (Task 3).
- Produces:
  - **Bars** (the type every later task uses): a `pd.DataFrame` with MultiIndex `(date: Timestamp, symbol: str)` sorted by date then symbol, and float columns `open, high, low, close, volume`.
  - `DhanClient(cfg: DataConfig, client_id: str, token: str, http: httpx.Client | None = None)`:
    - `.security_id(symbol: str) -> str` resolves through the cached scrip master, using `SEM_EXM_EXCH_ID == "NSE"` and `SEM_INSTRUMENT_NAME == "EQUITY"`, or `"INDEX"` for the benchmark.
    - `.daily(symbol: str, start: date, end: date) -> pd.DataFrame` POSTs to `/charts/historical` with body `{securityId, exchangeSegment: "NSE_EQ"|"IDX_I", instrument: "EQUITY"|"INDEX", expiryCode: 0, oi: false, fromDate, toDate}` and header `access-token`. The response's parallel arrays `open/high/low/close/volume/timestamp` (epoch seconds, converted to an IST date) become rows. Requests are chunked to `max_days_per_request`.
    - It retries 429/5xx and network errors `retries` times with exponential backoff from `backoff_seconds`. A 401 raises `DhanAuthError("access token expired or invalid")` without retrying.
  - `BarStore(cfg: AppConfig, client: DhanClient | None)`:
    - `.refresh(symbols, end: date) -> None` fetches only the dates missing from `cache_dir/{symbol}.parquet`.
    - `.research_bars(symbols) -> Bars` clamps to `[research_start, research_end]`.
    - `.holdout_bars(symbols) -> Bars` covers `[research_start, latest]` and is called only by Task 13.
    - `.benchmark(start, end) -> pd.Series` returns the close series.
    - It applies `adjust()` unless `prices_adjusted_by_dhan`.
  - `adjust(bars: Bars, actions: pd.DataFrame) -> Bars`. Actions have columns `symbol, ex_date, factor`, where factor = new shares per old share (a 1:5 split → 5, a 1:1 bonus → 2). Prices before ex_date are divided by the factor and volume is multiplied by it. Dividends are not adjusted.

- [ ] **Step 1: Write the failing tests.** Use `httpx.MockTransport`; there are no live calls in tests.

```python
def test_daily_parses_arrays(): ...      # 2 timestamps -> 2 rows with correct IST dates and values
def test_chunks_long_range(): ...        # 2012-01-01..2022-12-31 with max 1825 days -> 3 requests, contiguous
def test_retries_then_succeeds(): ...    # 500, 500, 200 -> returns data; 3 calls
def test_401_raises_auth_error_without_retry(): ...
def test_store_fetches_only_missing_dates(tmp_path): ...  # second refresh to same end -> 0 requests
def test_research_bars_never_include_holdout(tmp_path):
    assert store.research_bars(["A"]).index.get_level_values("date").max() <= pd.Timestamp(cfg.gates.research_end)
def test_split_adjustment():  # close 1000 before ex_date, 200 after, factor 5 -> all closes 200; volume x5 before
```

- [ ] **Step 2: Run** `uv run pytest tests/data -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Write** `scripts/probe_dhan.py`. It reads `.env` and prints, for 3 symbols plus the benchmark, the earliest date Dhan returns. For any symbol listed in `reference/corporate_actions.csv` it prints the close on the day before and the day of the ex_date, so the owner can see whether Dhan already adjusts.
- [ ] **Step 6: Owner action** (spec open item). Run `uv run python scripts/probe_dhan.py`. Set `research_start` in `gates.yaml` to a date no earlier than the latest "earliest date" across the universe, and set `prices_adjusted_by_dhan` from the split check.
- [ ] **Step 7: Commit** `git commit -m "feat: Dhan daily data, parquet cache, split adjustment"`

---

### Task 5: Data validation

**Files:**
- Create: `src/tradingsystem/data/checks.py`
- Test: `tests/data/test_checks.py`

**Interfaces:**
- Consumes: Bars (Task 4), `trading_days`, `Universe` (Task 3).
- Produces: `validate_bars(bars: Bars, universe: Universe, start: date, end: date, actions: pd.DataFrame, abnormal_move_pct: float) -> ValidationReport`. `ValidationReport` is a frozen dataclass with `ok: bool` and `issues: list[Issue]`; `Issue` has `kind: Literal["missing_symbol","missing_day","ohlc_invalid","abnormal_move","non_positive"]`, `symbol`, `date` and `detail: str`.

- [ ] **Step 1: Write the failing tests**, one per kind. Also test that a clean frame gives `ok is True` and `issues == []`, and that a 25% move on a corporate-action ex_date is **not** flagged.
- [ ] **Step 2: Run** `uv run pytest tests/data/test_checks.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `missing_symbol` means a member on a trading day with no row. `missing_day` means a trading day with no rows at all. `ohlc_invalid` means it's not true that low ≤ min(open, close) ≤ max(open, close) ≤ high. `abnormal_move` means |close/prev_close − 1| > `abnormal_move_pct` and that date is not an ex_date for the symbol.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: bar validation report"`

---

### Task 6: Strategy SPEC schema and loader

**Files:**
- Create: `src/tradingsystem/strategy/__init__.py`, `strategy/spec.py`, `strategy/loader.py`
- Test: `tests/strategy/test_spec.py`, `tests/strategy/test_loader.py`

**Interfaces:**
- Produces:
  - `ParamSpec(default: int|float, range: tuple[number, number])`, validated so that `lo < hi` and `lo ≤ default ≤ hi`. The type is int when all three values are ints.
  - `StrategySpec(name: str (^[a-z][a-z0-9_]*$), family: str, hypothesis: str (non-blank), universe: Literal["nifty100"], timeframe: Literal["1D"], params: dict[str, ParamSpec], max_holding_days: int ≥ 1)`, with the property `overfit_risk: bool` (len(params) > 5).
  - `Params(dict)` allows attribute access (`p.rsi_len`). `StrategySpec.defaults() -> Params`.
  - `LoadedStrategy(spec: StrategySpec, code: str, hash: str, path: Path)` and `load_strategy(path: Path) -> LoadedStrategy`. The SPEC is extracted by parsing the file with `ast` and calling `ast.literal_eval` on the `SPEC = {...}` assignment, so the file is never imported. The hash is the sha256 hex of the code with `\r\n` normalised to `\n`.
  - `StrategyFormatError(Exception)`

- [ ] **Step 1: Write the failing tests.** Cover: the spec example in section 3 of the design doc validates; a default outside its range fails; `lo >= hi` fails; a blank hypothesis fails; 6 params gives `overfit_risk`; a missing `SPEC` or missing `def signals(bars, p)` raises `StrategyFormatError`; a non-literal SPEC (`SPEC = make()`) raises; CRLF and LF copies of a file give the same hash; a file containing `raise SystemExit` at module level still loads, which proves it was never executed.
- [ ] **Step 2: Run** `uv run pytest tests/strategy -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: strategy SPEC schema and static loader"`

---

### Task 7: Static check and sandbox runner

**Files:**
- Create: `src/tradingsystem/strategy/static_check.py`, `strategy/runner.py` (child entrypoint), `strategy/sandbox.py` (parent side)
- Test: `tests/strategy/test_static_check.py`, `tests/strategy/test_sandbox.py`, `tests/fixtures/strategies/escape_*.py`

**Interfaces:**
- Consumes: `LoadedStrategy`, `Params` (Task 6); Bars.
- Produces:
  - `static_check(code: str) -> list[str]` returns violations, and an empty list means clean. It rejects:
    - imports outside {`pandas`, `numpy`, `math`};
    - the names `open, exec, eval, compile, __import__, globals, locals, vars, getattr, setattr, delattr, input, breakpoint`;
    - any attribute name starting and ending with `__`;
    - attributes matching `^read_`, `^to_(csv|excel|parquet|pickle|json|hdf|sql|feather|stata|html|xml|latex|markdown|clipboard|orc)$`, `^(save|savez|savez_compressed|savetxt|load|loadtxt|genfromtxt|fromfile|tofile|memmap)$`;
    - the attribute chains `pd.io` and `np.lib`.
  - `run_signals(strategy: LoadedStrategy, params: Params, bars: Bars, *, incremental_from: int | None = None, cfg: SandboxConfig = SandboxConfig()) -> pd.DataFrame`.
    - The child is started as `[sys.executable, "-I", "-m", "tradingsystem.strategy.runner"]` with cwd set to a fresh temp dir.
    - The parent sends bars as Arrow IPC on stdin, with the code and params in a length-prefixed JSON header. The child returns Arrow IPC on stdout.
    - The child execs the code with a restricted `__builtins__` whose `__import__` permits only the allowlist, and replaces `socket.socket` with a raiser before exec.
    - `incremental_from=k` makes the child loop t = k..n−1 over unique dates and keep only date t's rows of `signals(bars[:t+1])`.
  - `SandboxConfig(timeout_s: float = 120, max_rss_mb: int = 2048)`. Enforcement uses the timeout plus a `psutil` RSS watchdog on every platform, and on Linux also `resource.setrlimit(RLIMIT_AS)` in `preexec_fn`. The low-privilege user and no-network container are Plan 4 (Docker).
  - `SandboxError(Exception)` has `.reason: Literal["static","timeout","memory","crash","bad_output"]` and `.detail: str`.

- [ ] **Step 1: Write the failing tests.** Create one fixture strategy per escape attempt: `import os`, `import socket`, `__import__("os")`, `open("x","w")`, `bars.to_csv("x.csv")`, `pd.read_csv("x")`, `np.save("x", a)`, `().__class__.__bases__`, `while True: pass`, and allocating 4 GB.

```python
@pytest.mark.parametrize("name", ["import_os","import_socket","dunder_import","open_write","to_csv","read_csv","np_save","dunder_escape"])
def test_static_rejects(name): assert static_check(fixture(name).code)
def test_timeout_kills_child(): ...   # SandboxConfig(timeout_s=2) -> SandboxError reason "timeout"
def test_memory_limit(): ...          # max_rss_mb=256 -> reason "memory"
def test_runs_clean_strategy_and_writes_nothing(tmp_path): ...  # output index == bars index; no new files in cwd
def test_incremental_returns_only_rows_from_k(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/strategy/test_static_check.py tests/strategy/test_sandbox.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `run_signals` calls `static_check` first and raises `SandboxError("static", ...)` without spawning a child.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: strategy static check and subprocess sandbox"`

---

### Task 8: Smoke test, look-ahead check and acceptance pipeline

**Files:**
- Create: `src/tradingsystem/strategy/acceptance.py`, `tests/synthetic.py`, `tests/fixtures/strategies/{leaky.py,bad_output_*.py}`, `strategies/reference/sma_crossover.py`
- Test: `tests/strategy/test_acceptance.py`

**Interfaces:**
- Consumes: Tasks 6–7.
- Produces:
  - `make_bars(symbols: list[str], start: date, end: date, seed: int, drift: float = 0.0, vol: float = 0.02, planted_edge: bool = False) -> Bars` in `tests/synthetic.py`. Prices follow a geometric random walk on `trading_days`. With `planted_edge`, after any day with close/prev_close − 1 < −3%, the next 5 days get +1%/day drift.
  - `check_output(out: pd.DataFrame, bars: Bars) -> list[str]`. Output must have the same index as bars, columns `entry` (bool, no NaN), `exit` (bool, no NaN) and `stop` (float, NaN allowed), and no `stop ≤ 0`.
  - `lookahead_check(s: LoadedStrategy, params: Params, bars: Bars, window: int = 250) -> list[str]` runs batch and incremental (from n − window) and returns the dates and symbols where they differ (stop compared with `np.isclose`, NaN == NaN).
  - `accept(path: Path, bars: Bars) -> AcceptanceResult`. `AcceptanceResult` has `ok: bool`, `stage: Literal["schema","lint","smoke","lookahead","passed"]`, `errors: list[str]` and `strategy: LoadedStrategy | None`. The stages run in this order: load → `ruff check --select E,F,B` on the file → static check + run on 2 years of synthetic bars + `check_output` → lookahead. The run stops at the first failing stage. The Critic stage is Plan 2, and the 3-retry loop with the Code agent is Plan 2.
  - The reference strategy: SPEC `name: sma_crossover`, `family: trend_following`, params `fast {default 50, range [20, 100]}` and `slow {default 200, range [100, 300]}`, `max_holding_days: 250`. Entry when SMA(fast) crosses above SMA(slow); exit when it crosses below; stop is always NaN. Rows during warm-up are False.

- [ ] **Step 1: Write the failing tests**

```python
def test_reference_strategy_accepted(): assert accept(REF, bars).stage == "passed"
def test_leaky_strategy_caught():  # uses close.shift(-1)
    r = accept(fixture("leaky"), bars); assert (r.ok, r.stage) == (False, "lookahead")
@pytest.mark.parametrize("name,msg", [("bad_output_nan_entry","NaN in entry"),("bad_output_index","index"),("bad_output_int_flags","bool"),("bad_output_missing_stop","stop")])
def test_bad_output_rejected(name, msg):
    r = accept(fixture(name), bars); assert r.stage == "smoke" and any(msg in e for e in r.errors)
def test_lint_failure_stops_pipeline(): ...  # undefined name -> stage "lint"
```

- [ ] **Step 2: Run** `uv run pytest tests/strategy/test_acceptance.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: strategy acceptance pipeline and SMA reference strategy"`

---

### Task 9: Risk — sizing, admission and kill-switch

**Files:**
- Create: `src/tradingsystem/risk/__init__.py`, `risk/sizing.py`, `risk/admission.py`, `risk/killswitch.py`
- Test: `tests/risk/test_sizing.py`, `tests/risk/test_admission.py`, `tests/risk/test_killswitch.py`

**Interfaces:**
- Consumes: `RiskConfig` (Task 1).
- Produces:
  - `size_position(entry: float, stop: float, atr: float, equity: float, cfg: RiskConfig) -> int`. A NaN stop is replaced by `entry − atr_stop_mult × atr`. The result is capped by `max_position_pct`, and it returns 0 when the skip rules hit (< 1 share or value < `min_position_value`).
  - `OrderIntent(symbol: str, qty: int, stop: float, signal_date: Timestamp)` and `Holding(symbol: str, qty: int, entry_price: float, stop: float, entry_date: Timestamp)`.
  - `admit(intents: list[OrderIntent], holdings: dict[str, Holding], sectors: dict[str, str], cfg: RiskConfig) -> tuple[list[OrderIntent], list[tuple[OrderIntent, str]]]` returns (accepted, rejected with reason). Intents are processed in the order given, which the engine sorts by symbol for determinism. Reasons are `"max_positions"`, `"max_per_sector"`, `"already_held"` and `"unknown_sector"`.
  - `is_upper_circuit(open_: float, prev_close: float, cfg) -> bool` is `open_ >= prev_close × (1 + upper_circuit_gap_pct)`. This is a proxy, because Dhan bars carry no circuit band.
  - `KillSwitch(cfg, enforce: bool)` has `.update(equity: float) -> None`, `.tripped: bool`, `.trips: int` and `.resume() -> None`. `allows_entries()` returns `not (enforce and tripped)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_risk_sizing_capped_by_position_pct(): assert size_position(500, 480, 10, 100_000, cfg) == 40
def test_risk_sizing_exact(): assert size_position(200, 190, 5, 100_000, cfg) == 100
def test_nan_stop_uses_2_atr(): assert size_position(200, float("nan"), 5, 100_000, cfg) == 100
def test_skip_below_min_value(): assert size_position(4000, 3000, 50, 100_000, cfg) == 0
def test_skip_below_one_share(): assert size_position(9000, 7000, 50, 100_000, cfg) == 0
def test_max_six_positions(): ...      # 5 held + 3 intents -> 1 accepted, 2 "max_positions"
def test_max_two_per_sector(): ...
def test_killswitch_trips_at_10pct(): ks.update(100_000); ks.update(90_000); assert ks.tripped
def test_killswitch_not_at_9_99pct(): ...
def test_killswitch_stays_tripped_on_recovery_until_resume(): ...
```

- [ ] **Step 2: Run** `uv run pytest tests/risk -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: risk sizing, admission caps, kill-switch"`

---

### Task 10: Backtest engine

**Files:**
- Create: `src/tradingsystem/backtest/__init__.py`, `backtest/indicators.py`, `backtest/engine.py`
- Test: `tests/backtest/test_indicators.py`, `tests/backtest/test_engine.py`

**Interfaces:**
- Consumes: Bars; `run_signals` (Task 7); `day_charges`, `apply_slippage` (Task 2); everything from Task 9; `Universe`, `load_sectors` (Task 3).
- Produces:
  - `atr(bars: Bars, n: int) -> pd.Series` uses Wilder smoothing and is indexed like bars.
  - `BacktestConfig(start: date, end: date, initial_capital: float, slippage_mult: float = 1.0, enforce_kill_switch: bool = False)`. The kill-switch isn't enforced in backtests because nobody can resume it manually there. Trips are reported instead, and paper enforces it (Plan 3).
  - `run_backtest(signals: pd.DataFrame, bars: Bars, universe: Universe, sectors: dict[str,str], spec: StrategySpec, cfg: AppConfig, bt: BacktestConfig) -> BacktestResult`. It takes signals that were already computed, so Gate 2 reruns don't respawn the sandbox when only the cost or window changes.
  - `BacktestResult` has:
    - `equity: pd.Series` (by date, close-marked);
    - `trades: pd.DataFrame` with columns `symbol, entry_date, entry_price, exit_date, exit_price, qty, charges, pnl_net, exit_reason` (one of `signal, stop, max_holding, data_missing, end_of_test`);
    - `kill_switch_trips: int`;
    - `skipped: pd.DataFrame` with columns `date, symbol, reason`.

Daily event order for each trading day t in [start, end]:
1. **At the open of t**, fill queued orders: exits first, then entries.
   - A BUY fills at `apply_slippage(open, "BUY")`. It is skipped on `is_upper_circuit`. Qty is reduced to fit the cash that remains after charges and the 20% cap at the fill price, then the skip rules are re-applied.
   - A SELL fills at `apply_slippage(open, "SELL")`.
2. **Stops**, for each holding with a finite stop:
   - If `open ≤ stop`, exit at open × (1 − slip). This includes the entry day.
   - Otherwise, if `low ≤ stop`, exit at stop × (1 − slip).
3. **At the close of t**, mark to market and run `KillSwitch.update`.
   - Queue exits from `exit` signals, from holdings with days held ≥ `max_holding_days`, and from holdings missing a bar for `data_missing_exit_days`. Those last ones exit at their last close, not the next open.
   - Queue entries from `entry` signals for symbols that are members on t and not held. Size them with `close_t` as the entry estimate, `stop_t` and `atr_t`, then run them through `admit` (sorted by symbol).
4. **At end**, close every holding at the close of the last day with reason `end_of_test`.

Charges come from `day_charges` per day and are deducted from cash. `pnl_net` includes both legs' charges, allocated pro rata by turnover.

- [ ] **Step 1: Write the failing golden tests** on a hand-built 2-symbol, 10-day Bars frame with hand-set signals and slippage 0.1%. Each assertion is a value calculated in a comment in the test.

```python
def test_atr_wilder_matches_hand_calc(): ...
def test_signal_fills_next_open_with_slippage(): ...   # entry signal day 2 -> entry_price == open_day3 * 1.001
def test_charges_deducted_from_cash(): ...             # equity_end == capital + sum(pnl_net)
def test_gap_through_stop_fills_at_open(): ...         # Review Focus 2
def test_intraday_stop_fills_at_stop(): ...
def test_max_holding_exit(): ...
def test_insufficient_cash_reduces_qty(): ...          # Review Focus 3
def test_missing_bar_marks_last_close_then_exits(): ...  # Review Focus 1: exit_reason "data_missing" after 5 days
def test_non_member_entry_ignored(): ...
def test_upper_circuit_entry_skipped(): ...
def test_kill_switch_trip_counted_not_enforced(): ...
def test_deterministic(): assert run(...) == run(...)  # equity series equal
```

- [ ] **Step 2: Run** `uv run pytest tests/backtest -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Put the per-day loop in `engine.py`, with plain dicts for cash and holdings and no vectorisation. Correctness beats speed here; the bars are daily.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: event-driven daily backtest engine"`

---

### Task 11: Metrics and Gate 1

**Files:**
- Create: `src/tradingsystem/validate/__init__.py`, `validate/metrics.py`, `validate/gate1.py`
- Test: `tests/validate/test_metrics.py`, `tests/validate/test_gate1.py`

**Interfaces:**
- Consumes: `BacktestResult` (Task 10); `BarStore.benchmark` (Task 4); `GatesConfig`.
- Produces:
  - `sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float` (annualised √252; 0.0 when std is 0)
  - `max_drawdown(equity: pd.Series) -> float` (a positive fraction)
  - `profit_factor(pnl: pd.Series) -> float` (inf with no losses, 0 with no wins)
  - `CheckResult(name: str, value: float, threshold: float, passed: bool)` and `GateResult(gate: str, passed: bool, checks: list[CheckResult])`. Plan 4's scorecard renders these directly.
  - `gate1(result: BacktestResult, benchmark_close: pd.Series, cfg: GatesConfig) -> GateResult`. The checks are `sharpe`, `max_drawdown`, `trades`, `profit_factor` and `beats_benchmark` (strategy Sharpe > buy-and-hold Sharpe of the benchmark over the same dates).

- [ ] **Step 1: Write the failing tests.** Cover: Sharpe of a known series (hand calculation); max drawdown of [100, 120, 90, 130] == 0.25; profit factor of [100, −50, 50] == 3.0; Gate 1 fails on `trades` alone when there are 99 trades and everything else passes; it passes when all checks are just inside the thresholds.
- [ ] **Step 2: Run** `uv run pytest tests/validate/test_metrics.py tests/validate/test_gate1.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: performance metrics and Gate 1"`

---

### Task 12: Gate 2 — robustness

**Files:**
- Create: `src/tradingsystem/validate/gate2.py`, `validate/dsr.py`, `validate/grid.py`
- Test: `tests/validate/test_gate2.py`, `tests/validate/test_dsr.py`

**Interfaces:**
- Consumes: `run_signals`, `run_backtest`, metrics, `GateResult`.
- Produces:
  - `param_grid(spec: StrategySpec, n_per_param: int, max_combos: int, seed: int) -> list[Params]`. It takes linearly spaced values per param, rounded for ints and deduplicated. When the full product exceeds `max_combos` it takes a seeded random sample, and the defaults are always included.
  - `deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_var: float) -> float` follows Bailey and López de Prado and uses per-period (non-annualised) SR. SR0 = √sr_var · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))), where γ = 0.5772. With N = 1, SR0 is 0. DSR = Φ((sr−SR0)·√(n_obs−1) / √(1 − skew·sr + (kurt−1)/4·sr²)).
  - `gate2(strategy: LoadedStrategy, bars: Bars, ctx: BacktestContext, cfg: AppConfig, family_trials: int, family_sr_var: float | None) -> GateResult`. `BacktestContext` bundles the universe, sectors and benchmark. `family_trials` and `family_sr_var` come from the ideas library in Plan 3; for now they're passed by the caller. When `family_sr_var` is None or there are fewer than 2 trials, the variance of the per-period Sharpe across this strategy's param grid is used. The checks are:
    - `wf_oos_ratio`: rolling windows of fit 3y / test 1y, stepping 1y across the research period. Fit picks the grid point with the best in-sample Sharpe. The ratio is the Sharpe of concatenated OOS returns over the mean in-sample best Sharpe, and it must be ≥ 0.5.
    - `wf_losing_years`: OOS years with a net return below 0 must be ≤ 1.
    - `sensitivity`: one param at a time at default ± {10%, 20%} of range width, clamped. The fraction with net return > 0 must be ≥ 0.70, or 0.80 if `overfit_risk`.
    - `mc_dd95`: 1,000 seeded shuffles of the `pnl_net` order from the default backtest. The 95th-percentile drawdown of initial_capital + cumsum must be ≤ 0.30.
    - `cost_stress`: net return > 0 with `slippage_mult = 2.0`.
    - `dsr`: ≥ 0.95, or ≥ 0.99 if `overfit_risk`.
  - Signals are computed once per distinct param set through `run_signals` and reused across windows.

- [ ] **Step 1: Write the failing tests**

```python
def test_dsr_single_trial_equals_psr(): ...      # N=1 -> SR0 = 0
def test_dsr_drops_with_more_trials(): assert dsr(N=100) < dsr(N=10)
def test_grid_includes_defaults_and_respects_cap(): ...
def test_random_entry_fails_gate2():   # tests/fixtures/strategies/random_entry.py on make_bars(20 symbols, 2012-2022, seed=1)
    assert not gate2(...).passed
def test_planted_edge_passes_gate1_and_gate2():  # tests/fixtures/strategies/dip_buyer.py on make_bars(..., planted_edge=True)
    assert gate1(...).passed and gate2(...).passed
def test_overfit_strategy_uses_stricter_thresholds(): ...  # 6-param strategy -> sensitivity threshold 0.80
```

- [ ] **Step 2: Run** `uv run pytest tests/validate/test_gate2.py tests/validate/test_dsr.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `random_entry.py` derives a deterministic pseudo-random flag from a hash of the (date, symbol) string, with entry probability 5% and exit after 10 days. That keeps it free of look-ahead, so it reaches the gates.
- [ ] **Step 4: Run** the same command. Expected: PASS. If the planted-edge test doesn't pass Gate 1, strengthen the edge in `make_bars`, never the thresholds.
- [ ] **Step 5: Commit** `git commit -m "feat: Gate 2 walk-forward, sensitivity, Monte Carlo, cost stress, DSR"`

---

### Task 13: Gate 3 — locked holdout

**Files:**
- Create: `src/tradingsystem/validate/gate3.py`
- Test: `tests/validate/test_gate3.py`

**Interfaces:**
- Consumes: `BarStore.holdout_bars` (Task 4), `run_signals`, `run_backtest`, metrics.
- Produces:
  - `HoldoutLedger(path: Path = Path("state/holdout_runs.json"))` has `.used(hash: str) -> bool` and `.record(hash: str, result: GateResult) -> None`. It is append-only, and Plan 3 moves it to Postgres.
  - `HoldoutAlreadyUsed(Exception)`
  - `gate3(strategy, store, ctx, cfg, research_sharpe: float, research_passed_gates_1_2: bool, ledger: HoldoutLedger) -> GateResult`. It raises `ValueError` unless Gates 1–2 passed, and raises `HoldoutAlreadyUsed` if the hash is in the ledger. It computes signals on `holdout_bars` (research history included for warm-up) but backtests only from `holdout_start`. The checks are `sharpe_ratio_vs_research` (≥ 0.5 × research Sharpe) and `max_drawdown` (≤ the Gate 1 limit). The result is recorded **before** it's returned, so a crash after the run still counts as used.

- [ ] **Step 1: Write the failing tests.** Cover: a second call with the same hash raises `HoldoutAlreadyUsed`; it refuses when Gates 1–2 didn't pass; a one-character code change (new hash) is allowed again but appears as a separate ledger entry; trades never start before `holdout_start`.
- [ ] **Step 2: Run** `uv run pytest tests/validate/test_gate3.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: Gate 3 single-use holdout"`

---

### Task 14: Pipeline CLI and real-data acceptance

**Files:**
- Create: `src/tradingsystem/cli.py` (registered as `[project.scripts] ts = "tradingsystem.cli:main"`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `ts evaluate <strategy.py> [--holdout] [--family-trials N]` runs these steps and stops at the first failure:
  1. refresh data;
  2. `validate_bars` on research bars (stops with the issues on failure);
  3. `accept`;
  4. signals → `run_backtest` → `gate1` → `gate2` → optionally `gate3`.

  It prints one scorecard table (check, value, threshold, PASS/FAIL) per gate. Exit code is 0 when everything run passed, 1 on a gate failure, and 2 on a data or acceptance error.

- [ ] **Step 1: Write the failing tests.** Inject an in-memory `BarStore` backed by `make_bars` through a `--synthetic-seed` hidden option. The dip buyer on planted-edge data exits 0 and prints `Gate 2 PASS`; the random-entry strategy exits 1; the leaky strategy exits 2 and prints `lookahead`.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite with `uv run pytest -v`. Expected: all pass.
- [ ] **Step 5: Real-data run** (needs Tasks 2–4 owner actions and `.env`). Run `uv run ts evaluate strategies/reference/sma_crossover.py`. Record the scorecard in `docs/superpowers/plans/plan1-acceptance.md`. **Whatever it scores, don't change thresholds to make it pass.** If it fails Gate 1, that's a finding for the owner: the spec's success criterion assumes it passes.
- [ ] **Step 6: Commit** `git commit -m "feat: ts evaluate CLI; plan 1 acceptance run"`
