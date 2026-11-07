# Strategy Engine — Design Spec (Step 1)

**Date:** 2026-11-06
**Status:** Draft for review
**Scope:** Step 1 of 4 — core engine + Indian equities (daily bars) through paper trading.

---

## 1. Purpose and success criteria

Build an AI-assisted research and paper-trading engine for swing/positional trading of Indian equities. LLM agents propose and code strategy ideas; deterministic code validates them against strict gates; the owner approves what goes to paper trading. Real-money trading is a later, separate project, enabled only for strategies that pass every gate including paper trading.

**v1 is successful when:**
- A strategy brief entered in the dashboard flows through research → spec → code → review → Gates 1–3 → approval → paper trading with no manual steps other than the two approvals (shortlist, approve-for-paper).
- The reference strategy (50/200 SMA crossover) runs through the full pipeline on real data and gets a gate verdict. It is *expected* to fail Gate 1: the 2026-11-07 run scored Sharpe 0.75, drawdown 25.8%, and did not beat Nifty 100 buy-and-hold (see `docs/superpowers/plans/plan1-acceptance.md`). Its 60-day paper replay, run without gate approval for this check, matches a backtest of the same window.
- A strategy exploiting a planted edge in synthetic data passes Gates 1–2. This is the proof that a good strategy can get through the gates.
- A random-entry strategy is rejected by Gates 1–2; a deliberately look-ahead-biased strategy is rejected by the look-ahead check.
- Cost model matches real Dhan contract notes to the rupee.

### Roadmap (each step gets its own spec → plan → build)

| Step | Scope |
|---|---|
| **1 (this spec)** | Core engine + Indian equities, daily bars, research + paper trading |
| 2 | US equities, daily bars (new data/cost/broker adapters) |
| 3 | Intraday support (minute bars, sessions, square-off) — crypto first |
| 4 | Indian F&O intraday (option chains, expiries, margin) |
| Live | Dhan live order placement for strategies that are *ready for live* |

### Decisions taken

| Topic | Decision |
|---|---|
| Goal | Research + paper trading now; live later after proper testing |
| Market | Indian equities, Nifty 100 universe, daily bars, swing/positional |
| AI role | Agents propose and code; owner approves; gates/risk are deterministic code |
| Interface | Web dashboard over a Python core |
| LLMs | One model for all agents in dev/test (recommended: GLM 5.3 Flash, confirmed by bake-off); per-agent models possible later |
| Data / broker | Dhan (historical data now; live orders later) |
| Capital | ₹1–2 lakh |
| Architecture | Fixed pipeline with AI steps (modular monolith), not a free-form multi-agent framework |

---

## 2. Architecture

```
                ┌────────────── Web dashboard (React) ──────────────┐
                │ ideas · backtests · approvals · paper P&L · costs │
                └───────────────────────┬───────────────────────────┘
                                        │ REST
┌───────────────────────────── FastAPI app ─────────────────────────────┐
│  Pipeline orchestrator (job queue; one strategy = one "case" moving   │
│  through stages; each stage reads/writes the DB, never calls another) │
│                                                                       │
│  LLM stages (llm/ adapter, cost-capped)                               │
│    1 Research   2 Ideate   3 Code   4 Critic                          │
│                                                                       │
│  Deterministic stages                                                 │
│    5 Backtest   6 Validate (gates)   7 Risk check   8 Paper trader    │
└───────┬───────────────────────────┬───────────────────────────┬───────┘
   Market data                 Database                     Broker
   (Dhan adapter →             (Postgres: cases,            (PaperBroker now;
    Parquet cache)              runs, trades, costs)         DhanBroker later)
```

### Units

| Unit | Responsibility | Depends on |
|---|---|---|
| `data/` | Fetch daily OHLCV from Dhan; corporate-action adjustment; Parquet cache; point-in-time Nifty 100 membership; NSE trading calendar; data validation | Dhan API |
| `llm/` | Single `complete(agent, messages, schema)` interface; OpenAI-compatible client with configurable endpoint/key/model; token + ₹ cost logging; caps | `agents.yaml` |
| `agents/` | Research, Ideate, Code, Critic — prompt template + output schema each, versioned | `llm/` |
| `strategy/` | Strategy format (SPEC + `signals()`), schema validation, sandboxed execution, look-ahead check, content-hash versioning | — |
| `backtest/` | Event-driven daily engine: next-open fills, Indian cost model, whole-share sizing via `risk/`, stops, max holding period | `data/`, `risk/`, `strategy/` |
| `validate/` | Gate checks as pure functions; thresholds from `gates.yaml` | `backtest/` |
| `risk/` | Position sizing, position/sector caps, skip rules, kill-switch; same code in backtest, paper and live | `risk.yaml` |
| `paper/` | Daily EOD cycle: fill queued orders, update positions, generate next-day orders, drift tracking | all of the above, `broker/` |
| `broker/` | `Broker` interface; `PaperBroker` now, `DhanBroker` in live phase | — |
| `orchestrator/` | Case state machine, Postgres-backed job queue, APScheduler daily jobs | DB |
| `api/` + `web/` | FastAPI REST endpoints; React dashboard | orchestrator, DB |

**Invariant:** an LLM may create or modify a *strategy*; it can never modify gate thresholds, risk limits, config, or anything that places orders. Agents have no tools — the orchestrator passes text in and stores structured output.

### Case lifecycle

`Hypothesis → Shortlisted → Specified → Coding → Reviewed → Gate1 → Gate2 → Gate3 → AwaitingApproval → Paper → ReadyForLive | Rejected | Failed`

Every transition is logged with timestamp, actor (agent/system/owner) and reason.

---

## 3. Strategy format

Each strategy is one Python file:

```python
SPEC = {
  "name": "rsi_pullback_in_uptrend",
  "family": "mean_reversion_in_trend",
  "hypothesis": "Strong stocks that dip briefly tend to resume trend",
  "universe": "nifty100",
  "timeframe": "1D",
  "params": {
    "rsi_len":   {"default": 3,   "range": [2, 5]},
    "rsi_entry": {"default": 15,  "range": [5, 25]},
    "trend_ma":  {"default": 200, "range": [100, 250]},
  },
  "max_holding_days": 10,
}

def signals(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    """bars: daily OHLCV for all symbols up to and including *today* only.
    returns per symbol: entry (bool), exit (bool), stop (float | NaN)."""
```

**Rules**
- **Signals only.** Strategies never size positions, compute costs, or place orders.
- **No look-ahead.** Engine passes data up to the current bar only. The look-ahead check re-runs `signals()` incrementally (day by day) on a sample window and requires identical output to the batch run.
- **Parameter budget.** Each param must declare a range; > 5 params is flagged as overfit risk (Critic note + stricter Gate 2 penalty).
- **Hypothesis required** in plain English; Critic checks the code matches it.
- **Sandbox.** Separate low-privilege subprocess; no network, no filesystem writes; import allowlist (`pandas`, `numpy`, `math`, approved TA lib); CPU-time and memory limits.
- **Versioning.** Stored by content hash; every backtest, gate result and paper trade references the exact hash.

**Code acceptance pipeline:** schema check → lint → sandbox smoke test → look-ahead check → Critic review. On failure the error goes back to the Code agent; max 3 retries, then case → `Failed`.

---

## 4. Validation gates

### Data split
- **Research period:** ~2012 → ~2022. Agents and owner iterate freely here.
- **Locked holdout:** last ~2–3 years. Never shown to agents. Run **once** per strategy after Gates 1–2. Failure = reject; no re-tuning.
- Exact boundary dates set in `gates.yaml`.

### Cost model (applied in backtest, paper, live)
Delivery brokerage ₹0 (Dhan), STT 0.1% buy and sell, exchange transaction charges, SEBI fee, stamp duty (buy), GST on charges, DP charge per sell scrip per day, slippage 0.1% on next-day open. All rates in config; verified against real contract notes.

### Gate 1 — Backtest (research period, net of costs)
- Sharpe ≥ 1.0
- Max drawdown ≤ 20%
- Trades ≥ 100
- Profit factor ≥ 1.3
- Beats buy-and-hold Nifty 100 benchmark on risk-adjusted return

### Gate 2 — Robustness
- **Walk-forward:** rolling 3y fit / 1y test; OOS Sharpe ≥ 50% of in-sample; ≤ 1 losing test year.
- **Parameter sensitivity:** each param perturbed ±20% across its range; ≥ 70% of neighbouring settings profitable.
- **Monte Carlo:** 1,000 trade-order reshuffles; 95th-percentile drawdown ≤ 30%.
- **Cost stress:** 2× slippage still profitable.
- **Multiple-testing penalty:** each `family` tracks trial count; required Sharpe is adjusted via the deflated Sharpe ratio.

### Gate 3 — Holdout
Single run on locked holdout: Sharpe ≥ 50% of research-period Sharpe; max drawdown within Gate 1 limit.

### Owner approval
Dashboard scorecard, equity curve, trades, Critic notes → Approve / Reject (note stored in ideas library).

### Gate 4 — Paper
≥ 3 months and ≥ 30 trades; realised performance within the Monte Carlo 95% band; zero risk-rule breaches. Pass → `ReadyForLive`. Going live is the owner's decision in the live-phase project.

**Config ownership:** `gates.yaml` and `risk.yaml` are owner-edited only. Agents may read values (to aim proposals) but never write.

---

## 5. Risk and paper trading

### Sizing (capital ₹1–2L)
- Risk per trade 1% of equity. Shares = floor(1% × equity ÷ (entry − stop)). No stop given → 2×ATR(14) assumed for sizing.
- Max 20% of equity per position; max 6 open positions; max 2 per sector.
- Skip if < 1 share, position value < ₹5,000, or symbol at upper circuit at open.
- **Kill-switch:** portfolio drawdown 10% from peak → pause new entries, alert; exits continue; owner resumes manually.

### Multiple strategies
- Each paper strategy has its own notional account (default ₹1L, configurable) for isolated evaluation.
- A combined portfolio simulation runs all paper strategies on one shared pool with global caps (mirrors live).
- Recommended: ≤ 3 concurrent paper strategies.

### Daily schedule (IST, NSE trading days)
1. **18:30** — Fetch EOD bars from Dhan; validate (missing symbols, gaps, abnormal moves, corporate actions). Failure → skip day, red banner, no trades.
2. **18:30 (same job)** — Fill orders queued yesterday at today's actual open + slippage + costs; update positions, stops, P&L; process exits.
3. **18:45** — For each paper strategy: `signals()` on data up to today → risk sizing → queue orders for next open → show "tomorrow's planned trades".

### Drift tracking
Paper equity is compared to the backtest's Monte Carlo band; leaving the 95% band flags the strategy for review.

### Replay mode
Simulates the daily cycle over past N trading days for testing (no waiting for real days).

---

## 6. Agents

| Agent | Input | Output (JSON schema) |
|---|---|---|
| **Research** | Owner brief; deterministic market summary of research period (trend, volatility, breadth per year); ideas library | 3–5 hypotheses: rationale, market behaviour exploited, family, expected weaknesses |
| **Ideate** | One shortlisted hypothesis; gate thresholds (read-only) | `SPEC` + plain-English rules |
| **Code** | `SPEC` + format rules (+ previous error on retry) | Strategy file |
| **Critic** | `SPEC`, code (without Code agent reasoning), Gate 1–2 results | Code-matches-hypothesis verdict; bias/overfit risks; "why this might be luck"; advisory recommendation |

### Model configuration (`agents.yaml`)
- `mode: dev` (default) — all agents use one model. Recommended starting model: **GLM 5.3 Flash**.
- `mode: prod` — per-agent model mapping (enables mixed providers later).
- Candidates: MiniMax M2.7, DeepSeek Flash, GLM 5.3 Flash, Qwen 3.8 Flash — all via OpenAI-compatible endpoints.
- **Model bake-off (first agent task):** run 5 fixed briefs through each candidate; score first-pass rate on schema + sandbox + look-ahead checks, retries needed, and cost. Pick the dev model from results.
- **Same-model Critic mitigation:** separate adversarial prompt, no access to Code agent's reasoning, low temperature. Critic is advisory; gates decide.

### Owner touchpoints
1. Write brief → Research proposes → **owner shortlists**.
2. Ideate → Code → Critic → Gates 1–3 run automatically.
3. **Owner approves for paper** or rejects with a note.

### Ideas library
Every hypothesis with family, status, gate results, rejection reasons and owner notes. Fed to Research to avoid repeats; supplies trial counts for the deflated-Sharpe penalty.

### Cost control
Every call logged (agent, case, tokens, ₹). Per-run and monthly caps in config; hitting a cap pauses the pipeline with a banner — never silently skips steps.

### Out of v1
News/sentiment agents, web browsing, agent-to-agent debate. Can be added later as extra Research inputs.

---

## 7. Dashboard (v1 screens)

1. **Pipeline board** — kanban of cases by lifecycle state.
2. **Strategy page** — hypothesis, SPEC, read-only code + hash, gate scorecard (actual numbers vs thresholds), equity vs benchmark, drawdown, parameter-sensitivity heatmap, Monte Carlo band, trade list, Critic notes, **Approve / Reject (note)**.
3. **Research** — brief input, proposed hypotheses with shortlist checkboxes, searchable ideas library.
4. **Paper trading** — per strategy and combined: positions, tomorrow's planned trades, P&L within expected band, costs paid, drift alerts.
5. **System** — data health by day, job log, LLM spend vs caps, kill-switch status, read-only view of active gate/risk config.

Single-user login.

---

## 8. Tech stack and deployment

- **Backend:** Python 3.12, FastAPI, Pydantic (all schemas: SPEC, agent outputs, configs).
- **Storage:** Postgres (cases, runs, trades, costs, ideas, job queue); Parquet + DuckDB/pandas for market data.
- **Jobs:** Postgres-backed queue; APScheduler for daily runs. No Redis.
- **LLM client:** OpenAI-compatible client with configurable base URL/key/model.
- **Sandbox:** subprocess, resource limits, import allowlist, low-privilege user.
- **Frontend:** React + Vite + TypeScript; TradingView Lightweight Charts for price/equity; simple chart lib for scorecards.
- **Deployment:** Docker Compose. Develop locally (`C:\Projects\TradingSystem`); run paper trading on a small VPS so the 18:30 job never misses.
- **Secrets:** Dhan and LLM keys in `.env`, never committed; Dhan token refresh handled by the data adapter.

---

## 9. Error handling

| Failure | Behaviour |
|---|---|
| Dhan fetch fails / data invalid | Retry with backoff; then skip the day, banner, no trades on bad data |
| LLM returns invalid JSON / schema | Retry with validation error (max 3), then case → `Failed` with reason |
| Generated code fails sandbox / look-ahead | Back to Code agent with error (max 3), then `Failed` |
| Cost cap reached | Pipeline paused, banner; resumes when owner raises cap or month rolls over |
| Kill-switch triggered | New entries paused, alert; exits continue; manual resume |
| Any stage exception | Logged against the case; case stays in its state with error visible; never silent |

---

## 10. Testing

- **Cost model:** fixtures from real Dhan contract notes; exact-to-the-rupee assertions.
- **Backtest engine:** golden tests on small hand-calculated datasets (fills, sizing, rounding, stops, max holding, corporate actions).
- **Look-ahead check:** a deliberately leaky strategy must be caught; a clean one must pass.
- **Gates:** random-entry strategy must fail Gates 1–2; synthetic data with a planted edge must pass.
- **Risk:** sizing, caps, skip rules, kill-switch unit tests.
- **Sandbox:** network access, file writes and blocked imports must all fail.
- **Agents:** recorded responses as fixtures, including malformed outputs for retry/failure paths; optional live smoke test per model.
- **End to end:** the 50/200 SMA crossover runs through the full pipeline on real data and is rejected at Gate 1 (an expected failure). A 60-day replay of paper trading, run without gate approval, matches a backtest of the same window. A planted-edge strategy passes Gates 1–2 on synthetic data.
- Development is test-first for all deterministic units.

---

## 11. Out of scope for step 1

Live order placement; US equities, crypto, F&O; intraday; news/sentiment; mobile app; multi-user; portfolio optimisation beyond the fixed risk rules.

## 12. Open items to settle during planning

- Exact research/holdout boundary dates (depends on how far back Dhan history reliably goes).
- Source for point-in-time Nifty 100 membership history (NSE index change notices vs. a maintained list).
- Choice of TA library for the sandbox allowlist.
- VPS provider.
