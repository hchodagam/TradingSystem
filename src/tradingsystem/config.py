"""Owner-edited YAML config loaded into frozen Pydantic models."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Rounding(_Frozen):
    stt: int
    stamp: int
    other: int


class CostRates(_Frozen):
    brokerage_delivery: float
    stt_pct: float
    exchange_txn_pct: float
    sebi_pct: float
    stamp_buy_pct: float
    gst_pct: float
    dp_per_sell_scrip_day: float
    slippage_pct: float
    rounding: Rounding


class RiskConfig(_Frozen):
    risk_per_trade_pct: float
    atr_len: int
    atr_stop_mult: float
    max_position_pct: float
    max_open_positions: int
    max_per_sector: int
    min_position_value: float
    upper_circuit_gap_pct: float
    kill_switch_dd_pct: float
    data_missing_exit_days: int
    initial_capital: float


class Gate1Config(_Frozen):
    min_sharpe: float
    max_drawdown: float
    min_trades: int
    min_profit_factor: float


class Gate2Config(_Frozen):
    wf_fit_years: int
    wf_test_years: int
    wf_min_oos_ratio: float
    wf_max_losing_years: int
    sens_step_pct: float
    sens_min_profitable: float
    mc_runs: int
    mc_seed: int
    mc_dd_pct95_max: float
    cost_stress_slippage_mult: float
    dsr_min: float
    grid_values_per_param: int
    grid_max_combos: int
    overfit_param_count: int
    overfit_dsr_min: float
    overfit_sens_min_profitable: float
    lookahead_param_window: int
    sandbox_chunk_size: int


class Gate3Config(_Frozen):
    min_sharpe_ratio_vs_research: float


class GatesConfig(_Frozen):
    research_start: date
    research_end: date
    holdout_start: date
    risk_free_rate: float
    gate1: Gate1Config
    gate2: Gate2Config
    gate3: Gate3Config

    @model_validator(mode="after")
    def _ordered(self) -> GatesConfig:
        if not (self.research_start < self.research_end < self.holdout_start):
            raise ValueError("need research_start < research_end < holdout_start")
        return self


class DataConfig(_Frozen):
    cache_dir: Path
    reference_dir: Path
    dhan_base_url: str
    scrip_master_url: str
    max_days_per_request: int
    retries: int
    backoff_seconds: float
    prices_adjusted_by_dhan: bool
    benchmark_symbol: str
    abnormal_move_pct: float


class AppConfig(_Frozen):
    costs: CostRates
    risk: RiskConfig
    gates: GatesConfig
    data: DataConfig


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Path = Path("config")) -> AppConfig:
    d = Path(config_dir)
    return AppConfig(
        costs=_read(d / "costs.yaml"),
        risk=_read(d / "risk.yaml"),
        gates=_read(d / "gates.yaml"),
        data=_read(d / "data.yaml"),
    )
