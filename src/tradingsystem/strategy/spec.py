"""Strategy SPEC schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingsystem.strategy.params import Params

__all__ = ["Params", "ParamSpec", "StrategySpec"]

Number = int | float


class ParamSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    default: Number
    range: tuple[Number, Number]

    @model_validator(mode="after")
    def _check(self) -> ParamSpec:
        lo, hi = self.range
        if not lo < hi:
            raise ValueError(f"range low {lo} must be below high {hi}")
        if not lo <= self.default <= hi:
            raise ValueError(f"default {self.default} outside range [{lo}, {hi}]")
        return self

    @property
    def is_int(self) -> bool:
        return all(isinstance(v, int) and not isinstance(v, bool) for v in (self.default, *self.range))


class StrategySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    family: str = Field(min_length=1)
    hypothesis: str
    universe: Literal["nifty100"]
    timeframe: Literal["1D"]
    params: dict[str, ParamSpec]
    max_holding_days: int = Field(ge=1)

    @field_validator("hypothesis")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("hypothesis must be plain English, not blank")
        return v

    @property
    def overfit_risk(self) -> bool:
        return len(self.params) > 5

    def defaults(self) -> Params:
        return Params({k: v.default for k, v in self.params.items()})
