"""Validation gates. Thresholds come from config/gates.yaml (owner-edited only)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

    @classmethod
    def of(cls, gate: str, checks: list[CheckResult]) -> GateResult:
        return cls(gate, all(c.passed for c in checks), checks)
