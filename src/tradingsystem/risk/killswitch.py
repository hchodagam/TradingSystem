"""Portfolio drawdown kill-switch: pause new entries at kill_switch_dd_pct below the peak."""
from __future__ import annotations

from tradingsystem.config import RiskConfig


class KillSwitch:
    """enforce=True (paper/live): stays tripped until resume(); resume() restarts the peak at current equity.
    enforce=False (backtest): never blocks entries; counts trips and re-arms at a new equity peak."""

    def __init__(self, cfg: RiskConfig, enforce: bool):
        self.cfg = cfg
        self.enforce = enforce
        self.tripped = False
        self.trips = 0
        self._peak: float | None = None
        self._last: float | None = None

    def update(self, equity: float) -> None:
        self._last = equity
        if self._peak is None or equity > self._peak:
            self._peak = equity
            if not self.enforce:
                self.tripped = False
        floor = self._peak * (1 - self.cfg.kill_switch_dd_pct) * (1 + 1e-12)
        if not self.tripped and equity <= floor:
            self.tripped = True
            self.trips += 1

    def resume(self) -> None:
        self.tripped = False
        self._peak = self._last

    def allows_entries(self) -> bool:
        return not (self.enforce and self.tripped)
