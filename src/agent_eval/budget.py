"""Soft budget. A case already running is allowed to finish; later cases are skipped."""

from __future__ import annotations

import threading

from agent_eval.config import BudgetConfig


class Budget:
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self.spent = 0.0
        self.started = 0
        self.judge_calls = 0
        self.estimated = False
        self.hit_usd = False
        self.hit_cases = False
        self.hit_judge_calls = False

    def try_reserve_case(self) -> bool:
        with self._lock:
            if self.config.max_cases is not None and self.started >= self.config.max_cases:
                self.hit_cases = True
                return False
            if self.config.max_usd is not None and self.spent >= self.config.max_usd:
                self.hit_usd = True
                return False
            if (
                self.config.max_judge_calls is not None
                and self.judge_calls >= self.config.max_judge_calls
            ):
                self.hit_judge_calls = True
                return False
            self.started += 1
            return True

    def charge(self, usd: float, *, estimated: bool, judge_calls: int = 0) -> None:
        with self._lock:
            self.spent += max(usd, 0.0)
            self.judge_calls += judge_calls
            self.estimated = self.estimated or estimated
            if self.config.max_usd is not None and self.spent >= self.config.max_usd:
                self.hit_usd = True
            if (
                self.config.max_judge_calls is not None
                and self.judge_calls >= self.config.max_judge_calls
            ):
                self.hit_judge_calls = True

    @property
    def exceeded(self) -> bool:
        return self.hit_usd or self.hit_cases or self.hit_judge_calls

    def skip_reason(self) -> str:
        if self.hit_usd:
            return f"budget max_usd {self.config.max_usd} reached (spent {self.spent:.4f})"
        if self.hit_cases:
            return f"budget max_cases {self.config.max_cases} reached"
        if self.hit_judge_calls:
            return f"budget max_judge_calls {self.config.max_judge_calls} reached"
        return "budget exhausted"
