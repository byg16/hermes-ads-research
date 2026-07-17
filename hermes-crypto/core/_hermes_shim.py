"""Minimal fallback implementation of the bits of the `hermes-agent`
framework's API this project relies on (`Agent`, `Loop`).

`hermes-agent` (https://github.com/nousresearch/hermes-agent) is an early
-stage framework; its public package interface can shift. If it's not
installed/importable, `core/orchestrator.py` imports this shim instead so
the project always runs end-to-end. If you have the real package
installed, this file is unused — `orchestrator.py` prefers the real import.

The shim implements the same conceptual primitives we need:
  - Agent: a named callable step with typed run().
  - Loop: chains agents and supports a `run_forever`/`run_once` interface
    with a feedback hook invoked after every cycle.
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

from core.logging_config import get_logger

logger = get_logger("hermes_shim")


class Agent:
    """Base class mirroring hermes_agent.Agent's minimal contract."""

    name: str = "agent"

    def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


class Loop:
    """Mirrors hermes_agent.Loop: runs a pipeline function repeatedly,
    optionally calling an `on_cycle_complete` feedback hook after each run,
    which is exactly the (5) feedback-loop requirement from the brief."""

    def __init__(
        self,
        cycle_fn: Callable[[], Any],
        on_cycle_complete: Optional[Callable[[Any], None]] = None,
        interval_seconds: int = 60,
    ):
        self.cycle_fn = cycle_fn
        self.on_cycle_complete = on_cycle_complete
        self.interval_seconds = interval_seconds
        self._stop = False

    def run_once(self) -> Any:
        result = self.cycle_fn()
        if self.on_cycle_complete:
            try:
                self.on_cycle_complete(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("on_cycle_complete_failed", error=str(exc))
        return result

    def run_forever(self) -> None:
        logger.info("loop_started", interval_seconds=self.interval_seconds)
        while not self._stop:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 — never let one bad cycle kill the loop
                logger.error("cycle_failed", error=str(exc))
            time.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._stop = True
