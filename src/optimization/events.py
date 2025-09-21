"""Typed event system for optimization lifecycle notifications."""
from __future__ import annotations
from typing import Generic, Protocol, Dict, Literal, TYPE_CHECKING, Set
from .types import ST, OT

if TYPE_CHECKING:
    from .optimization_engine import OptimizationEngine

# Stage channels for event routing
Channel = Literal["run_start", "iteration", "run_end"]



# Unified strategy protocol (single entry point; access via engine, stage-aware)
class OptimizationStageStrategy(Protocol[ST, OT]):
    def execute(self, engine: "OptimizationEngine[ST, OT]", stage: Channel) -> None: ...


class EventDispatcher(Generic[ST, OT]):
    """Synchronous stage dispatcher for executing registered strategies per stage."""

    def __init__(self) -> None:
        # Per-stage strategies (uniqueness guaranteed by set semantics)
        self._channel_strategies: Dict[Channel, Set[OptimizationStageStrategy[ST, OT]]] = {
            "run_start": set(),
            "iteration": set(),
            "run_end": set(),
        }
    # Strategy registration
    def add_strategy(self, channel: Channel, strategy: OptimizationStageStrategy[ST, OT]) -> None:
        self._channel_strategies[channel].add(strategy)

    def remove_strategy(self, channel: Channel, strategy: OptimizationStageStrategy[ST, OT]) -> None:
        self._channel_strategies[channel].discard(strategy)

    # Strategy execution helpers (also emits context events for compatibility)
    def emit_run_start(self, engine: "OptimizationEngine[ST, OT]") -> None:
        for s in tuple(self._channel_strategies["run_start"]):
            s.execute(engine, "run_start")

    def emit_iteration(self, engine: "OptimizationEngine[ST, OT]") -> None:
        for s in tuple(self._channel_strategies["iteration"]):
            s.execute(engine, "iteration")

    def emit_run_end(self, engine: "OptimizationEngine[ST, OT]") -> None:
        for s in tuple(self._channel_strategies["run_end"]):
            s.execute(engine, "run_end")


__all__ = [
    "Channel",
    "OptimizationStageStrategy",
    "EventDispatcher",
]
