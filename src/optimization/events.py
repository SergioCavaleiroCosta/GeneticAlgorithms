"""Typed event system for optimization lifecycle notifications."""
from __future__ import annotations
from typing import Generic, Protocol, Dict, TYPE_CHECKING, Set
from enum import Enum
from .types import ST, OT

if TYPE_CHECKING:
    from .optimization_engine import OptimizationEngine

# Stage channels as an enum for clarity and type-safety
class Stage(Enum):
    RUN_START = "run_start"
    ITERATION = "iteration"
    RUN_END = "run_end"

# Ordered iterable of all stages (auto-derived from the Enum)
STAGES: tuple[Stage, ...] = tuple(Stage)


# Unified strategy protocol (single entry point; access via engine, stage-aware)
class OptimizationStageStrategy(Protocol[ST, OT]):
    def execute(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None: ...


class EventDispatcher(Generic[ST, OT]):
    """Synchronous stage dispatcher for executing registered strategies per stage."""

    def __init__(self) -> None:
        # Per-stage strategies (uniqueness guaranteed by set semantics)
        self._channel_strategies: Dict[Stage, Set[OptimizationStageStrategy[ST, OT]]] = {
            stage: set() for stage in Stage
        }
    # Strategy registration
    def add_strategy(self, stage: Stage, strategy: OptimizationStageStrategy[ST, OT]) -> None:
        self._channel_strategies[stage].add(strategy)

    def remove_strategy(self, stage: Stage, strategy: OptimizationStageStrategy[ST, OT]) -> None:
        self._channel_strategies[stage].discard(strategy)

    # Strategy execution (single entry point)
    def emit(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None:
        for s in tuple(self._channel_strategies[stage]):
            s.execute(engine, stage)


__all__ = [
    "Stage",
    "STAGES",
    "OptimizationStageStrategy",
    "EventDispatcher",
]
