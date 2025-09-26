"""Optimization event system and reusable stage strategies.

Public API mirrors previous single-module version; additional strategies live
in sibling modules (e.g. plotting utilities).
"""
from __future__ import annotations

from typing import Generic, Protocol, Dict, TYPE_CHECKING, Set
from enum import Enum
from ..types import ST, OT

if TYPE_CHECKING:
    from ..optimization_engine import OptimizationEngine


class Stage(Enum):
    RUN_START = "run_start"
    ITERATION = "iteration"
    RUN_END = "run_end"


STAGES: tuple[Stage, ...] = tuple(Stage)


class OptimizationStageStrategy(Protocol[ST, OT]):
    def execute(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None: ...


class EventDispatcher(Generic[ST, OT]):
    """Synchronous stage dispatcher for executing registered strategies per stage."""

    def __init__(self) -> None:
        self._channel_strategies: Dict[Stage, Set[OptimizationStageStrategy[ST, OT]]] = {
            stage: set() for stage in Stage
        }

    def add_strategy(self, stage: Stage, strategy: OptimizationStageStrategy[ST, OT]) -> None:
        self._channel_strategies[stage].add(strategy)

    def remove_strategy(self, stage: Stage, strategy: OptimizationStageStrategy[ST, OT]) -> None:
        self._channel_strategies[stage].discard(strategy)

    def emit(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None:
        for s in tuple(self._channel_strategies[stage]):
            s.execute(engine, stage)


# Re-export built-in strategy utilities
from .plotting import ContourPopulationPlotter2D  # noqa: E402
from .population_logging import PopulationLogger, PopulationLoggerConfig  # noqa: E402
from .frame_saver import FrameSaverStrategy, FrameSaverConfig  # noqa: E402

__all__ = [
    "Stage",
    "STAGES",
    "OptimizationStageStrategy",
    "EventDispatcher",
    "ContourPopulationPlotter2D",
    "PopulationLogger",
    "PopulationLoggerConfig",
    "FrameSaverStrategy",
    "FrameSaverConfig",
]
