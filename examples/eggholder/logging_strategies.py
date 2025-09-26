"""Example convenience import of reusable PopulationLogger.

This file now simply re-exports the generic implementation from
`optimization.events.population_logging` so existing example code keeps
working without modification.
"""
from __future__ import annotations

from pathlib import Path
from optimization.events import PopulationLogger as _BasePopulationLogger, PopulationLoggerConfig
from optimization.types import NDArrayFloat


class PopulationLogger(_BasePopulationLogger[NDArrayFloat, float]):  # noqa: D401
    def __init__(self, out_dir: Path) -> None:
        super().__init__(out_dir, config=PopulationLoggerConfig())

__all__ = ["PopulationLogger", "PopulationLoggerConfig"]
