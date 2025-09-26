"""Generic frame saving strategy for population plotters.

This decouples image writing from specific plotter subclasses so that
individual examples do not need bespoke *Plotter classes with embedded
file I/O logic. It works with any object exposing a Matplotlib figure on
an attribute named ``_fig`` (as used by ``ContourPopulationPlotter2D``).

Usage (in an example run script)::

    plotter = AckleyPlotter(update_every=1, param_pair=(0,1))
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)

    frame_saver = FrameSaverStrategy(plotter, save_dir=figures_dir, prefix="frame")
    for st in (Stage.RUN_START, Stage.ITERATION):
        dispatcher.add_strategy(st, frame_saver)

By default frames are written at RUN_START and whenever the plotter decides
to refresh its scatter (detected via the plotter's internal ``_tick`` and
``_update_every`` attributes). You can also force a custom frequency with
``every`` and set ``respect_plotter_update=False`` to save each eligible
stage regardless of the plotter's own update cadence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import OptimizationStageStrategy, Stage


@dataclass
class FrameSaverConfig:
    prefix: str = "frame"
    fmt: str = "png"
    dpi: int = 120
    every: int = 1  # additional global frequency gating
    respect_plotter_update: bool = True  # only save when plotter updated


class FrameSaverStrategy(OptimizationStageStrategy):  # type: ignore[type-arg]
    def __init__(
        self,
        plotter: object,
        save_dir: Path,
        *,
        config: FrameSaverConfig | None = None,
        stages: Sequence[Stage] = (Stage.RUN_START, Stage.ITERATION),
    ) -> None:
        self._plotter = plotter
        self._save_dir = save_dir
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or FrameSaverConfig()
        self._stages = tuple(stages)
        self._frame = 0
        self._global_tick = 0

    def execute(self, engine, stage: Stage) -> None:  # type: ignore[override]
        if stage not in self._stages:
            return
        if stage == Stage.ITERATION:
            self._global_tick += 1
        if (self._global_tick % self._config.every) != 0 and stage != Stage.RUN_START:
            return

        # Access figure from plotter
        fig = getattr(self._plotter, "_fig", None)
        if fig is None:
            return

        if self._config.respect_plotter_update and stage == Stage.ITERATION:
            # Only save when the plotter itself just performed an update.
            tick = getattr(self._plotter, "_tick", None)
            upd = getattr(self._plotter, "_update_every", 1)
            if tick is None or (tick % max(1, upd)) != 0:
                return

        path = self._save_dir / f"{self._config.prefix}_{self._frame:04d}.{self._config.fmt}"
        try:
            fig.savefig(path, dpi=self._config.dpi)
            self._frame += 1
        except Exception:
            pass

__all__ = ["FrameSaverStrategy", "FrameSaverConfig"]