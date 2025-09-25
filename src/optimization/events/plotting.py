"""Reusable plotting strategies for optimization runs."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, cast, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

from ..types import NDArrayFloat, ST, OT
from . import OptimizationStageStrategy, Stage
if TYPE_CHECKING:  # import only for typing to avoid runtime circular dependency
    from ..optimization_engine import OptimizationEngine


class ContourPopulationPlotter2D(OptimizationStageStrategy[ST, OT]):
    """Minimal 2D population scatter plotter.

    Parameters
    ----------
    (Contour background removed for simplicity; only scatter is shown.)
    update_every : int
        Update scatter every N iterations.
    scatter_kwargs : Mapping[str, Any]
        Extra kwargs for matplotlib scatter.
    (Population points are shown in their native coordinate system.)
    """

    def __init__(
        self,
        *,
        update_every: int = 1,
        scatter_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Create plotter.

        Bounds are always derived on first RUN_START from
        `engine.problem.parameters` (expects at least 2 parameters exposing
        `normalizer.lo` / `normalizer.hi`).
        """
        self._scatter_kwargs = dict(scatter_kwargs or {"c": "yellow", "edgecolors": "k", "s": 30, "alpha": 0.8})
        self._update_every = max(1, update_every)
        self._tick = 0
        self._fig = None
        self._ax = None
        self._scatter = None

    # --- Bounds helper (stateless) ---------------------------------------
    def _compute_bounds(self, engine: "OptimizationEngine[ST, OT]") -> list[Tuple[float, float]]:
        params = getattr(getattr(engine, "problem", object()), "parameters", None)
        if not params:
            return [(0.0, 1.0), (0.0, 1.0)]
        lo_hi: list[Tuple[float, float]] = []
        for p in params[:2]:
            lo = float(getattr(p.normalizer, "lo", 0.0))
            hi = float(getattr(p.normalizer, "hi", 1.0))
            lo_f, hi_f = (lo, hi) if lo <= hi else (hi, lo)
            lo_hi.append((lo_f, hi_f))
        while len(lo_hi) < 2:
            lo_hi.append((0.0, 1.0))
        return lo_hi

    def _setup_figure(self, bounds: list[Tuple[float, float]]) -> None:
        plt_mod = cast(Any, plt)
        plt_mod.ion()
        fig, ax = plt_mod.subplots(figsize=(7, 6))
        ax.set_title("Population (scatter)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        self._fig = fig
        self._ax = ax
        plt_mod.show(block=False)

    def _update_scatter(self, points: NDArrayFloat) -> None:
        if self._fig is None or self._ax is None:
            return
        if self._scatter is None:
            self._scatter = self._ax.scatter(points[:, 0], points[:, 1], **self._scatter_kwargs)
        else:
            self._scatter.set_offsets(points)
        canvas = self._fig.canvas
        canvas.draw()
        canvas.flush_events()

    # --- Protocol entry point ---------------------------------------------
    def execute(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None:  # type: ignore[override]
        if stage == Stage.RUN_START:
            # derive bounds and create axes
            bounds = self._compute_bounds(engine)
            self._setup_figure(bounds)
            if engine.population.size > 0:
                pts = np.asarray(engine.population.candidates, dtype=np.float64)
                self._update_scatter(pts)
        elif stage == Stage.ITERATION:
            if self._fig is None:
                return
            self._tick += 1
            if (self._tick % self._update_every) != 0:
                return
            pts = np.asarray(engine.population.candidates, dtype=np.float64)
            if pts.size == 0:
                return
            self._update_scatter(pts)
        else:
            # Keep figure open on RUN_END
            pass


__all__ = ["ContourPopulationPlotter2D"]
