"""Reusable plotting strategies for optimization runs (scatter + optional contour)."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, TYPE_CHECKING, cast
import numpy as np
import matplotlib.pyplot as plt

from ..types import NDArrayFloat, ST, OT
from . import OptimizationStageStrategy, Stage

if TYPE_CHECKING:  # avoid runtime circular dependency
    from ..optimization_engine import OptimizationEngine


class ContourPopulationPlotter2D(OptimizationStageStrategy[ST, OT]):
    """2D population plotter with one-time contour background.

    Features
    --------
    - Derives real bounds from problem parameter normalizers (first two only)
    - Optionally draws objective contour at RUN_START (cached)
    - Auto-denormalizes population (assumes normalized [0,1]^d internal repr.)
    - Updates scatter every `update_every` iterations

    Parameters
    ----------
    update_every : int
        Update scatter every N iterations (>=1).
    scatter_kwargs : Mapping[str, Any] | None
        Extra matplotlib scatter kwargs.
    (Objective contour computed once directly via problem.evaluate; evaluation
    counter is restored so precomputation is "free".)
    grid_size : int
        Resolution of contour grid per axis (grid_size^2 evaluations if
        fallback path used). Keep modest to avoid overhead.
    contour_levels : int
        Number of contour levels for contourf.
    """

    def __init__(
        self,
        *,
        update_every: int = 1,
        scatter_kwargs: Optional[Mapping[str, Any]] = None,
        grid_size: int = 120,
        contour_levels: int = 40,
    ) -> None:
        self._scatter_kwargs = dict(scatter_kwargs or {"c": "yellow", "edgecolors": "k", "s": 30, "alpha": 0.8})
        self._update_every = max(1, update_every)
        self._grid_size = max(10, grid_size)
        self._contour_levels = max(2, contour_levels)

        # Runtime state
        self._tick = 0
        self._grid_cache: Optional[tuple[Any, Any, Any]] = None  # (X, Y, Z)
        self._fig = None
        self._ax = None
        self._contour = None
        self._scatter = None

    # ------------------------------------------------------------------
    def _compute_bounds(self, engine: "OptimizationEngine[ST, OT]") -> list[Tuple[float, float]]:
        params = engine.parameters
        return [(p.normalizer.lo, p.normalizer.hi) for p in params[:2]]

    def _build_grid(self, engine: "OptimizationEngine[ST, OT]", bounds: list[Tuple[float, float]]) -> tuple[Any, Any, Any]:
        if self._grid_cache is not None:
            return self._grid_cache
        (x_min, x_max), (y_min, y_max) = bounds
        xs = np.linspace(x_min, x_max, self._grid_size, dtype=np.float64)
        ys = np.linspace(y_min, y_max, self._grid_size, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        
        # Use problem.evaluate for each grid point; restore evaluation counter
        prev_eval = engine.problem.evaluation_count
        shape = X.shape
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        vals: list[float] = []
        for row in pts:
            try:
                vals.append(float(engine.problem.evaluate(row))) 
            except Exception:
                vals.append(np.nan)
        try:
            engine.problem.evaluation_count = prev_eval
        except Exception:
            pass
        Z = np.asarray(vals, dtype=np.float64).reshape(shape)

        self._grid_cache = (X, Y, Z)
        return self._grid_cache

    def _setup_figure(self, bounds: list[Tuple[float, float]], engine: "OptimizationEngine[ST, OT]") -> None:
        plt_mod = cast(Any, plt)
        plt_mod.ion()
        fig, ax = plt_mod.subplots(figsize=(7, 6))
        ax.set_title("Population with contour")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        try:
            X, Y, Z = self._build_grid(engine, bounds)
            self._contour = ax.contourf(X, Y, Z, levels=self._contour_levels, cmap="viridis")
            fig.colorbar(self._contour, ax=ax, shrink=0.85)
        except Exception:
            self._contour = None
        self._fig = fig
        self._ax = ax
        plt_mod.show(block=False)

    def _denormalize(self, engine: "OptimizationEngine[ST, OT]", points: NDArrayFloat) -> NDArrayFloat:
        # Direct attribute access; assume engine.problem.parameters exists.
        params = list(engine.parameters)
        if not params or points.ndim != 2:
            return points
        real = points.copy()
        dim = real.shape[1]
        for j, p in enumerate(params[:dim]):
            norm = p.normalizer  # assume exists
            to_real = norm.to_real  # assume exists & callable
            converted_vals: list[float] = []
            for v in real[:, j]:  # type: ignore[assignment]
                try:
                    converted_vals.append(float(to_real(float(v))))
                except Exception:
                    converted_vals.append(float(v))
            real[:, j] = np.array(converted_vals, dtype=real.dtype)
        return real

    def _update_scatter(self, pts: NDArrayFloat) -> None:
        if self._fig is None or self._ax is None:
            return
        if self._scatter is None:
            self._scatter = self._ax.scatter(pts[:, 0], pts[:, 1], **self._scatter_kwargs)
        else:
            self._scatter.set_offsets(pts[:, :2])
        canvas = self._fig.canvas
        canvas.draw()
        canvas.flush_events()

    # ------------------------------------------------------------------
    def execute(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None:
        if stage == Stage.RUN_START:
            bounds = self._compute_bounds(engine)
            self._setup_figure(bounds, engine)
            if engine.population.size > 0:
                pts = np.asarray(engine.population.candidates, dtype=np.float64)
                pts = self._denormalize(engine, pts)
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
            pts = self._denormalize(engine, pts)
            self._update_scatter(pts)
        else:
            # RUN_END: leave figure open
            pass


__all__ = ["ContourPopulationPlotter2D"]
