from __future__ import annotations

from typing import Callable, Optional, List, Tuple, Any, Mapping, cast
import numpy as np
import matplotlib.pyplot as plt

from optimization.events import OptimizationStageStrategy, Stage
from optimization.types import NDArrayFloat


def eggholder_function(xy: NDArrayFloat) -> float:
    x = float(xy[0])
    y = float(xy[1])
    term1 = -(y + 47.0) * np.sin(np.sqrt(abs(x / 2.0 + (y + 47.0))))
    term2 = -x * np.sin(np.sqrt(abs(x - (y + 47.0))))
    return float(term1 + term2)


class ContourPopulationPlotter(OptimizationStageStrategy[NDArrayFloat, float]):
    """Plot population positions over a contour of a 2D function in real-time.

    Parameters
    ----------
    fn : Callable[[NDArrayFloat], float]
        Objective function taking a vector [x, y] and returning a float.
        It should NOT mutate evaluation counters; intended for visualization only.
    bounds : list of (min, max)
        Bounds for x and y; required to set the plotting area.
    grid_size : int
        Number of points per dimension for the contour grid.
    scatter_kwargs : dict
        Optional matplotlib kwargs for the scatter plot.
    """

    def __init__(
        self,
        fn: Callable[[NDArrayFloat], float],
        bounds: List[Tuple[float, float]],
        grid_size: int = 300,
        update_every: int = 1,
        scatter_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if len(bounds) != 2:
            raise ValueError("ContourPopulationPlotter requires 2D bounds [(x_min,x_max),(y_min,y_max)]")
        self._fn = fn
        self._bounds = bounds
        self._grid_size = grid_size
        self._scatter_kwargs = scatter_kwargs or {"c": "yellow", "edgecolors": "k", "s": 30, "alpha": 0.8}
        self._update_every = max(1, update_every)
        self._tick = 0

        # Matplotlib artifacts
        self._fig = None
        self._ax = None
        self._scatter = None

    def _setup_figure(self) -> None:
        (x_min, x_max), (y_min, y_max) = self._bounds
        xs = np.linspace(x_min, x_max, self._grid_size, dtype=np.float64)
        ys = np.linspace(y_min, y_max, self._grid_size, dtype=np.float64)
        xg, yg = np.meshgrid(xs, ys)
        # Fast vectorized eggholder for plotting (no evaluation counter changes)
        z = -(yg + 47.0) * np.sin(np.sqrt(np.abs(xg / 2.0 + (yg + 47.0))))
        z = z + (-xg * np.sin(np.sqrt(np.abs(xg - (yg + 47.0)))))

        plt_mod = cast(Any, plt)
        plt_mod.ion()
        fig, ax = plt_mod.subplots(figsize=(7, 6))
        cs = ax.contourf(xg, yg, z, levels=40, cmap="viridis")
        fig_any: Any = fig
        ax_any: Any = ax
        fig_any.colorbar(cs, ax=ax_any, shrink=0.9)
        ax_any.set_title("Population over Eggholder Contour")
        ax_any.set_xlabel("x")
        ax_any.set_ylabel("y")
        ax_any.set_xlim(x_min, x_max)
        ax_any.set_ylim(y_min, y_max)

        self._fig = fig_any
        self._ax = ax_any
        # Non-blocking show; we'll manually draw/flush in updates
        plt_mod.show(block=False)

    def _update_scatter(self, points: NDArrayFloat) -> None:
        assert self._ax is not None
        if self._scatter is None:
            kwargs_any: Any = self._scatter_kwargs
            self._scatter = self._ax.scatter(points[:, 0], points[:, 1], **kwargs_any)
        else:
            self._scatter.set_offsets(points)
        # Draw and flush events without starting a nested mainloop
        assert self._fig is not None
        canvas = self._fig.canvas
        canvas.draw()
        canvas.flush_events()

    def execute(self, engine: Any, stage: Stage) -> None:
        # Only handle RUN_START and ITERATION
        if stage == Stage.RUN_START:
            self._setup_figure()
            # Draw initial population if present
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
            # On RUN_END, keep the figure open for inspection
            pass


__all__ = ["ContourPopulationPlotter", "eggholder_function"]