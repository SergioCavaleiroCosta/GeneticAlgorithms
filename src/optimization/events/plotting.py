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
    param_pair : tuple[int, int] | None
        Explicit zero-based indices of the two parameters to visualize. Mutually
        exclusive with `param_names`. Defaults to the first two parameters when
        omitted.
    param_names : tuple[str, str] | None
        Names of the two parameters to visualize (if indices not supplied).
    fixed_values : Mapping[str, float] | None
        Real-domain fixed values for parameters not in the chosen pair. Keys are
        parameter names. Unspecified names use midpoint if `midpoint_fallback` is True
        else the lower bound.
    midpoint_fallback : bool
        When True (default) fill unspecified fixed parameters with the midpoint of
        their real interval; otherwise use the lower bound.
    """

    def __init__(
        self,
        *,
        update_every: int = 1,
        scatter_kwargs: Optional[Mapping[str, Any]] = None,
        grid_size: int = 120,
        contour_levels: int = 40,
        param_pair: Optional[Tuple[int, int]] = None,
        param_names: Optional[Tuple[str, str]] = None,
        fixed_values: Optional[Mapping[str, float]] = None,
        midpoint_fallback: bool = True,
        interactive: bool = False,
    ) -> None:
        self._scatter_kwargs = dict(
            scatter_kwargs or {"c": "yellow", "edgecolors": "k", "s": 30, "alpha": 0.8}
        )
        # Ensure scatter always draws above filled contours even after contour redraws
        if "zorder" not in self._scatter_kwargs:
            self._scatter_kwargs["zorder"] = 10
        self._update_every = max(1, update_every)
        self._grid_size = max(10, grid_size)
        self._contour_levels = max(2, contour_levels)
        # Parameter selection configuration
        self._param_pair = param_pair
        self._param_names = param_names
        self._fixed_values = dict(fixed_values or {})
        self._midpoint_fallback = midpoint_fallback

        # Runtime state
        self._tick = 0
        self._grid_cache: Optional[tuple[Any, Any, Any]] = None  # (X, Y, Z)
        self._fig = None
        self._ax = None
        self._contour = None
        self._scatter = None
        # Track last best solution (normalized) snapshot for cache invalidation
        self._last_best_key: Optional[tuple[float, ...]] = None
        # Headless/interactive mode toggle
        self._interactive = bool(interactive)

    # ------------------------------------------------------------------
    def _resolve_param_indices(self, engine: "OptimizationEngine[ST, OT]") -> Tuple[int, int]:
        params = list(engine.parameters)
        n = len(params)
        if n < 2:
            raise ValueError("Need at least two parameters for 2D plotting")
        if self._param_pair is not None:
            i, j = self._param_pair
            if not (0 <= i < n and 0 <= j < n and i != j):
                raise ValueError("param_pair indices out of range or equal")
            return i, j
        if self._param_names is not None:
            names = [p.name for p in params]
            try:
                i = names.index(self._param_names[0])
                j = names.index(self._param_names[1])
            except ValueError as e:
                raise ValueError("param_names not found in parameters") from e
            if i == j:
                raise ValueError("param_names must refer to two distinct parameters")
            return i, j
        return 0, 1

    def _compute_bounds(self, engine: "OptimizationEngine[ST, OT]") -> list[Tuple[float, float]]:
        params = list(engine.parameters)
        i, j = self._resolve_param_indices(engine)
        return [
            (params[i].normalizer.lo, params[i].normalizer.hi),
            (params[j].normalizer.lo, params[j].normalizer.hi),
        ]

    def _build_grid(self, engine: "OptimizationEngine[ST, OT]", bounds: list[Tuple[float, float]]) -> tuple[Any, Any, Any]:
        if self._grid_cache is not None:
            return self._grid_cache
        (x_min, x_max), (y_min, y_max) = bounds
        xs = np.linspace(x_min, x_max, self._grid_size, dtype=np.float64)
        ys = np.linspace(y_min, y_max, self._grid_size, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)

        # Prepare base vector for other dimensions
        params = list(engine.parameters)
        i, j = self._resolve_param_indices(engine)
        d = len(params)
        base = np.zeros(d, dtype=np.float64)
        # Determine template real values for non-plotted dimensions
        state = getattr(engine, "_state", None)
        use_best = (not self._fixed_values) and state is not None and hasattr(state, "best_solution")
        best_real: list[float] | None = None
        if use_best:
            try:
                # best_solution is normalized; convert to real
                norm_best = np.asarray(state.best_solution, dtype=np.float64)  # type: ignore[attr-defined]
                best_real = []
                for idx2, p in enumerate(params):
                    nb = norm_best[idx2] if idx2 < norm_best.shape[0] else 0.0
                    try:
                        best_real.append(float(p.normalizer.to_real(float(nb))))
                    except Exception:
                        best_real.append(float(nb))
            except Exception:
                best_real = None
        for idx, p in enumerate(params):
            if idx in (i, j):
                continue
            name = p.name
            if name in self._fixed_values:
                base[idx] = float(self._fixed_values[name])
            elif best_real is not None:
                base[idx] = best_real[idx]
            elif self._midpoint_fallback:
                base[idx] = 0.5 * (p.normalizer.lo + p.normalizer.hi)
            else:
                base[idx] = p.normalizer.lo

        # Evaluate grid
        prev_eval = engine.problem.evaluation_count
        shape = X.shape
        vals: list[float] = []
        for x_val, y_val in zip(X.ravel(), Y.ravel()):
            vec = base.copy()
            vec[i] = x_val
            vec[j] = y_val
            try:
                vals.append(float(engine.problem.evaluate(cast(Any, vec))))
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
        if self._interactive:
            plt_mod.ion()
        fig, ax = plt_mod.subplots(figsize=(7, 6))
        i, j = self._resolve_param_indices(engine)
        params = list(engine.parameters)
        ax.set_xlabel(params[i].name, fontsize=16)
        ax.set_ylabel(params[j].name, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
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
        if self._interactive:
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

    def _update_scatter(self, pts: NDArrayFloat, engine: "OptimizationEngine[ST, OT]") -> None:
        if self._fig is None or self._ax is None:
            return
        i, j = self._resolve_param_indices(engine)
        proj = pts[:, [i, j]]
        if self._scatter is None:
            self._scatter = self._ax.scatter(proj[:, 0], proj[:, 1], **self._scatter_kwargs)
            # Reassert z-order explicitly (some backends may ignore initial kwargs)
            try:
                self._scatter.set_zorder(self._scatter_kwargs.get("zorder", 10))
            except Exception:
                pass
        else:
            self._scatter.set_offsets(proj)
            try:
                self._scatter.set_zorder(self._scatter_kwargs.get("zorder", 10))
            except Exception:
                pass
        if self._interactive:
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
                self._update_scatter(pts, engine)
        elif stage == Stage.ITERATION:
            if self._fig is None:
                return
            self._tick += 1
            if (self._tick % self._update_every) != 0:
                return
            # Invalidate grid cache if best solution changed (when using dynamic best fill)
            if not self._fixed_values:
                try:
                    state = getattr(engine, "_state", None)
                    if state is not None:
                        norm_best = tuple(float(x) for x in state.best_solution)  # type: ignore[attr-defined]
                    else:
                        norm_best = ()
                    if norm_best != self._last_best_key:
                        self._grid_cache = None
                        self._last_best_key = norm_best
                        # Replot contour if present
                        if self._ax is not None and self._contour is not None:
                            try:
                                bounds = self._compute_bounds(engine)
                                X, Y, Z = self._build_grid(engine, bounds)
                                for c in self._contour.collections:
                                    c.remove()
                                self._contour = self._ax.contourf(X, Y, Z, levels=self._contour_levels, cmap="viridis")
                                # After replotting contour, push scatter (if any) back to front
                                if self._scatter is not None:
                                    try:
                                        base_z = 1
                                        if self._contour is not None and self._contour.collections:
                                            base_z = max(col.get_zorder() for col in self._contour.collections)
                                        self._scatter.set_zorder(base_z + 1)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass
            pts = np.asarray(engine.population.candidates, dtype=np.float64)
            if pts.size == 0:
                return
            pts = self._denormalize(engine, pts)
            self._update_scatter(pts, engine)
        else:
            # RUN_END: leave figure open
            pass


__all__ = ["ContourPopulationPlotter2D"]
