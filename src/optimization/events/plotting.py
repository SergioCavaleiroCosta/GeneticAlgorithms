"""Reusable plotting strategies for optimization runs (scatter + optional contour)."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, TYPE_CHECKING, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
        # Human-friendly label of fixed parameter values used for the contour
        self._fixed_info_label = None

    @staticmethod
    def _format_decimal_tick(value: float, _position: int) -> str:
        if not np.isfinite(value):
            return ""
        if np.isclose(value, 0.0):
            value = 0.0
        return f"{value:g}".replace(".", ",")

    def _apply_decimal_formatters(self) -> None:
        if self._fig is None or self._ax is None:
            return

        formatter = FuncFormatter(self._format_decimal_tick)
        self._ax.xaxis.set_major_formatter(formatter)
        self._ax.yaxis.set_major_formatter(formatter)

        for axis in self._fig.axes:
            if axis is self._ax:
                continue
            axis.xaxis.set_major_formatter(formatter)
            axis.yaxis.set_major_formatter(formatter)

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
        # Sempre recalcula o grid, ignorando o cache
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
                        real_val = float(p.normalizer.to_real(float(nb)))
                        best_real.append(real_val)
                    except Exception:
                        best_real.append(float(nb))
            except Exception:
                best_real = None
        fixed_parts: list[str] = []
        for idx, p in enumerate(params):
            if idx in (i, j):
                continue
            name = p.name
            if name in self._fixed_values:
                base[idx] = float(self._fixed_values[name])
                fixed_parts.append(f"{name}={base[idx]:.4g}")
            elif best_real is not None:
                base[idx] = best_real[idx]
                fixed_parts.append(f"{name}={base[idx]:.4g}")
            elif self._midpoint_fallback:
                base[idx] = 0.5 * (p.normalizer.lo + p.normalizer.hi)
                fixed_parts.append(f"{name}~mid={base[idx]:.4g}")
            else:
                base[idx] = p.normalizer.lo
                fixed_parts.append(f"{name}~lo={base[idx]:.4g}")
        # Store a compact label for figure title
        try:
            self._fixed_info_label = ", ".join(fixed_parts)
        except Exception:
            self._fixed_info_label = None

        # Evaluate grid
        prev_eval = engine.problem.evaluation_count
        shape = X.shape
        vals: list[float] = []
        for x_val, y_val in zip(X.ravel(), Y.ravel()):
            vec = base.copy()
            vec[i] = x_val
            vec[j] = y_val
            try:
                vals.append(float(engine.problem.evaluate(cast(Any, vec))))  # type: ignore[arg-type]
            except Exception:
                vals.append(np.nan)
        try:
            engine.problem.evaluation_count = prev_eval
        except Exception:
            pass
        Z = np.asarray(vals, dtype=np.float64).reshape(shape)
        return (X, Y, Z)

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
        self._apply_decimal_formatters()
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
            # Invalidate any existing grid cache at the start of a run
            self._grid_cache = None
            self._last_best_key = None
            self._setup_figure(bounds, engine)
            if engine.population.size > 0:
                pts = np.asarray(engine.population.candidates, dtype=np.float64)
                pts = self._denormalize(engine, pts)
                self._update_scatter(pts, engine)
        elif stage == Stage.ITERATION:
            if self._fig is None:
                return
            self._tick += 1  # Increment tick for FrameSaver compatibility
            
            # Recompute contour every iteration (always use best individual for fixed dims)
            try:
                if self._ax is not None and self._contour is not None:
                    self._grid_cache = None
                    # Track best key (normalized) for info/cache
                    if not self._fixed_values:
                        state = getattr(engine, "_state", None)
                        if state is not None:
                            self._last_best_key = tuple(float(x) for x in state.best_solution)  # type: ignore[attr-defined]
                        else:
                            self._last_best_key = None
                    bounds = self._compute_bounds(engine)
                    X, Y, Z = self._build_grid(engine, bounds)
                    # Clear the axis and redraw everything (contour + colorbar will be recreated)
                    self._ax.clear()
                    # Reset scatter reference since axis was cleared
                    self._scatter = None
                    # Reset axis properties
                    i, j = self._resolve_param_indices(engine)
                    params = list(engine.parameters)
                    self._ax.set_xlabel(params[i].name, fontsize=16)
                    self._ax.set_ylabel(params[j].name, fontsize=16)
                    self._ax.tick_params(axis='both', which='major', labelsize=12)
                    self._ax.set_xlim(*bounds[0])
                    self._ax.set_ylim(*bounds[1])
                    # Redraw contour
                    self._contour = self._ax.contourf(X, Y, Z, levels=self._contour_levels, cmap="viridis")
                    self._apply_decimal_formatters()
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
