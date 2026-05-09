"""Plotting utilities for Radiative Efficiency optimization.

This module provides visualization for the 5-dimensional radiative efficiency
optimization problem. Since we can't visualize 5D directly, we focus on:
- 2D slices showing selected parameter pairs
- Parameter distributions in the population
- Convergence history
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from matplotlib.ticker import FuncFormatter

from optimization.events import ContourPopulationPlotter2D, Stage

if TYPE_CHECKING:
    from optimization.optimization_engine import OptimizationEngine
    from optimization.types import ST, OT


class RadiativeEfficiencyPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Radiative Efficiency specific plotter.
    
    For visualization of the 5D problem, shows 2D contour plots over selected
    parameter pairs (e.g., u_avg vs phi) while fixing remaining dimensions to
    the current best solution's values.
    
    Args:
        update_every: Plot update frequency (iterations)
        param_pair: Tuple of parameter indices to plot (default: (1, 0) for u_avg vs phi)
        param_names: Tuple of parameter names for labels
        fixed_values: Fixed values for non-plotted dimensions
        midpoint_fallback: Use midpoint if fixed_values not provided
        interactive: Enable interactive plotting
        **scatter_kwargs: Additional scatter plot arguments
    """
    
    def __init__(
        self,
        update_every: int = 1,
        *,
        param_pair: tuple[int, int] | None = None,
        param_names: tuple[str, str] | None = None,
        fixed_values: dict[str, float] | None = None,
        midpoint_fallback: bool = True,
        interactive: bool = False,
        **scatter_kwargs: object,
    ) -> None:
        # Default to u_avg (x1) vs phi (x0) - velocity vs equivalence ratio
        if param_pair is None:
            param_pair = (1, 0)
        
        if param_names is None:
            param_names = ("u_avg", "phi")
        
        super().__init__(
            update_every=update_every,
            scatter_kwargs=scatter_kwargs,
            param_pair=param_pair,
            param_names=param_names,
            fixed_values=fixed_values,
            midpoint_fallback=midpoint_fallback,
            interactive=interactive,
        )

    @staticmethod
    def _format_decimal_tick(value: float, _position: int) -> str:
        if not np.isfinite(value):
            return ""
        if np.isclose(value, 0.0):
            value = 0.0
        return f"{value:g}".replace(".", ",")

    def _apply_decimal_formatters(self) -> None:
        formatter = FuncFormatter(self._format_decimal_tick)

        if self._ax is not None:
            self._ax.xaxis.set_major_formatter(formatter)
            self._ax.yaxis.set_major_formatter(formatter)

        if self._fig is not None:
            for axis in self._fig.axes:
                if axis is self._ax:
                    continue
                axis.xaxis.set_major_formatter(formatter)
                axis.yaxis.set_major_formatter(formatter)

    def _apply_colorbar_label(self) -> None:
        if self._fig is None:
            return

        for axis in self._fig.axes:
            if axis is self._ax:
                continue
            axis.set_ylabel(r"$\eta_{\mathrm{rad}}$", fontsize=16)
            axis.tick_params(axis="y", which="major", labelsize=12)
    
    def _build_grid(self, engine: "OptimizationEngine[ST, OT]", bounds: list[tuple[float, float]]) -> tuple[object, object, object]:
        """Build grid with absolute value for efficiency (positive values)."""
        X, Y, Z = super()._build_grid(engine, bounds)
        # Take absolute value since efficiency should be positive
        Z_abs = np.abs(Z)
        return (X, Y, Z_abs)
    
    def _setup_figure(self, bounds: list[tuple[float, float]], engine: "OptimizationEngine[ST, OT]") -> None:
        """Setup figure with custom axis labels using LaTeX."""
        super()._setup_figure(bounds, engine)
        
        # Override axis labels with LaTeX formatting
        self._update_axis_labels(engine)
        self._apply_decimal_formatters()
        self._apply_colorbar_label()
    
    def _update_axis_labels(self, engine: "OptimizationEngine[ST, OT]") -> None:
        """Update axis labels with LaTeX formatting."""
        if self._ax is not None:
            i, j = self._resolve_param_indices(engine)
            params = list(engine.parameters)
            
            # Custom labels for phi and u_avg
            xlabel = params[i].name
            ylabel = params[j].name
            
            if xlabel == "phi":
                xlabel = r"$\phi$"
            elif xlabel == "u_avg":
                xlabel = r"$\overline{u}$ [m.s$^{-1}$]"
            
            if ylabel == "phi":
                ylabel = r"$\phi$"
            elif ylabel == "u_avg":
                ylabel = r"$\overline{u}$ [m.s$^{-1}$]"
            
            self._ax.set_xlabel(xlabel, fontsize=16)
            self._ax.set_ylabel(ylabel, fontsize=16)
    
    def execute(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None:
        """Execute with custom labels applied after parent execution."""
        super().execute(engine, stage)
        
        if stage in (Stage.RUN_START, Stage.ITERATION) and self._ax is not None:
            self._update_axis_labels(engine)
            self._apply_decimal_formatters()
            self._apply_colorbar_label()


__all__ = ["RadiativeEfficiencyPlotter"]
