"""Plotting utilities for Radiative Efficiency optimization.

This module provides visualization for the 5-dimensional radiative efficiency
optimization problem. Since we can't visualize 5D directly, we focus on:
- 2D slices showing selected parameter pairs
- Parameter distributions in the population
- Convergence history
"""
from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


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


__all__ = ["RadiativeEfficiencyPlotter"]
