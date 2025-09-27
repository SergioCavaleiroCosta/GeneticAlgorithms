from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class SpherePlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Sphere-specific thin wrapper.

    Uses same interface as base plotter. For high-dimensional sphere (e.g. 10D),
    a 2D contour of the first two parameters (or chosen pair) is shown while
    the remaining dimensions are fixed either by provided fixed_values or
    automatically filled using the current best solution (fallback to midpoint
    if enabled and best not yet available).
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
        super().__init__(
            update_every=update_every,
            scatter_kwargs=scatter_kwargs,
            param_pair=param_pair,
            param_names=param_names,
            fixed_values=fixed_values,
            midpoint_fallback=midpoint_fallback,
            interactive=interactive,
        )

__all__ = ["SpherePlotter"]
