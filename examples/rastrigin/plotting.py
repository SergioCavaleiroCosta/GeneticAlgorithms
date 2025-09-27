from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class RastriginPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Rastrigin-specific thin wrapper.

    For d>2, shows a 2D contour over a selected parameter pair while fixing
    remaining dimensions to the current best solution's real values (dynamic
    hyperplane)."""

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

__all__ = ["RastriginPlotter"]
