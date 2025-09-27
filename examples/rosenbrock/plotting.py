from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class RosenbrockPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Rosenbrock-specific thin wrapper.

    Uses dynamic best hyperplane for non-plotted dimensions so the slice
    follows current best candidate during optimization.
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

__all__ = ["RosenbrockPlotter"]
