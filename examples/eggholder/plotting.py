from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class EggholderPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Example-specific thin wrapper.

    Accepts the same parameter-selection arguments as base plotter:
      - param_pair: tuple of two indices
      - param_names: tuple of two parameter names
      - fixed_values: mapping name->real value for non-plotted params
      - midpoint_fallback: if True use midpoint for unspecified fixed values
    """

    def __init__(
        self,
        update_every: int = 1,
        *,
        param_pair: tuple[int, int] | None = None,
        param_names: tuple[str, str] | None = None,
        fixed_values: dict[str, float] | None = None,
        midpoint_fallback: bool = True,
        **scatter_kwargs: object,
    ) -> None:
        super().__init__(
            update_every=update_every,
            scatter_kwargs=scatter_kwargs,
            param_pair=param_pair,
            param_names=param_names,
            fixed_values=fixed_values,
            midpoint_fallback=midpoint_fallback,
        )

__all__ = ["EggholderPlotter"]
