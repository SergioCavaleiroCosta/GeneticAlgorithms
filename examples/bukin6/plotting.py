from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class Bukin6Plotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Bukin N.6 wrapper (2D)."""

    def __init__(self, update_every: int = 1, *, interactive: bool = False, **scatter_kwargs: object) -> None:
        super().__init__(update_every=update_every, scatter_kwargs=scatter_kwargs, interactive=interactive)

__all__ = ["Bukin6Plotter"]
