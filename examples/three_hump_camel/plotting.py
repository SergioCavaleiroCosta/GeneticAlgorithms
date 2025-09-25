from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class ThreeHumpCamelPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Three-hump camel wrapper (2D)."""

    def __init__(self, update_every: int = 1, **scatter_kwargs: object) -> None:
        super().__init__(update_every=update_every, scatter_kwargs=scatter_kwargs)

__all__ = ["ThreeHumpCamelPlotter"]
