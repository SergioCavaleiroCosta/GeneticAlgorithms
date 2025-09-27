from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class GriewankPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Griewank wrapper (slice first two dims when d>2)."""

    def __init__(self, update_every: int = 1, *, interactive: bool = False, **scatter_kwargs: object) -> None:
        super().__init__(update_every=update_every, scatter_kwargs=scatter_kwargs, interactive=interactive)

__all__ = ["GriewankPlotter"]
