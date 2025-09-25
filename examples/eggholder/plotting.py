from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D


class EggholderPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Thin wrapper kept for compatibility; now only shows scatter in normalized space."""

    def __init__(self, update_every: int = 1, **scatter_kwargs: object):
        super().__init__(update_every=update_every, scatter_kwargs=scatter_kwargs)

__all__ = ["EggholderPlotter"]
