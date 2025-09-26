from __future__ import annotations

from optimization.events import ContourPopulationPlotter2D, Stage
from pathlib import Path
from typing import Optional


class EggholderPlotter(ContourPopulationPlotter2D):  # type: ignore[type-arg]
    """Eggholder population + contour plotter with optional frame saving.

    If `save_dir` is provided, a PNG file will be written each time the scatter
    updates (at RUN_START and then every `update_every` iterations) using the
    pattern: {prefix}_{frame:04d}.png.
    """

    def __init__(
        self,
        update_every: int = 1,
        *,
        param_pair: tuple[int, int] | None = None,
        param_names: tuple[str, str] | None = None,
        fixed_values: dict[str, float] | None = None,
        midpoint_fallback: bool = True,
        save_dir: Optional[Path] = None,
        image_prefix: str = "frame",
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
        self._save_dir = save_dir
        self._image_prefix = image_prefix
        self._frame = 0
        if self._save_dir is not None:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_save(self, stage: Stage) -> None:
        if self._save_dir is None:
            return
        # Access base class internal figure attribute (_fig)
        fig = getattr(self, "_fig", None)
        # Save on RUN_START or when tick just triggered an update
        if fig is not None and (stage == Stage.RUN_START or (getattr(self, "_tick", 0) % getattr(self, "_update_every", 1) == 0)):
            path = self._save_dir / f"{self._image_prefix}_{self._frame:04d}.png"
            try:
                fig.savefig(path, dpi=120)
                self._frame += 1
            except Exception:
                pass

    def execute(self, engine, stage: Stage) -> None:  # type: ignore[override]
        super().execute(engine, stage)
        if stage in (Stage.RUN_START, Stage.ITERATION):
            self._maybe_save(stage)

__all__ = ["EggholderPlotter"]
