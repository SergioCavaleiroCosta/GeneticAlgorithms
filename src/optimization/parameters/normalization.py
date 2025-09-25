from __future__ import annotations
from dataclasses import dataclass
from .protocols import NormalizationStrategy


__all__ = ["LinearNormalization"]


@dataclass(frozen=True, init=False)
class LinearNormalization(NormalizationStrategy):  # structural + explicit for clarity
    """Affine mapping between a real interval [lo, hi] and normalized [0,1].

    Degenerate intervals (lo == hi) map every value to 0.0 and back to lo.
    """
    _lo: float
    _hi: float

    def __init__(self, lo: float, hi: float) -> None:
        # Normalize ordering to ensure _lo <= _hi
        object.__setattr__(self, "_lo", min(lo, hi))
        object.__setattr__(self, "_hi", max(lo, hi))

    @property
    def span(self) -> float:
        return self._hi - self._lo

    def to_norm(self, real: float) -> float:
        if self.span == 0.0:
            return 0.0
        return (real - self._lo) / self.span

    def to_real(self, norm: float) -> float:
        return self._lo if self.span == 0.0 else self._lo + norm * self.span

    @property
    def lo(self) -> float:
        return self._lo
    
    @property
    def hi(self) -> float:
        return self._hi
