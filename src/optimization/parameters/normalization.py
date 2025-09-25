from __future__ import annotations
from dataclasses import dataclass
from .protocols import NormalizationStrategy

__all__ = ["LinearNormalization"]


@dataclass(frozen=True)
class LinearNormalization(NormalizationStrategy):  # structural + explicit for clarity
    """Affine mapping between a real interval [lo, hi] and normalized [0,1].

    Degenerate intervals (lo == hi) map every value to 0.0 and back to lo.
    """
    _lo: float
    _hi: float

    def __post_init__(self) -> None:
        orig_lo, orig_hi = float(self._lo), float(self._hi)
        lo, hi = (orig_lo, orig_hi) if orig_lo <= orig_hi else (orig_hi, orig_lo)
        object.__setattr__(self, "_lo", lo)
        object.__setattr__(self, "_hi", hi)

    @property
    def span(self) -> float:
        return self._hi - self._lo

    def to_norm(self, real: float) -> float:
        if self.span == 0.0:
            return 0.0
        return (float(real) - self._lo) / self.span

    def to_real(self, norm: float) -> float:
        return self._lo if self.span == 0.0 else self._lo + float(norm) * self.span

    @property
    def lo(self) -> float:
        return self._lo
    
    @property
    def hi(self) -> float:
        return self._hi
