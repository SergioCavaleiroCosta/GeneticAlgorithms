from __future__ import annotations
from dataclasses import dataclass
from .protocols import NormalizationStrategy

__all__ = ["LinearNormalization"]


@dataclass(frozen=True)
class LinearNormalization(NormalizationStrategy):  # structural + explicit for clarity
    """Affine mapping between a real interval [lo, hi] and normalized [0,1].

    Degenerate intervals (lo == hi) map every value to 0.0 and back to lo.
    """
    lo: float
    hi: float

    def __post_init__(self) -> None:
        orig_lo, orig_hi = float(self.lo), float(self.hi)
        lo, hi = (orig_lo, orig_hi) if orig_lo <= orig_hi else (orig_hi, orig_lo)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def to_norm(self, real: float) -> float:
        if self.span == 0.0:
            return 0.0
        return (float(real) - self.lo) / self.span

    def to_real(self, norm: float) -> float:
        return self.lo if self.span == 0.0 else self.lo + float(norm) * self.span
