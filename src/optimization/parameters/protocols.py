from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Mapping


class NormalizationStrategy(Protocol):
    """Protocol for mapping between real domain values and normalized [0,1]."""
    def to_norm(self, real: float) -> float: ...
    def to_real(self, norm: float) -> float: ...


@runtime_checkable
class ParameterSpec(Protocol):
    """Describes a single optimizable parameter with metadata and normalization."""
    @property
    def name(self) -> str: ...

    @property
    def alias(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def normalizer(self) -> NormalizationStrategy: ...

    @property
    def unit(self) -> str: ...

    @property
    def meta(self) -> Mapping[str, Any] | None: ...


__all__ = [
    "NormalizationStrategy",
    "ParameterSpec",
]
