from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
from .protocols import ParameterSpec, NormalizationStrategy

__all__ = ["ContinuousParameter"]


@dataclass(frozen=True, init=False)
class ContinuousParameter(ParameterSpec):
    """Concrete continuous parameter definition.

    Users supply name and real-domain bounds; a LinearNormalization is built
    unless an explicit normalizer is provided. Metadata (alias, description,
    meta) are optional and preserved for downstream reporting/plots.
    """
    _name: str
    _normalizer: NormalizationStrategy
    _alias: str = ""
    _description: str = ""
    _unit: str = ""
    _meta: Mapping[str, Any] | None = field(default=None)

    def __init__(
        self,
        name: str,
        normalizer: NormalizationStrategy,
        *,
        alias: str = "",
        description: str = "",
        unit: str = "",
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_normalizer", normalizer)
        object.__setattr__(self, "_alias", alias)
        object.__setattr__(self, "_description", description)
        object.__setattr__(self, "_unit", unit)
        object.__setattr__(self, "_meta", meta)

    # Protocol property implementations
    @property
    def name(self) -> str:
        return self._name

    @property
    def alias(self) -> str:
        return self._alias

    @property
    def description(self) -> str:
        return self._description

    @property
    def normalizer(self) -> NormalizationStrategy:
        return self._normalizer

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def meta(self) -> Mapping[str, Any] | None:  # type: ignore[override]
        return self._meta
