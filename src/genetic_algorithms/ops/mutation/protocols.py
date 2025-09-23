from __future__ import annotations
from typing import Protocol, TypeVar


S = TypeVar("S")


class MutationStrategy(Protocol[S]):
    """Mutate an individual solution, returning a new solution of same type."""
    def mutate(self, x: S) -> S: ...


__all__ = ["MutationStrategy"]
