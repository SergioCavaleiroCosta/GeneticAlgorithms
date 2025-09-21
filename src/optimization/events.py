"""Typed event system for optimization lifecycle notifications."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, List, Protocol
from .types import ST, OT


# Base event type (non-generic for listener typing simplicity)
class OptimizationEventBase:
    """Marker base class for optimization events."""
    pass

# Event payloads
@dataclass(frozen=True)
class RunStarted(OptimizationEventBase, Generic[ST, OT]):
    elapsed: float
    evaluations: int
    initial_solution: ST
    initial_objective: OT


@dataclass(frozen=True)
class IterationStarted(OptimizationEventBase):
    iteration: int
    elapsed: float


@dataclass(frozen=True)
class BestImproved(OptimizationEventBase, Generic[OT]):
    iteration: int
    previous_best: OT
    new_best: OT


@dataclass(frozen=True)
class IterationCompleted(OptimizationEventBase, Generic[ST, OT]):
    iteration: int
    elapsed: float
    current_solution: ST
    current_objective: OT
    best_solution: ST
    best_objective: OT
    evaluations: int


@dataclass(frozen=True)
class RunCompleted(OptimizationEventBase, Generic[ST, OT]):
    iterations: int
    elapsed: float
    best_solution: ST
    best_objective: OT
    evaluations: int
    success: bool
    termination_reason: str


class OptimizationEventListener(Protocol):
    def on_event(self, event: OptimizationEventBase) -> None:
        """Handle an optimization event. Implementations should be fast and non-blocking."""
        ...


class EventDispatcher(Generic[ST, OT]):
    """Simple synchronous dispatcher for optimization events."""

    def __init__(self) -> None:
        self._listeners: List[OptimizationEventListener] = []

    def subscribe(self, listener: OptimizationEventListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: OptimizationEventListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def emit(self, event: OptimizationEventBase) -> None:
        # Synchronous dispatch
        for listener in self._listeners:
            listener.on_event(event)


__all__ = [
    "RunStarted",
    "IterationStarted",
    "BestImproved",
    "IterationCompleted",
    "RunCompleted",
    "OptimizationEventBase",
    "OptimizationEventListener",
    "EventDispatcher",
]
