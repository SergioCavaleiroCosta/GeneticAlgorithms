"""Typed event system for optimization lifecycle notifications."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, List, Protocol, Dict, Literal
from .types import ST, OT
# Stage channels for event routing
Channel = Literal["run_start", "iteration", "run_end"]


# Base event type (non-generic for listener typing simplicity)
class OptimizationEventBase:
    """Marker base class for optimization events."""
    pass

# Stage contexts (preferred names)
@dataclass(frozen=True)
class RunStartContext(OptimizationEventBase, Generic[ST, OT]):
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
class IterationContext(OptimizationEventBase, Generic[ST, OT]):
    iteration: int
    elapsed: float
    current_solution: ST
    current_objective: OT
    best_solution: ST
    best_objective: OT
    evaluations: int


@dataclass(frozen=True)
class RunEndContext(OptimizationEventBase, Generic[ST, OT]):
    iterations: int
    elapsed: float
    best_solution: ST
    best_objective: OT
    evaluations: int
    success: bool
    termination_reason: str

# Backward-compatible aliases
RunStarted = RunStartContext
IterationCompleted = IterationContext
RunCompleted = RunEndContext


class OptimizationEventListener(Protocol):
    def on_event(self, event: OptimizationEventBase) -> None:
        """Handle an optimization event. Implementations should be fast and non-blocking."""
        ...


class EventDispatcher(Generic[ST, OT]):
    """Simple synchronous dispatcher for optimization events with optional stage channels.

    Listeners can subscribe globally (all events) or to a specific stage channel
    like "run", "iteration", or "completion". Emitting to a channel notifies
    both the listeners of that channel and any global listeners.
    """

    def __init__(self) -> None:
        self._listeners: List[OptimizationEventListener] = []
        self._channel_listeners: Dict[Channel, List[OptimizationEventListener]] = {
            "run_start": [],
            "iteration": [],
            "run_end": [],
        }

    # Global subscriptions
    def subscribe(self, listener: OptimizationEventListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: OptimizationEventListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def emit(self, event: OptimizationEventBase) -> None:
        # Synchronous dispatch to global listeners
        for listener in list(self._listeners):
            listener.on_event(event)

    # Channel-specific subscriptions
    def subscribe_to(self, channel: Channel, listener: OptimizationEventListener) -> None:
        listeners = self._channel_listeners[channel]
        if listener not in listeners:
            listeners.append(listener)

    def unsubscribe_from(self, channel: Channel, listener: OptimizationEventListener) -> None:
        listeners = self._channel_listeners[channel]
        try:
            listeners.remove(listener)
        except ValueError:
            pass

    def emit_to(self, channel: Channel, event: OptimizationEventBase) -> None:
        # Notify channel listeners then global listeners
        for listener in list(self._channel_listeners[channel]):
            listener.on_event(event)
        for listener in list(self._listeners):
            listener.on_event(event)


__all__ = [
    "Channel",
    "RunStartContext",
    "RunEndContext",
    "IterationContext",
    # Aliases for backward compatibility
    "RunStarted",
    "IterationStarted",
    "BestImproved",
    "IterationCompleted",
    "RunCompleted",
    "OptimizationEventBase",
    "OptimizationEventListener",
    "EventDispatcher",
]
