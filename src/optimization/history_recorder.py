"""History recording strategies driven by optimization events."""
from __future__ import annotations
from typing import Generic, List, Protocol, cast
import csv
from .types import OT
from .events import (
    OptimizationEventBase,
    OptimizationEventListener,
    RunStarted,
    IterationCompleted,
    RunCompleted,
)


class HistoryRecorder(OptimizationEventListener, Protocol[OT]):
    """Protocol for pluggable history recorders.

    Recorders consume events to store the convergence history in memory,
    on disk, or nowhere at all.
    """

    def on_event(self, event: OptimizationEventBase) -> None:  # pragma: no cover - protocol
        ...

    def get_history(self) -> List[OT]:  # pragma: no cover - protocol
        ...


class InMemoryHistoryRecorder(Generic[OT]):
    """Stores best objective values in memory.

    Appends the initial objective on RunStarted and best objective on each
    IterationCompleted event.
    """

    def __init__(self) -> None:
        self._history: List[OT] = []

    def on_event(self, event: OptimizationEventBase) -> None:
        if isinstance(event, RunStarted):
            rs = cast(RunStarted[object, OT], event)
            self._history.append(rs.initial_objective)
        elif isinstance(event, IterationCompleted):
            ic = cast(IterationCompleted[object, OT], event)
            self._history.append(ic.best_objective)
        elif isinstance(event, RunCompleted):
            # nothing to do; kept for completeness
            pass

    def get_history(self) -> List[OT]:
        return list(self._history)


class NoOpHistoryRecorder:
    """Does not record anything; always returns an empty history."""

    def on_event(self, event: OptimizationEventBase) -> None:
        return

    def get_history(self) -> List[OT]:  # type: ignore[override]
        return []


class CSVHistoryRecorder(Generic[OT]):
    """Writes convergence info to a CSV file incrementally.

    Columns: iteration, elapsed, current_objective, best_objective, evaluations.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._header_written = False

    def on_event(self, event: OptimizationEventBase) -> None:
        if isinstance(event, IterationCompleted):
            ic = cast(IterationCompleted[object, OT], event)
            self._write_row([
                str(ic.iteration),
                f"{ic.elapsed:.6f}",
                str(ic.current_objective),
                str(ic.best_objective),
                str(ic.evaluations),
            ])
        elif isinstance(event, RunStarted):
            # Write header lazily on first event
            if not self._header_written:
                self._write_row(["iteration", "elapsed", "current_objective", "best_objective", "evaluations"], header=True)
        elif isinstance(event, RunCompleted):
            # nothing additional needed; rows already written
            pass

    def _write_row(self, row: List[str], header: bool = False) -> None:
        mode = "a" if self._header_written or not header else "a"
        with open(self._path, mode, newline="") as f:
            writer = csv.writer(f)
            if header and not self._header_written:
                writer.writerow(row)
                self._header_written = True
            elif not header:
                if not self._header_written:
                    writer.writerow(["iteration", "elapsed", "current_objective", "best_objective", "evaluations"])
                    self._header_written = True
                writer.writerow(row)

    def get_history(self) -> List[OT]:  # type: ignore[override]
        # CSV recorder doesn't keep an in-memory history
        return []


__all__ = [
    "HistoryRecorder",
    "InMemoryHistoryRecorder",
    "NoOpHistoryRecorder",
    "CSVHistoryRecorder",
]
