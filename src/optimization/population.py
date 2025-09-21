"""Generic population container for optimization candidates and objectives."""
from __future__ import annotations
from typing import Generic, List
from .types import ST, OT


class Population(Generic[ST, OT]):
    """Container managing one or many candidates and tracking the best.

    Works for both single-candidate (e.g., Newton, coordinate descent) and
    population-based (e.g., GA, PSO) algorithms.
    """

    def __init__(self, candidates: List[ST], objectives: List[OT]) -> None:
        if len(candidates) != len(objectives):
            raise ValueError("candidates and objectives must have the same length")
        if not candidates:
            raise ValueError("population must contain at least one candidate")
        self._candidates: List[ST] = list(candidates)
        self._objectives: List[OT] = list(objectives)
        self._best_index: int = self._compute_best_index()

    @classmethod
    def from_single(cls, solution: ST, objective: OT) -> Population[ST, OT]:
        return cls([solution], [objective])

    def _compute_best_index(self) -> int:
        best_i = 0
        best_val = self._objectives[0]
        for i in range(1, len(self._objectives)):
            if self._objectives[i] < best_val:
                best_i = i
                best_val = self._objectives[i]
        return best_i

    @property
    def candidates(self) -> List[ST]:
        return self._candidates

    @property
    def objectives(self) -> List[OT]:
        return self._objectives

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def best_index(self) -> int:
        return self._best_index

    @property
    def best_solution(self) -> ST:
        return self._candidates[self._best_index]

    @property
    def best_objective(self) -> OT:
        return self._objectives[self._best_index]

    @property
    def current_solution(self) -> ST:
        # Convention: current is the last candidate
        return self._candidates[-1]

    @property
    def current_objective(self) -> OT:
        return self._objectives[-1]

    def update_single(self, solution: ST, objective: OT) -> None:
        """Append or replace current candidate with a new one and update best."""
        # Replace current (last) with new current
        if self._candidates:
            self._candidates[-1] = solution
            self._objectives[-1] = objective
        else:
            self._candidates.append(solution)
            self._objectives.append(objective)
        # Update best
        if objective < self._objectives[self._best_index]:
            self._best_index = len(self._objectives) - 1

    def replace_all(self, candidates: List[ST], objectives: List[OT]) -> None:
        if len(candidates) != len(objectives):
            raise ValueError("candidates and objectives must have the same length")
        if not candidates:
            raise ValueError("population must contain at least one candidate")
        self._candidates = list(candidates)
        self._objectives = list(objectives)
        self._best_index = self._compute_best_index()


__all__ = ["Population"]