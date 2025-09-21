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
        # Allow empty population; best index will be -1 until first candidate arrives
        self._candidates: List[ST] = list(candidates)
        self._objectives: List[OT] = list(objectives)
        self._best_index: int = self._compute_best_index()

    @classmethod
    def from_single(cls, solution: ST, objective: OT) -> Population[ST, OT]:
        return cls([solution], [objective])

    def _compute_best_index(self) -> int:
        if not self._objectives:
            return -1
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
        if self.size == 0 or self._best_index < 0:
            raise RuntimeError("Population is empty; no best index available")
        return self._best_index

    @property
    def best_solution(self) -> ST:
        if self.size == 0 or self._best_index < 0:
            raise RuntimeError("Population is empty; no best solution available")
        return self._candidates[self._best_index]

    @property
    def best_objective(self) -> OT:
        if self.size == 0 or self._best_index < 0:
            raise RuntimeError("Population is empty; no best objective available")
        return self._objectives[self._best_index]

    @property
    def current_solution(self) -> ST:
        # Convention: current is the last candidate
        if self.size == 0:
            raise RuntimeError("Population is empty; no current solution available")
        return self._candidates[-1]

    @property
    def current_objective(self) -> OT:
        if self.size == 0:
            raise RuntimeError("Population is empty; no current objective available")
        return self._objectives[-1]

    def update_single(self, solution: ST, objective: OT) -> None:
        """Append or replace current candidate with a new one and update best."""
        # Replace current (last) with new current or append if empty
        if self._candidates:
            self._candidates[-1] = solution
            self._objectives[-1] = objective
        else:
            self._candidates.append(solution)
            self._objectives.append(objective)
        # Recompute best to keep invariant correct even when current worsens
        self._best_index = self._compute_best_index()

    def replace_all(self, candidates: List[ST], objectives: List[OT]) -> None:
        if len(candidates) != len(objectives):
            raise ValueError("candidates and objectives must have the same length")
        # Allow replacing with an empty population
        self._candidates = list(candidates)
        self._objectives = list(objectives)
        self._best_index = self._compute_best_index()


__all__ = ["Population"]