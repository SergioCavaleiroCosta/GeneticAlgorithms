"""NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation."""
from __future__ import annotations

from typing import Optional, Sequence, List, Tuple, Any, cast
import random
import numpy as np

from optimization.update_rule import UpdateRule  
from optimization.events import EventDispatcher
from optimization.population import Population
from optimization.optimization_problem import OptimizationProblem
from optimization.types import NDArrayFloat
from genetic_algorithms.ops.crossover import CrossoverStrategy
from genetic_algorithms.ops.crossover.arithmetic import ArithmeticCrossover
from genetic_algorithms.ops.mutation import MutationStrategy
from genetic_algorithms.ops.mutation.gaussian import GaussianMutation
from .pareto_utils import non_dominated_sort, crowding_distance_assignment


class NSGAII(UpdateRule[NDArrayFloat, Any]):
    """NSGA-II algorithm for multi-objective optimization.
    
    Features:
    - Non-dominated sorting
    - Crowding distance for diversity preservation
    - Elitist selection
    - Standard genetic operators (crossover, mutation)
    
    Note: This uses Any for objective type to work with the existing framework
    while handling multi-objective sequences internally.
    """

    def __init__(
        self,
        problem: OptimizationProblem[NDArrayFloat, Any],
        *,
        population_size: int = 100,
        rng: Optional[random.Random] = None,
        crossover: Optional[CrossoverStrategy[NDArrayFloat]] = None,
        mutation: Optional[MutationStrategy[NDArrayFloat]] = None,
    ) -> None:
        self._problem = problem
        self._population_size = population_size
        self._rng = rng or random.Random()
        self._dispatcher: Optional[EventDispatcher[NDArrayFloat, Any]] = None

        # Genetic operators
        self._crossover: CrossoverStrategy[NDArrayFloat] = crossover or ArithmeticCrossover()
        self._mutation: MutationStrategy[NDArrayFloat] = mutation or GaussianMutation()

    def set_dispatcher(self, dispatcher: EventDispatcher[NDArrayFloat, Any]) -> None:
        self._dispatcher = dispatcher

    @property
    def problem(self) -> OptimizationProblem[NDArrayFloat, Any]:
        return self._problem

    def seed(self, initial_solution: NDArrayFloat, initial_objective: Any) -> None:
        # NSGA-II doesn't require special seeding beyond initial population
        pass

    def _evaluate_candidate(self, candidate: NDArrayFloat, engine: Any) -> Sequence[float]:
        """Evaluate a candidate solution."""
        try:
            params = getattr(engine, 'parameters', [])
            if params and len(params) == candidate.shape[0]:
                # Convert from normalized [0,1] domain to real domain
                real = np.array([p.normalizer.to_real(candidate[i]) for i, p in enumerate(params)], dtype=np.float64)
                result = self._problem.evaluate(real)
            else:
                result = self._problem.evaluate(candidate)
            
            # Ensure result is a sequence
            if isinstance(result, (list, tuple, np.ndarray)):
                return list(result)
            else:
                return [float(result)]  # Convert scalar to single-element list
        except Exception:
            # Fallback to direct evaluation
            result = self._problem.evaluate(candidate)
            if isinstance(result, (list, tuple, np.ndarray)):
                return list(result)
            else:
                return [float(result)]

    def _tournament_selection(self, 
                            candidates: List[NDArrayFloat], 
                            objectives: List[Sequence[float]],
                            ranks: List[int],
                            crowding_distances: List[float]) -> int:
        """Binary tournament selection using NSGA-II criteria."""
        n = len(candidates)
        if n == 0:
            raise ValueError("Cannot select from empty population")
        if n == 1:
            return 0
        
        i1 = self._rng.randrange(n)
        i2 = self._rng.randrange(n)
        
        if i1 == i2:
            return i1
        
        # Compare by rank first (lower rank is better)
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i1] > ranks[i2]:
            return i2
        else:
            # Same rank, compare by crowding distance (higher is better)
            if crowding_distances[i1] > crowding_distances[i2]:
                return i1
            else:
                return i2

    def _environmental_selection(self, 
                               combined_candidates: List[NDArrayFloat],
                               combined_objectives: List[Sequence[float]]) -> Tuple[List[NDArrayFloat], List[Sequence[float]]]:
        """Environmental selection using non-dominated sorting and crowding distance."""
        if len(combined_candidates) <= self._population_size:
            return combined_candidates, combined_objectives
        
        # Perform non-dominated sorting
        fronts = non_dominated_sort(combined_objectives)
        
        selected_candidates: List[NDArrayFloat] = []
        selected_objectives: List[Sequence[float]] = []
        
        # Add complete fronts until we exceed population size  
        for front in fronts:
            if len(selected_candidates) + len(front) <= self._population_size:
                # Add entire front
                for idx in front:
                    selected_candidates.append(combined_candidates[idx])
                    selected_objectives.append(combined_objectives[idx])
            else:
                # This front would exceed population size - need to select subset
                remaining_slots = self._population_size - len(selected_candidates)
                if remaining_slots > 0:
                    # Calculate crowding distances for this front
                    distances = crowding_distance_assignment(combined_objectives, front)
                    
                    # Sort by crowding distance (descending - higher distance is better)
                    front_with_distances = list(zip(front, distances))
                    front_with_distances.sort(key=lambda x: x[1], reverse=True)
                    
                    # Select the most diverse solutions
                    for i in range(remaining_slots):
                        idx = front_with_distances[i][0]
                        selected_candidates.append(combined_candidates[idx])
                        selected_objectives.append(combined_objectives[idx])
                break
        
        return selected_candidates, selected_objectives

    def step(self, engine: Any) -> None:
        """Perform one NSGA-II generation."""
        population = engine.population
        n = population.size
        
        if n == 0:
            raise RuntimeError("Population is empty; cannot step NSGA-II")
        
        # Ensure population size matches target
        if n != self._population_size:
            # Adjust population to target size (this is a simplification)
            current_candidates = population.candidates[:self._population_size]
            current_objectives = population.objectives[:self._population_size]
            population.replace_all(current_candidates, current_objectives)
            n = self._population_size
        
        # Get current population and convert objectives to sequences
        parent_candidates = population.candidates
        parent_objectives = []
        for obj in population.objectives:
            if isinstance(obj, (list, tuple, np.ndarray)):
                parent_objectives.append(list(obj))
            else:
                parent_objectives.append([float(obj)])
        
        # Perform non-dominated sorting and calculate crowding distances
        fronts = non_dominated_sort(parent_objectives)
        ranks = [0] * n
        crowding_distances = [0.0] * n
        
        for rank, front in enumerate(fronts):
            distances = crowding_distance_assignment(parent_objectives, front)
            for i, solution_idx in enumerate(front):
                ranks[solution_idx] = rank
                crowding_distances[solution_idx] = distances[i]
        
        # Generate offspring through selection, crossover, and mutation
        offspring_candidates: List[NDArrayFloat] = []
        offspring_objectives: List[Sequence[float]] = []
        
        while len(offspring_candidates) < self._population_size:
            # Select parents using tournament selection
            parent1_idx = self._tournament_selection(parent_candidates, parent_objectives, ranks, crowding_distances)
            parent2_idx = self._tournament_selection(parent_candidates, parent_objectives, ranks, crowding_distances)
            
            parent1 = parent_candidates[parent1_idx]
            parent2 = parent_candidates[parent2_idx]
            
            # Crossover
            child1, child2 = self._crossover.crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutation.mutate(child1)
            child2 = self._mutation.mutate(child2)
            
            # Ensure solutions remain in [0,1] domain
            np.clip(child1, 0.0, 1.0, out=child1)
            np.clip(child2, 0.0, 1.0, out=child2)
            
            # Evaluate offspring
            for child in [child1, child2]:
                if len(offspring_candidates) < self._population_size:
                    obj = self._evaluate_candidate(child, engine)
                    offspring_candidates.append(child)
                    offspring_objectives.append(obj)
        
        # Combine parents and offspring
        combined_candidates = parent_candidates + offspring_candidates
        combined_objectives = parent_objectives + offspring_objectives
        
        # Environmental selection
        new_candidates, new_objectives = self._environmental_selection(combined_candidates, combined_objectives)
        
        # Update population (convert back to original objective format for framework)
        population.replace_all(new_candidates, cast(List[Any], new_objectives))


__all__ = ["NSGAII"]