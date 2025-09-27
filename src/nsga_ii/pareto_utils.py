"""Pareto dominance utilities and non-dominated sorting for multi-ob    # Build subsequent fronts
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        if next_front:
            fronts.append(next_front)
        current_front += 1
    
    # Remove empty last front if it exists
    if fronts and not fronts[-1]:
        fronts.pop()ion."""
from __future__ import annotations
from typing import List, Sequence


def is_dominated(obj1: Sequence[float], obj2: Sequence[float]) -> bool:
    """Check if obj1 is dominated by obj2 (minimization assumed).
    
    obj1 is dominated by obj2 if:
    - obj2 is at least as good as obj1 in all objectives, AND
    - obj2 is strictly better than obj1 in at least one objective
    
    Args:
        obj1: First objective vector
        obj2: Second objective vector
        
    Returns:
        True if obj1 is dominated by obj2, False otherwise
    """
    if len(obj1) != len(obj2):
        raise ValueError("Objective vectors must have the same length")
    
    at_least_as_good = True
    strictly_better = False
    
    for o1, o2 in zip(obj1, obj2):
        if o2 > o1:  # obj2 is worse in this objective
            at_least_as_good = False
            break
        if o2 < o1:  # obj2 is better in this objective
            strictly_better = True
    
    return at_least_as_good and strictly_better


def non_dominated_sort(objectives: List[Sequence[float]]) -> List[List[int]]:
    """Perform non-dominated sorting on a set of objective vectors.
    
    Args:
        objectives: List of objective vectors
        
    Returns:
        List of fronts, where each front is a list of indices into the objectives list
    """
    n = len(objectives)
    if n == 0:
        return []
    
    # Initialize data structures
    domination_count = [0] * n  # Number of solutions that dominate solution i
    dominated_solutions: List[List[int]] = [[] for _ in range(n)]  # Solutions dominated by solution i
    fronts: List[List[int]] = [[]]  # List of fronts
    
    # Compare all pairs
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_dominated(objectives[i], objectives[j]):
                    # Solution j dominates solution i
                    domination_count[i] += 1
                elif is_dominated(objectives[j], objectives[i]):
                    # Solution i dominates solution j
                    dominated_solutions[i].append(j)
        
        # If no solution dominates i, it belongs to the first front
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Build subsequent fronts
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front: List[int] = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        if next_front:
            fronts.append(next_front)
        current_front += 1
    
    # Remove empty last front if it exists
    if fronts and not fronts[-1]:
        fronts.pop()
    
    return fronts


def crowding_distance_assignment(objectives: List[Sequence[float]], front: List[int]) -> List[float]:
    """Calculate crowding distance for solutions in a front.
    
    Args:
        objectives: List of all objective vectors
        front: Indices of solutions in the current front
        
    Returns:
        List of crowding distances for solutions in the front (same order as front)
    """
    if len(front) == 0:
        return []
    
    if len(front) <= 2:
        # For fronts with 1-2 solutions, assign infinite distance
        return [float('inf')] * len(front)
    
    num_objectives = len(objectives[front[0]])
    distances = [0.0] * len(front)
    
    # For each objective
    for obj_idx in range(num_objectives):
        # Sort front by this objective
        front_with_values = [(front[i], objectives[front[i]][obj_idx]) for i in range(len(front))]
        front_with_values.sort(key=lambda x: x[1])
        
        # Find the range for normalization
        obj_min = front_with_values[0][1]
        obj_max = front_with_values[-1][1]
        obj_range = obj_max - obj_min
        
        # Boundary solutions get infinite distance
        distances[front.index(front_with_values[0][0])] = float('inf')
        distances[front.index(front_with_values[-1][0])] = float('inf')
        
        # Calculate distances for intermediate solutions
        if obj_range > 0:  # Avoid division by zero
            for i in range(1, len(front_with_values) - 1):
                solution_idx = front_with_values[i][0]
                next_obj = front_with_values[i + 1][1]
                prev_obj = front_with_values[i - 1][1]
                
                # Add normalized distance for this objective
                front_position = front.index(solution_idx)
                if distances[front_position] != float('inf'):
                    distances[front_position] += (next_obj - prev_obj) / obj_range
    
    return distances


__all__ = ["is_dominated", "non_dominated_sort", "crowding_distance_assignment"]