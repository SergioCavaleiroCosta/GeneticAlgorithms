"""Run NSGA-II for radiation-pressure multi-objective optimization."""
from __future__ import annotations
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Sequence

# Make project src/ importable when running the example directly
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from optimization.optimization_engine import OptimizationEngine
from optimization.state import OptimizationState
from optimization.convergence_checker import ConvergenceChecker, MaxIterationsStop, MaxEvaluationsStop
from optimization.events import EventDispatcher, Stage
from optimization.types import NDArrayFloat
from optimization.parameters import ContinuousParameter, LinearNormalization

from nsga_ii import NSGAII, MultiObjectiveInitializer
from nsga_ii.pareto_utils import non_dominated_sort

from problem import create_problem


def run_nsga2(
    population_size: int = 100,
    max_generations: int = 200,
    seed: int | None = None,
) -> None:
    """Run NSGA-II optimization.
    
    Args:
        population_size: Population size
        max_generations: Maximum number of generations
        seed: Random seed
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Create problem
    problem = create_problem()
    lower_bounds, upper_bounds = problem.get_bounds()
    
    # Create parameters
    param_names = list(problem.bounds.keys())
    parameters = [
        ContinuousParameter(
            name=param_names[i],
            normalizer=LinearNormalization(lower_bounds[i], upper_bounds[i]),
            description=param_names[i],
            unit="",
        )
        for i in range(len(param_names))
    ]
    
    # NSGA-II setup
    initializer = MultiObjectiveInitializer(
        population_size=population_size,
        parameters=parameters,
    )
    
    updater = NSGAII(
        problem,
        population_size=population_size,
    )
    
    # Convergence criteria
    max_evaluations = population_size * max_generations
    convergence = ConvergenceChecker[NDArrayFloat, Sequence[float]](
        strategies=[MaxEvaluationsStop(max_evaluations)],
        iteration_strategy=MaxIterationsStop(max_generations),
    )
    
    # State and events
    state = OptimizationState[NDArrayFloat, Sequence[float]]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, Sequence[float]]()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"output_{timestamp}"
    
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    print("\n" + "=" * 80)
    print("NSGA-II Multi-Objective Optimization")
    print("=" * 80)
    print(f"Problem: Radiation-Pressure MOOP")
    print(f"Objectives: Maximize η, Minimize ΔP")
    print(f"Variables: {len(lower_bounds)}D")
    print(f"Population size: {population_size}")
    print(f"Max generations: {max_generations}")
    if seed is not None:
        print(f"Random seed: {seed}")
    print("=" * 80)
    print()
    
    # Create and run optimization engine
    engine = OptimizationEngine[NDArrayFloat, Sequence[float]](
        initializer, updater, convergence, state, dispatcher, parameters=parameters
    )
    
    result = engine.run()
    
    # Results summary
    print("\n" + "=" * 80)
    print("NSGA-II Optimization Complete!")
    print("=" * 80)
    print(f"Iterations: {result.iterations}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Total evaluations: {problem.evaluation_count}")
    print(f"Population size: {len(engine.population.candidates)}")
    
    # Extract objectives
    objectives_list = engine.population.objectives
    objectives = np.array(objectives_list)
    solutions_array = np.array(engine.population.candidates)
    
    # Show objective value ranges
    print(f"\nObjective ranges:")
    print(f"  f1 (-η): [{objectives[:, 0].min():.6f}, {objectives[:, 0].max():.6f}]")
    print(f"  f2 (ΔP): [{objectives[:, 1].min():.2f}, {objectives[:, 1].max():.2f}]")
    print(f"  η:       [{-objectives[:, 0].max():.6f}, {-objectives[:, 0].min():.6f}]")
    
    # Find Pareto front
    fronts = non_dominated_sort(objectives)
    pareto_indices = fronts[0]  # First front is the Pareto front
    
    pareto_objectives = objectives[pareto_indices]
    pareto_solutions = solutions_array[pareto_indices]
    
    print(f"\nNumber of Pareto-optimal solutions: {len(pareto_indices)}")
    
    # Save configuration
    config = {
        "problem": "radiation_pressure_moop",
        "algorithm": "NSGA-II",
        "population_size": population_size,
        "max_generations": max_generations,
        "seed": seed,
        "num_variables": len(lower_bounds),
        "num_objectives": problem.num_objectives,
        "total_evaluations": problem.evaluation_count,
        "timestamp": datetime.now().isoformat(),
    }
    
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")
    
    # Save Pareto front
    pareto_data = {
        "objectives": pareto_objectives.tolist(),
        "solutions": pareto_solutions.tolist(),
        "num_pareto_solutions": len(pareto_indices),
    }
    
    pareto_path = output_path / "pareto_front.json"
    with open(pareto_path, "w") as f:
        json.dump(pareto_data, f, indent=2)
    
    print(f"Pareto front saved to: {pareto_path}")
    
    # Print summary of Pareto front
    print("\n" + "=" * 80)
    print("Pareto Front Summary")
    print("=" * 80)
    print(f"Objective 1 (-η): min = {pareto_objectives[:, 0].min():.6f}, "
          f"max = {pareto_objectives[:, 0].max():.6f}")
    print(f"Objective 2 (ΔP): min = {pareto_objectives[:, 1].min():.2f}, "
          f"max = {pareto_objectives[:, 1].max():.2f}")
    print()
    print(f"Radiative efficiency η: min = {-pareto_objectives[:, 0].max():.6f}, "
          f"max = {-pareto_objectives[:, 0].min():.6f}")
    print("=" * 80)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_nsga2(
        population_size=200,
        max_generations=500,
        seed=42,
    )
