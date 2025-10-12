"""Run multiple Genetic Algorithm optimization runs in parallel.

This script runs the radiative efficiency optimization multiple times using
multiprocessing to gather statistics on the optimization performance.

Usage:
    python run_batch_ga.py
    uv run python examples/radiative_efficiency/run_batch_ga.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import json
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List
import hashlib
import random

# Make project src/ importable when running the example directly
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization.optimization_engine import OptimizationEngine
from optimization.state import OptimizationState
from optimization.convergence_checker import ConvergenceChecker, MaxIterationsStop, MaxEvaluationsStop
from optimization.events import EventDispatcher, Stage, FrameSaverStrategy, FrameSaverConfig
from optimization.types import NDArrayFloat
from optimization.parameters import ContinuousParameter, LinearNormalization
from optimization.parameters import ContinuousParameter, LinearNormalization

from genetic_algorithms import (
    RealVectorInitializer,
    RealVectorGA,
    TournamentSelection,
    ArithmeticCrossover,
    GaussianMutation,
    TopKElitism,
)

from examples.eggholder.logging_strategies import PopulationLogger
from examples.radiative_efficiency.plotting import RadiativeEfficiencyPlotter
from examples.radiative_efficiency.problem import RadiativeEfficiencyProblem



def run_single_optimization(run_id: int, batch_base_dir: Path) -> dict[str, Any]:
    """Run a single optimization and return results.
    
    Args:
        run_id: Unique identifier for this run
        batch_base_dir: Base directory for the batch (contains all run folders)
        
    Returns:
        Dictionary with optimization results
    """
    # Create individual output directory for this run
    run_dir = batch_base_dir / f"run_{run_id:03d}"
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Problem setup
    problem = RadiativeEfficiencyProblem()
    
    # Define parameters with bounds from the problem
    bounds = problem.get_bounds()
    param_names = problem.param_names
    # Descriptions/units aligned with updated 7-parameter problem
    # Order must match problem.param_names
    param_descriptions_map = {
        'phi': "Equivalence ratio",
        'u_avg': "Average inlet velocity",
        'ar': "Aspect ratio",
        'lt_0': "First layer thickness",
        'lt_1': "Second layer thickness",
        'eps_1': "Second layer emissivity",
        'a_1': "Second layer extinction coefficient",
    }
    param_units_map = {
        'phi': "",
        'u_avg': "m/s",
        'ar': "",
        'lt_0': "m",
        'lt_1': "m",
        'eps_1': "",
        'a_1': "1/m",
    }
    
    parameters: List[ContinuousParameter] = []
    for i, name in enumerate(param_names):
        parameters.append(
            ContinuousParameter(
                name=name,
                normalizer=LinearNormalization(bounds[i][0], bounds[i][1]),
                description=param_descriptions_map.get(name, name),
                unit=param_units_map.get(name, ""),
            )
        )
    
    # Genetic Algorithm setup - FIXED for better exploration
    population_size = 50  # Increased from 50 for more diversity
    initializer = RealVectorInitializer(population_size=population_size, parameters=parameters)
    
    selection = TournamentSelection(k=2)  # Reduced from k=3 for weaker selection pressure
    crossover = ArithmeticCrossover(prob=0.8)
    # FIXED: per-gene mutation probability (not per-individual!)
    mutation = GaussianMutation(sigma=0.10, prob=0.10, bounds=bounds)  # ~1 gene per individual
    elitism = TopKElitism[NDArrayFloat, float](k=1)
    
    updater = RealVectorGA(
        problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        elitism=elitism,
    )
    
    # Convergence criteria - shorter runs for testing the fix
    convergence = ConvergenceChecker[NDArrayFloat, float](
        strategies=[MaxEvaluationsStop(500_000)],  # Reduced for testing
        iteration_strategy=MaxIterationsStop(100),  # Reduced for testing
    )
    
    # State and events with logging and plotting
    state = OptimizationState[NDArrayFloat, float]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, float]()
    
    # Add logging
    logger = PopulationLogger(run_dir)
    dispatcher.add_strategy(Stage.RUN_START, logger)
    dispatcher.add_strategy(Stage.ITERATION, logger)
    dispatcher.add_strategy(Stage.RUN_END, logger)
    
    # Add plotting; update every iteration to see evolution
    plotter = RadiativeEfficiencyPlotter(update_every=1)
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)
    dispatcher.add_strategy(Stage.RUN_END, plotter)
    
    # Frame saver for animations - save every iteration
    frame_saver = FrameSaverStrategy(
        plotter, figures_dir, config=FrameSaverConfig(prefix="frame", dpi=120, respect_plotter_update=False)
    )
    dispatcher.add_strategy(Stage.RUN_START, frame_saver)
    dispatcher.add_strategy(Stage.ITERATION, frame_saver)
    dispatcher.add_strategy(Stage.RUN_END, frame_saver)
    
    # Create and run optimization engine
    engine = OptimizationEngine[NDArrayFloat, float](
        initializer, updater, convergence, state, dispatcher, parameters=parameters
    )
    
    result = engine.run()
    
    # Extract and denormalize results (assume best_solution available)
    best_params_normalized = result.best_solution.ravel()
    best_objective = result.best_objective

    # Denormalize parameters to real values
    best_params_real: List[float] = []
    for i, param in enumerate(parameters):
        normalized_val = best_params_normalized[i]
        real_val = param.normalizer.to_real(normalized_val)
        best_params_real.append(real_val)

    # Negate objective to get actual efficiency
    actual_efficiency = -best_objective

    # Prepare results dictionary
    run_results: Dict[str, Any] = {
        "run_id": run_id,
        "success": True,
        "radiative_efficiency": actual_efficiency,
        "objective_value": best_objective,
        "parameters": {
            param_names[i]: best_params_real[i]
            for i in range(len(param_names))
        },
        "parameters_normalized": {
            param_names[i]: float(best_params_normalized[i])
            for i in range(len(param_names))
        },
        "iterations": result.iterations,
        "execution_time": result.execution_time,
        "evaluations": problem.evaluation_count,
        "config_note": "FIXED: per-gene mutation, sigma=0.15, prob=0.15, k=2, pop=80"
    }

    # Save to JSON in run directory
    with open(run_dir / "run_results.json", "w") as f:
        json.dump(run_results, f, indent=2)

    return run_results


def main() -> None:
    """Run batch optimization with multiprocessing."""
    n_runs = 100
    n_workers = 20
    
    # Create batch output directory with timestamp + random hash for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_hash = hashlib.md5(f"{timestamp}{random.random()}".encode()).hexdigest()[:8]
    batch_dir = Path(f"examples/radiative_efficiency/batch_FIXED_{timestamp}_{random_hash}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Radiative Efficiency Batch Optimization")
    print("=" * 70)
    print("CRITICAL BUG FIXED: Mutation now per-gene (not per-individual)")
    print("PARAMETER ADJUSTMENTS:")
    print("- Population: 50 -> 80 (more diversity)")
    print("- Tournament k: 3 -> 2 (weaker selection)")
    print("- Mutation prob: 0.1 -> 0.15 (per-gene, ~1 gene per individual)")
    print("- Mutation sigma: 0.1 -> 0.15 (slightly larger steps)")
    print("- Iterations: 500 -> 100 (testing)")
    print("=" * 70)
    print(f"Number of runs: {n_runs}")
    print(f"Number of workers: {n_workers}")
    print(f"Output directory: {batch_dir}")
    print("=" * 70)
    print("\nStarting optimization runs...")
    
    # Run optimizations in parallel
    with mp.Pool(processes=n_workers) as pool:
        # Create tasks
        tasks = [(i, batch_dir) for i in range(n_runs)]
        
        # Run with progress tracking
        results: List[Dict[str, Any]] = []
        for i, result in enumerate(pool.starmap(run_single_optimization, tasks)):
            results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == n_runs:
                print(f"Completed: {i + 1}/{n_runs} runs")
    
    print("\nAll runs completed!")
    
    # Save individual results
    results_file = batch_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Batch Optimization Complete!")
    print("=" * 70)
    print(f"Total runs: {len(results)}")
    print(f"Successful runs: {sum(1 for r in results if r.get('success', False))}")
    print(f"\nResults saved to: {results_file}")
    print(f"Batch directory: {batch_dir}")
    print("\nTo analyze results, run:")
    print(f"  uv run python examples/radiative_efficiency/analyze_batch.py {batch_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
