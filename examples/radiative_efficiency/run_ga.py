"""Run Genetic Algorithm on Radiative Efficiency Optimization.

This script runs a genetic algorithm to optimize the 7 parameters of a
radiative efficiency model (trained using PySR symbolic regression).

The objective is to MAXIMIZE radiative efficiency by finding optimal
combinations of fluidized bed combustion parameters.

Usage:
    python run_ga.py
    uv run python examples/radiative_efficiency/run_ga.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import json
from datetime import datetime

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

from genetic_algorithms import (
    RealVectorInitializer,
    RealVectorGA,
    TournamentSelection,
    ArithmeticCrossover,
    GaussianMutation,
    TopKElitism,
)

from examples.radiative_efficiency.problem import RadiativeEfficiencyProblem
from examples.radiative_efficiency.plotting import RadiativeEfficiencyPlotter
from examples.eggholder.logging_strategies import PopulationLogger


def main() -> None:
    # Problem setup - maximize radiative efficiency
    problem = RadiativeEfficiencyProblem()
    
    print("=" * 70)
    print(problem.describe())
    print("=" * 70)
    
    # Define parameters with bounds from the problem
    bounds = problem.get_bounds()
    param_names = problem.param_names
    param_descriptions = [
        "Equivalence ratio",
        "Average inlet velocity",
        "Aspect ratio",
        "Second layer thickness",
        "First layer extinction coefficient"
    ]
    param_units = ["", "m/s", "", "m", "1/m"]
    
    parameters = [
        ContinuousParameter(
            name=param_names[i],
            normalizer=LinearNormalization(bounds[i][0], bounds[i][1]),
            description=param_descriptions[i],
            unit=param_units[i],
        )
        for i in range(5)
    ]
    
    # Genetic Algorithm setup
    population_size = 100
    initializer = RealVectorInitializer(population_size=population_size, parameters=parameters)
    
    selection = TournamentSelection(k=3)
    crossover = ArithmeticCrossover(prob=0.8)
    mutation = GaussianMutation(sigma=0.1, prob=0.1, bounds=bounds)
    elitism = TopKElitism[NDArrayFloat, float](k=1)
    
    updater = RealVectorGA(
        problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        elitism=elitism,
    )
    
    # Convergence criteria - radiative efficiency is complex, may need more evaluations
    convergence = ConvergenceChecker[NDArrayFloat, float](
        strategies=[MaxEvaluationsStop(50_000)],
        iteration_strategy=MaxIterationsStop(50),
    )
    
    # State and events
    state = OptimizationState[NDArrayFloat, float]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, float]()
    
    # Output directories (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"examples/radiative_efficiency/output_{timestamp}")
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging and visualization
    logger = PopulationLogger(out_dir)
    dispatcher.add_strategy(Stage.RUN_START, logger)
    dispatcher.add_strategy(Stage.ITERATION, logger) 
    dispatcher.add_strategy(Stage.RUN_END, logger)
    
    plotter = RadiativeEfficiencyPlotter(update_every=1)
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)
    dispatcher.add_strategy(Stage.RUN_END, plotter)
    
    # Frame saver for animations
    frame_saver = FrameSaverStrategy(
        plotter, figures_dir, config=FrameSaverConfig(prefix="frame", dpi=120)
    )
    dispatcher.add_strategy(Stage.RUN_START, frame_saver)
    dispatcher.add_strategy(Stage.ITERATION, frame_saver)
    dispatcher.add_strategy(Stage.RUN_END, frame_saver)
    
    # Create and run optimization engine
    engine = OptimizationEngine[NDArrayFloat, float](
        initializer, updater, convergence, state, dispatcher, parameters=parameters
    )
    
    result = engine.run()
    
    # Results summary
    print("\n" + "=" * 70)
    print("Genetic Algorithm Optimization Complete!")
    print("=" * 70)
    print(f"Iterations: {result.iterations}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Total evaluations: {problem.evaluation_count}")
    
    # Best solution
    if result.best_solution is not None:
        best_params_normalized = result.best_solution.ravel()
        best_objective = result.best_objective
        
        # Denormalize parameters to real values
        best_params_real = []
        for i, param in enumerate(parameters):
            normalized_val = best_params_normalized[i]
            real_val = param.normalizer.to_real(normalized_val)
            best_params_real.append(real_val)
        
        # Remember: we minimized negative efficiency, so negate to get actual efficiency
        actual_efficiency = -best_objective
        
        print(f"\n{'Best Solution':^70}")
        print("=" * 70)
        print(f"Radiative Efficiency: {actual_efficiency:.6f}")
        print(f"\nOptimal Parameters:")
        for i, (name, desc, unit) in enumerate(zip(param_names, param_descriptions, param_units)):
            unit_str = f" {unit}" if unit else ""
            print(f"  {name:12s} = {best_params_real[i]:10.6f}{unit_str:8s}  ({desc})")
        
        # Save results to JSON file
        results_dict = {
            "optimization_info": {
                "problem": "Radiative Efficiency Optimization",
                "model_id": "20251010_162508_k1g9rP",
                "timestamp": timestamp,
                "iterations": result.iterations,
                "execution_time_seconds": result.execution_time,
                "total_evaluations": problem.evaluation_count
            },
            "best_solution": {
                "radiative_efficiency": actual_efficiency,
                "objective_value": best_objective,
                "parameters": {
                    param_names[i]: {
                        "value": best_params_real[i],
                        "normalized_value": float(best_params_normalized[i]),
                        "unit": param_units[i] if param_units[i] else "dimensionless",
                        "description": param_descriptions[i],
                        "bounds": {
                            "min": bounds[i][0],
                            "max": bounds[i][1]
                        }
                    }
                    for i in range(len(param_names))
                }
            },
            "algorithm_config": {
                "population_size": population_size,
                "selection": "TournamentSelection(k=3)",
                "crossover": "ArithmeticCrossover(prob=0.8)",
                "mutation": "GaussianMutation(sigma=0.1, prob=0.1)",
                "elitism": "TopKElitism(k=1)",
                "max_iterations": 50,
                "max_evaluations": 50000
            }
        }
        
        results_file = out_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Output directory: {out_dir}")
    else:
        print("Warning: No best candidate found!")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
