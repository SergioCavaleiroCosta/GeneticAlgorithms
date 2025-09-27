"""Run NSGA-II on the DTLZ2 multi-objective test problem."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence
from datetime import datetime

# Make project src/ importable when running the example directly
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization.optimization_engine import OptimizationEngine
from optimization.state import OptimizationState
from optimization.convergence_checker import ConvergenceChecker, MaxIterationsStop, MaxEvaluationsStop
from optimization.events import EventDispatcher, Stage
from optimization.types import NDArrayFloat
from optimization.parameters import ContinuousParameter, LinearNormalization

from nsga_ii import NSGAII, MultiObjectiveInitializer
from examples.dtlz2.problems import DTLZ2Problem
from examples.dtlz2.plotting import ParetoPlotter, MultiObjectivePopulationLogger, plot_pareto_front


def main() -> None:
    # Problem setup - can be configured for different numbers of objectives
    num_objectives = 2  # Try 2, 3, or more
    dimension = num_objectives + 10  # Recommended: M + 10 variables
    problem = DTLZ2Problem(num_objectives=num_objectives, dimension=dimension)
    
    print(f"DTLZ2 Configuration:")
    print(f"  Objectives: {num_objectives}")
    print(f"  Variables: {dimension}")
    print(f"  Expected Pareto front: {num_objectives}D hyper-sphere section")
    
    # Parameters (all in [0,1] for DTLZ2)
    parameters = [
        ContinuousParameter(
            name=f"x{i}",
            normalizer=LinearNormalization(0.0, 1.0),  # DTLZ2 domain is [0,1]
            description=f"DTLZ2 dimension {i}",
            unit="units",
        )
        for i in range(dimension)
    ]
    
    # NSGA-II setup
    population_size = 100
    initializer = MultiObjectiveInitializer(population_size=population_size, parameters=parameters)
    
    updater = NSGAII(
        problem,
        population_size=population_size,
    )
    
    # Convergence criteria
    convergence = ConvergenceChecker[NDArrayFloat, Sequence[float]](
        strategies=[MaxEvaluationsStop(25_000)],
        iteration_strategy=MaxIterationsStop(250),
    )
    
    # State and events
    state = OptimizationState[NDArrayFloat, Sequence[float]]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, Sequence[float]]()
    
    # Output directories (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"examples/dtlz2/output_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging and visualization
    logger = MultiObjectivePopulationLogger(out_dir, num_objectives=num_objectives)
    dispatcher.add_strategy(Stage.RUN_START, logger)
    dispatcher.add_strategy(Stage.ITERATION, logger) 
    dispatcher.add_strategy(Stage.RUN_END, logger)
    
    plotter = ParetoPlotter(out_dir, plot_every=25, num_objectives=num_objectives)
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)
    dispatcher.add_strategy(Stage.RUN_END, plotter)
    
    # Create and run optimization engine
    engine = OptimizationEngine[NDArrayFloat, Sequence[float]](
        initializer, updater, convergence, state, dispatcher, parameters=parameters
    )
    
    result = engine.run()
    
    # Results summary
    print("NSGA-II Optimization Complete!")
    print(f"Iterations: {result.iterations}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Total evaluations: {problem.evaluation_count}")
    print(f"Population size: {len(engine.population.candidates)}")
    
    # Show objective value ranges
    objectives = engine.population.objectives
    for i in range(num_objectives):
        f_vals = [obj[i] for obj in objectives]
        print(f"f{i+1} range: [{min(f_vals):.4f}, {max(f_vals):.4f}]")
    
    print(f"Output written to: {out_dir}")
    
    # Create final plot with actual final generation number
    final_iteration = result.iterations or 0
    plot_pareto_front(objectives, out_dir, final_iteration, num_objectives)
    print(f"Final Pareto front plot saved to: {out_dir}/final_pareto_front.png")


if __name__ == "__main__":
    main()