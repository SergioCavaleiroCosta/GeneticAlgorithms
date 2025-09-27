"""Run NSGA-II on a multi-objective test problem."""
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
from examples.nsga_ii_demo.problems import ZDT1Problem
from examples.nsga_ii_demo.plotting import ParetoPlotter, MultiObjectivePopulationLogger, plot_pareto_front





def main() -> None:
    # Problem setup
    dimension = 10
    problem = ZDT1Problem(dimension=dimension)
    
    # Parameters (all in [0,1] for ZDT1)
    parameters = [
        ContinuousParameter(
            name=f"x{i}",
            normalizer=LinearNormalization(0.0, 1.0),  # ZDT1 domain is [0,1]
            description=f"ZDT1 dimension {i}",
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
    out_dir = Path(f"examples/nsga_ii_demo/output_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging and visualization
    logger = MultiObjectivePopulationLogger(out_dir)
    dispatcher.add_strategy(Stage.RUN_START, logger)
    dispatcher.add_strategy(Stage.ITERATION, logger) 
    dispatcher.add_strategy(Stage.RUN_END, logger)
    
    plotter = ParetoPlotter(out_dir, plot_every=25)
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
    f1_vals = [obj[0] for obj in objectives]
    f2_vals = [obj[1] for obj in objectives]
    
    print(f"f1 range: [{min(f1_vals):.4f}, {max(f1_vals):.4f}]")
    print(f"f2 range: [{min(f2_vals):.4f}, {max(f2_vals):.4f}]")
    print(f"Output written to: {out_dir}")
    
    # Create final plot with actual final generation number
    plot_pareto_front(objectives, out_dir, result.iterations)
    print(f"Final Pareto front plot saved to: {out_dir}/final_pareto_front.png")


if __name__ == "__main__":
    main()