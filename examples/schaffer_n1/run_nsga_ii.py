"""Run NSGA-II on the Schaffer N.1 multi-objective test problem."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence
from datetime import datetime
import numpy as np

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
from examples.schaffer_n1.problems import SchafferN1Problem
from examples.schaffer_n1.plotting import ParetoPlotter, MultiObjectivePopulationLogger, plot_pareto_front, plot_decision_space


def main() -> None:
    # Problem setup
    domain_bound = 10.0  # x ∈ [-10, 10] (smaller than typical 1000 for easier visualization)
    problem = SchafferN1Problem(domain_bound=domain_bound)
    
    print(f"Schaffer N.1 Configuration:")
    print(f"  Variables: 1 (x)")
    print(f"  Domain: x in [{-domain_bound}, {domain_bound}]")
    print(f"  Objectives: f1(x) = x^2, f2(x) = (x-2)^2")
    print(f"  Pareto optimal region: x in [0, 2]")
    
    # Parameters (single variable in [-10, 10])
    parameters = [
        ContinuousParameter(
            name="x",
            normalizer=LinearNormalization(-domain_bound, domain_bound),
            description="Schaffer N.1 decision variable",
            unit="units",
        )
    ]
    
    # NSGA-II setup
    population_size = 50  # Smaller population for single variable
    initializer = MultiObjectiveInitializer(population_size=population_size, parameters=parameters)
    
    updater = NSGAII(
        problem,
        population_size=population_size,
    )
    
    # Convergence criteria (fewer evaluations needed for simple problem)
    convergence = ConvergenceChecker[NDArrayFloat, Sequence[float]](
        strategies=[MaxEvaluationsStop(5_000)],
        iteration_strategy=MaxIterationsStop(100),
    )
    
    # State and events
    state = OptimizationState[NDArrayFloat, Sequence[float]]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, Sequence[float]]()
    
    # Output directories (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"examples/schaffer_n1/output_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging and visualization
    logger = MultiObjectivePopulationLogger(out_dir)
    dispatcher.add_strategy(Stage.RUN_START, logger)
    dispatcher.add_strategy(Stage.ITERATION, logger) 
    dispatcher.add_strategy(Stage.RUN_END, logger)
    
    plotter = ParetoPlotter(out_dir, plot_every=10)
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
    
    # Show decision variable range
    x_vals = [float(np.asarray(candidate).flatten()[0]) for candidate in engine.population.candidates]
    
    print(f"f1 range: [{min(f1_vals):.4f}, {max(f1_vals):.4f}]")
    print(f"f2 range: [{min(f2_vals):.4f}, {max(f2_vals):.4f}]")
    print(f"x range: [{min(x_vals):.4f}, {max(x_vals):.4f}]")
    print(f"Pareto optimal x in [0, 2]")
    print(f"Output written to: {out_dir}")
    
    # Create final plots
    final_iteration = result.iterations or 0
    plot_pareto_front(objectives, out_dir, final_iteration)
    plot_decision_space(engine.population.candidates, out_dir, final_iteration)
    print(f"Final plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()