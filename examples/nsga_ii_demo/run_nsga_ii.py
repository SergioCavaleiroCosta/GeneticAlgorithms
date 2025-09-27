"""Run NSGA-II on a multi-objective test problem."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

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
from examples.nsga_ii_demo.problems import ZDT1Problem, true_pareto_front_zdt1
from datetime import datetime


class MultiObjectivePopulationLogger:
    """Logger for multi-objective population data."""
    
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.objectives_file = self.output_dir / "objectives.csv"
        self.solutions_file = self.output_dir / "solutions.csv"
        
        # Initialize files with headers
        with open(self.objectives_file, 'w') as f:
            f.write("iteration,individual,f1,f2\n")
        with open(self.solutions_file, 'w') as f:
            f.write("iteration,individual,solution\n")
    
    def execute(self, engine, stage) -> None:
        iteration = getattr(engine._convergence, 'iteration', 0)
        population = engine.population
        
        # Log objectives
        with open(self.objectives_file, 'a') as f:
            for i, objectives in enumerate(population.objectives):
                f.write(f"{iteration},{i},{objectives[0]},{objectives[1]}\n")
        
        # Log solutions (first 5 dimensions only to avoid huge files)
        with open(self.solutions_file, 'a') as f:
            for i, solution in enumerate(population.candidates):
                sol_str = ",".join(f"{x:.6f}" for x in solution[:5])
                f.write(f"{iteration},{i},{sol_str}\n")


def plot_pareto_front(objectives: list[Sequence[float]], output_dir: Path, iteration: int = -1) -> None:
    """Plot the current population and true Pareto front."""
    # Extract objective values
    f1_vals = [obj[0] for obj in objectives]
    f2_vals = [obj[1] for obj in objectives]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(f1_vals, f2_vals, alpha=0.6, s=30, label=f'NSGA-II Population (Gen {iteration})')
    
    # Plot true Pareto front
    true_front = true_pareto_front_zdt1(100)
    plt.plot(true_front[:, 0], true_front[:, 1], 'r-', linewidth=2, label='True Pareto Front')
    
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'NSGA-II on ZDT1 Problem - Generation {iteration}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    if iteration >= 0:
        filename = f"pareto_front_gen_{iteration:03d}.png"
    else:
        filename = "final_pareto_front.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


class ParetoPlotter:
    """Event strategy for plotting Pareto fronts during optimization."""
    
    def __init__(self, output_dir: Path, plot_every: int = 10) -> None:
        self.output_dir = output_dir
        self.plot_every = plot_every
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, engine, stage) -> None:
        iteration = getattr(engine._convergence, 'iteration', 0)
        
        if iteration % self.plot_every == 0 or iteration == 0:
            objectives = engine.population.objectives
            plot_pareto_front(objectives, self.figures_dir, iteration)


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
    
    # Create final plot
    plot_pareto_front(objectives, out_dir, -1)
    print(f"Final Pareto front plot saved to: {out_dir}/final_pareto_front.png")


if __name__ == "__main__":
    main()