from __future__ import annotations

from pathlib import Path
import sys

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

from genetic_algorithms import RealVectorInitializer, RealVectorGA, TournamentSelection, ArithmeticCrossover, UniformMutation, TopKElitism
from examples.eggholder.problem import EggholderProblem
from examples.eggholder.plotting import ContourPopulationPlotter, eggholder_function


def main() -> None:
    problem = EggholderProblem()

    # Components
    initializer = RealVectorInitializer(population_size=50)
    # Define GA operators explicitly (probabilities live in strategies)
    selection = TournamentSelection(k=3)
    crossover = ArithmeticCrossover(alpha=0.5, prob=0.9)
    mutation = UniformMutation(scale=2.0, prob=0.2, bounds=problem.bounds)
    elitism = TopKElitism[NDArrayFloat, float](k=2)

    updater = RealVectorGA(
        problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        elitism=elitism,
    )

    convergence = ConvergenceChecker[NDArrayFloat, float](
        strategies=[MaxEvaluationsStop(20_000)],
        iteration_strategy=MaxIterationsStop(500),
    )

    state = OptimizationState[NDArrayFloat, float]()
    dispatcher = EventDispatcher[NDArrayFloat, float]()
    # Add real-time plotting strategy
    plotter = ContourPopulationPlotter(eggholder_function, problem.bounds or [(-512.0, 512.0), (-512.0, 512.0)])
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)

    engine = OptimizationEngine[NDArrayFloat, float](
        initializer, updater, convergence, state, dispatcher
    )

    result = engine.run()
    print("Best solution:", result.best_solution)
    print("Best objective:", result.best_objective)
    print("Iterations:", result.iterations)
    print("Execution time (s):", result.execution_time)


if __name__ == "__main__":
    main()
