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
from examples.eggholder.plotting import EggholderPlotter


def main() -> None:
    problem = EggholderProblem()

    # Components
    initializer = RealVectorInitializer(population_size=50)
    # Define GA operators explicitly (probabilities live in strategies)
    selection = TournamentSelection(k=3)
    crossover = ArithmeticCrossover(alpha=0.5, prob=0.9)
    # Internal representation is normalized [0,1]^d
    mutation = UniformMutation(scale=0.2, prob=0.2, normalized=True)
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
        iteration_strategy=MaxIterationsStop(50),
    )

    state = OptimizationState[NDArrayFloat, float]()
    dispatcher = EventDispatcher[NDArrayFloat, float]()
    # Add real-time plotting strategy (auto-derives bounds & denormalizes internally)
    plotter = EggholderPlotter(update_every=1)
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)

    engine = OptimizationEngine[NDArrayFloat, float](
        initializer, updater, convergence, state, dispatcher
    )

    result = engine.run()
    # Denormalize best solution for reporting
    params = getattr(problem, "parameters", None)
    best_real = None
    if params is not None:
        import numpy as np  # local import to avoid unused if not used
        norm = result.best_solution
        best_real = np.array([p.normalizer.to_real(norm[i]) for i, p in enumerate(params)], dtype=float)
    print("Best solution (normalized):", result.best_solution)
    if best_real is not None:
        print("Best solution (real):", best_real)
    print("Best objective:", result.best_objective)
    print("Iterations:", result.iterations)
    print("Execution time (s):", result.execution_time)


if __name__ == "__main__":
    main()
