from __future__ import annotations

from pathlib import Path
import sys
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

from genetic_algorithms import (
    RealVectorInitializer,
    RealVectorGA,
    TournamentSelection,
    ArithmeticCrossover,
    UniformMutation,
    TopKElitism,
)
from optimization.parameters import ContinuousParameter, LinearNormalization
from examples.sphere.problem import SphereProblem
from examples.sphere.plotting import SpherePlotter


def main() -> None:
    dimension = 10
    problem = SphereProblem(dimension=dimension)

    # Parameter specifications for each dimension
    parameters = [
        ContinuousParameter(
            name=f"x{i}",
            normalizer=LinearNormalization(-5.12, 5.12),
            description=f"Sphere dimension {i}",
            unit="units",
        )
        for i in range(dimension)
    ]

    # Components
    population_size = 80
    initializer = RealVectorInitializer(population_size=population_size, parameters=parameters)
    selection = TournamentSelection(k=3)
    crossover = ArithmeticCrossover(prob=0.9)
    mutation = UniformMutation(scale=0.15, prob=0.25, normalized=True)
    elitism = TopKElitism[NDArrayFloat, float](k=2)

    updater = RealVectorGA(
        problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        elitism=elitism,
    )

    convergence = ConvergenceChecker[NDArrayFloat, float](
        strategies=[MaxEvaluationsStop(50_000)],
        iteration_strategy=MaxIterationsStop(100),
    )

    state = OptimizationState[NDArrayFloat, float]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, float]()

    # Plot selected pair (defaults to first two). The remaining 8 dimensions are
    # dynamically fixed to the CURRENT BEST individual's real values each time
    # the best changes (hyperplane follows the best). This is already provided
    # by the base plotter when fixed_values is not specified.
    # Focus visualization on x3 vs x5 (zero-based indices 3 and 5). All other
    # dimensions are dynamically fixed to the real values of the CURRENT BEST
    # individual (hyperplane follows the best). This leverages the built-in
    # dynamic best fill: we provide no fixed_values so the plotter uses the
    # best solution for remaining coordinates.
    plotter = SpherePlotter(update_every=1, param_pair=(3, 5))
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)

    engine = OptimizationEngine[NDArrayFloat, float](
        initializer, updater, convergence, state, dispatcher, parameters=parameters
    )

    result = engine.run()

    # Denormalize best solution for reporting
    norm = result.best_solution
    best_real = np.array([p.normalizer.to_real(norm[i]) for i, p in enumerate(parameters)], dtype=float)
    print("Best solution (normalized):", result.best_solution)
    print("Best solution (real):", best_real)
    print("Best objective:", result.best_objective)
    print("Iterations:", result.iterations)
    print("Execution time (s):", result.execution_time)


if __name__ == "__main__":
    main()
