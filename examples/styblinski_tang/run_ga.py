from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

# Ensure src/ is importable when executing example directly
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
from examples.styblinski_tang.problem import StyblinskiTangProblem
from examples.styblinski_tang.plotting import StyblinskiTangPlotter


def main() -> None:
    dim = 10
    problem = StyblinskiTangProblem(dimension=dim)

    parameters = [
        ContinuousParameter(
            name=f"x{i+1}",
            normalizer=LinearNormalization(-5.0, 5.0),
            description=f"Styblinski–Tang dimension {i+1}",
            unit="units",
        )
        for i in range(dim)
    ]

    # Slightly larger population & evaluation budget because the function has many local minima.
    population_size = 120
    initializer = RealVectorInitializer(population_size=population_size, parameters=parameters)
    selection = TournamentSelection(k=3)
    crossover = ArithmeticCrossover(prob=0.9)
    mutation = UniformMutation(scale=0.25, prob=0.3, normalized=True)
    elitism = TopKElitism[NDArrayFloat, float](k=2)

    updater = RealVectorGA(
        problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        elitism=elitism,
    )

    convergence = ConvergenceChecker[NDArrayFloat, float](
        strategies=[MaxEvaluationsStop(120_000)],
        iteration_strategy=MaxIterationsStop(250),
    )

    state = OptimizationState[NDArrayFloat, float]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, float]()

    plotter = StyblinskiTangPlotter(update_every=2)
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)

    engine = OptimizationEngine[NDArrayFloat, float](
        initializer, updater, convergence, state, dispatcher, parameters=parameters
    )

    result = engine.run()

    norm = result.best_solution
    best_real = np.array([p.normalizer.to_real(norm[i]) for i, p in enumerate(parameters)], dtype=float)
    print("Best solution (normalized first 5):", result.best_solution[:5])
    print("Best solution (real first 5):", best_real[:5])
    print("Best objective:", result.best_objective)
    print("Iterations:", result.iterations)
    print("Execution time (s):", result.execution_time)


if __name__ == "__main__":
    main()
