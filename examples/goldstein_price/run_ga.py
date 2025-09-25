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
from examples.goldstein_price.problem import GoldsteinPriceProblem
from examples.goldstein_price.plotting import GoldsteinPricePlotter


def main() -> None:
    problem = GoldsteinPriceProblem()

    parameters = [
        ContinuousParameter(
            name="x",
            normalizer=LinearNormalization(-2.0, 2.0),
            description="Goldstein-Price x dimension",
            unit="units",
        ),
        ContinuousParameter(
            name="y",
            normalizer=LinearNormalization(-2.0, 2.0),
            description="Goldstein-Price y dimension",
            unit="units",
        ),
    ]

    population_size = 60
    initializer = RealVectorInitializer(population_size=population_size, parameters=parameters)
    selection = TournamentSelection(k=3)
    crossover = ArithmeticCrossover(prob=0.9)
    mutation = UniformMutation(scale=0.2, prob=0.25, normalized=True)
    elitism = TopKElitism[NDArrayFloat, float](k=1)

    updater = RealVectorGA(
        problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        elitism=elitism,
    )

    convergence = ConvergenceChecker[NDArrayFloat, float](
        strategies=[MaxEvaluationsStop(50_000)],
        iteration_strategy=MaxIterationsStop(150),
    )

    state = OptimizationState[NDArrayFloat, float]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, float]()

    plotter = GoldsteinPricePlotter(update_every=1)
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)

    engine = OptimizationEngine[NDArrayFloat, float](
        initializer, updater, convergence, state, dispatcher, parameters=parameters
    )

    result = engine.run()

    norm = result.best_solution
    best_real = np.array([p.normalizer.to_real(norm[i]) for i, p in enumerate(parameters)], dtype=float)
    print("Best solution (normalized):", result.best_solution)
    print("Best solution (real):", best_real)
    print("Best objective:", result.best_objective)
    print("Iterations:", result.iterations)
    print("Execution time (s):", result.execution_time)


if __name__ == "__main__":
    main()
