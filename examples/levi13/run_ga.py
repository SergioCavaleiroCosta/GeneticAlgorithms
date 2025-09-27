from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
from datetime import datetime

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
from examples.levi13.problem import Levi13Problem
from examples.levi13.plotting import Levi13Plotter
from optimization.events import PopulationLogger, FrameSaverStrategy, FrameSaverConfig


def main() -> None:
    problem = Levi13Problem()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"examples/levi13/output_{timestamp}")
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parameters = [
        ContinuousParameter(
            name="x",
            normalizer=LinearNormalization(-10.0, 10.0),
            description="Levi13 x dimension",
            unit="units",
        ),
        ContinuousParameter(
            name="y",
            normalizer=LinearNormalization(-10.0, 10.0),
            description="Levi13 y dimension",
            unit="units",
        ),
    ]

    population_size = 60
    initializer = RealVectorInitializer(population_size=population_size, parameters=parameters)
    selection = TournamentSelection(k=3)
    crossover = ArithmeticCrossover(prob=0.9)
    mutation = UniformMutation(scale=0.25, prob=0.3, normalized=True)
    elitism = TopKElitism[NDArrayFloat, float](k=1)

    updater = RealVectorGA(
        problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        elitism=elitism,
    )

    convergence = ConvergenceChecker[NDArrayFloat, float](
        strategies=[MaxEvaluationsStop(40_000)],
        iteration_strategy=MaxIterationsStop(120),
    )

    state = OptimizationState[NDArrayFloat, float]()
    state.parameters = parameters
    dispatcher = EventDispatcher[NDArrayFloat, float]()

    pop_logger: PopulationLogger[NDArrayFloat, float] = PopulationLogger(out_dir)
    dispatcher.add_strategy(Stage.RUN_START, pop_logger)
    dispatcher.add_strategy(Stage.ITERATION, pop_logger)
    dispatcher.add_strategy(Stage.RUN_END, pop_logger)

    plotter = Levi13Plotter(update_every=1, interactive=False)
    dispatcher.add_strategy(Stage.RUN_START, plotter)
    dispatcher.add_strategy(Stage.ITERATION, plotter)

    frame_saver = FrameSaverStrategy(plotter, figures_dir, config=FrameSaverConfig(prefix="frame", dpi=120))
    dispatcher.add_strategy(Stage.RUN_START, frame_saver)
    dispatcher.add_strategy(Stage.ITERATION, frame_saver)

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
    print(f"Logs written to: {out_dir}")
    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
