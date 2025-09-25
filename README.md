# Optimization Engine (typed, stage-based)

Strictly typed, Pylance-clean optimization loop with pluggable components and stage-based strategy hooks. No use of `Any`.

Core ideas:
- Engine orchestrates initialize → iterate (update) → converge.
- Convergence owns the loop condition via `should_continue(population)` and tracks iterations internally.
- Evaluation counting lives inside the problem (objective) and is queried by others.
- Population abstraction supports both single- and multi-candidate flows.
- Event dispatcher runs strategies per Stage (RUN_START, ITERATION, RUN_END) via a single execute(engine, stage) method.

## Quickstart

Below is a minimal end-to-end example. It uses a single-candidate flow but the same engine works for population-based methods.

```python
from optimization.population import Population
from optimization.events import EventDispatcher, Stage, OptimizationStageStrategy
from optimization.optimization_engine import OptimizationEngine
from optimization.solution_initializer import SolutionInitializer
from optimization.update_rule import UpdateRule
from optimization.convergence_checker import CompositeConvergenceChecker, MaxIterationsStop
from optimization.optimization_problem import OptimizationProblem
from optimization.state import OptimizationState

# 1) Define your problem (used by the UpdateRule)
class SimpleProblem(OptimizationProblem[float, float]):
	def __init__(self) -> None:
		self._evals = 0

	def evaluate(self, x: float) -> float:
		self._evals += 1
		return (x - 3.0) ** 2  # convex parabola with optimum at x=3

	def is_feasible(self, x: float) -> bool:
		return True

	def generate_random_solution(self) -> float:
		return 0.0

	def get_evaluation_count(self) -> int:
		return self._evals

# 2) Pluggable components
class SimpleInitializer(SolutionInitializer[float, float]):
	def initialize(self, problem: SimpleProblem, updater: UpdateRule[float, float]) -> Population[float, float]:
		x0 = 0.0
		f0 = problem.evaluate(x0)
		updater.seed(x0, f0)  # seed updater with the initial state
		return Population.from_single(x0, f0)

class GradientLikeUpdate(UpdateRule[float, float]):
	def __init__(self, problem: SimpleProblem, step: float = 0.1) -> None:
		self._problem = problem
		self._step = step
		self._x: float | None = None

	def set_dispatcher(self, dispatcher: EventDispatcher[float, float]) -> None:
		self._dispatcher = dispatcher  # optional, unused here

	@property
	def problem(self) -> SimpleProblem:
		return self._problem

	def seed(self, initial_solution: float, initial_objective: float) -> None:
		self._x = initial_solution

	def step(self, engine: OptimizationEngine[float, float]) -> None:
		assert self._x is not None, "Updater must be seeded before stepping"
		eps = 1e-6
		f = self._problem.evaluate(self._x)
		g = (self._problem.evaluate(self._x + eps) - f) / eps
		x_new = self._x - self._step * g
		f_new = self._problem.evaluate(x_new)
		self._x = x_new
		engine.population.update_single(x_new, f_new)

# 3) Optional: a single strategy handling all stages
class PrintObserver(OptimizationStageStrategy[float, float]):
	def execute(self, engine: OptimizationEngine[float, float], stage: Stage) -> None:
		pop = engine.population
		it = engine._convergence.iteration  # iteration is owned by convergence
		evals = engine._updater.problem.get_evaluation_count()
		if stage is Stage.RUN_START:
			print(f"Run started: x={pop.current_solution:.3f}, f={pop.current_objective:.6f}, evals={evals}")
		elif stage is Stage.ITERATION:
			print(
				f"it={it:03d} x={pop.current_solution:.4f} f={pop.current_objective:.6f} "
				f"best={pop.best_objective:.6f} evals={evals}"
			)
		elif stage is Stage.RUN_END:
			print(
				f"done in {it} iters, best x={pop.best_solution:.3f}, "
				f"f={pop.best_objective:.6f}, evals={evals}"
			)

# 4) Wire everything and run
problem = SimpleProblem()
updater = GradientLikeUpdate(problem)
initializer = SimpleInitializer()
# Convergence as a composition of simple stop strategies
convergence = CompositeConvergenceChecker[float, float]([
	MaxIterationsStop(30),
])
dispatcher: EventDispatcher[float, float] = EventDispatcher()
dispatcher.add_strategy(Stage.RUN_START, PrintObserver())
dispatcher.add_strategy(Stage.ITERATION, PrintObserver())
dispatcher.add_strategy(Stage.RUN_END, PrintObserver())

state: OptimizationState[float, float] = OptimizationState()
engine = OptimizationEngine(initializer, updater, convergence, state, dispatcher)
result = engine.run()
print("Result:", result.best_solution, result.best_objective)
```

Notes:
- The engine emits a stage signal with `dispatcher.emit(engine, stage)` at RUN_START, for each ITERATION (from its template step), and at RUN_END.
- The `Population` always exists; before initialization it may be empty. After `initialize()`, it contains one or more evaluated candidates.
- Iteration counting and loop control are encapsulated in your `ConvergenceChecker`.
- Evaluation counting is owned by the problem (queried via `problem.get_evaluation_count()`).

## Included Examples

### Eggholder (2D)
Located in `examples/eggholder/`. Demonstrates:
- External parameter specification (`ContinuousParameter`) with real-domain normalization
- Real-time contour plotting with dynamic best-solution driven slicing (only 2D so direct)
- Advanced arithmetic crossover (random subset multi-allele blending)

Run:
```bash
uv run python examples/eggholder/run_ga.py
```

### Sphere (10D)
Located in `examples/sphere/`. Demonstrates:
- High-dimensional continuous optimization with 10 parameters in [-5.12, 5.12]
- 2D contour projection of the first two dimensions while fixing others to the current best solution
- Same GA pipeline components reused (selection, crossover, mutation, elitism)

Run:
```bash
uv run python examples/sphere/run_ga.py
```

Both examples denormalize the final best candidate for reporting and showcase the stage-based event system for live visualization.

### Rastrigin (10D)
Located in `examples/rastrigin/`. Demonstrates:
- Multimodal landscape (many local minima) stressing exploration
- Dynamic best-based slicing for 2D contour (first two dimensions by default)
- Slightly larger population and run length to sample rugged landscape

Run:
```bash
uv run python examples/rastrigin/run_ga.py
```

All examples share the same core engine and operator abstractions, highlighting reuse.

