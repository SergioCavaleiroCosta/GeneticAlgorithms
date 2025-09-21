# Optimization Engine (typed, event-driven)

Strictly typed, Pylance-clean optimization loop with pluggable components and stage-based events. No use of `Any`.

Core ideas:
- Engine orchestrates initialize → iterate (update) → converge.
- Convergence owns the loop condition via `should_continue(population)`.
- Evaluation counting lives inside the problem (objective) and is queried by others.
- Population abstraction supports both single- and multi-candidate flows.
- Event dispatcher emits stage-scoped contexts: `run_start`, `iteration`, `run_end` and can run registered strategies for each stage.

## Quickstart

Below is a minimal end-to-end example. It uses a single-candidate flow but the same engine works for population-based methods.

```python
from optimization.types import ST, OT
from optimization.population import Population
from optimization.events import EventDispatcher, IterationStrategy, RunStartStrategy, RunEndStrategy
from optimization.optimization_engine import OptimizationEngine
from optimization.solution_initializer import SolutionInitializer
from optimization.update_rule import UpdateRule
from optimization.convergence_checker import ConvergenceChecker
from optimization.optimization_result import OptimizationResult

# 1) Define your problem (must be provided/owned by the UpdateRule)
class SimpleProblem:
	def __init__(self) -> None:
		self._evals = 0

	def evaluate(self, x: float) -> float:
		self._evals += 1
		return (x - 3.0) ** 2  # convex parabola with optimum at x=3

	def get_evaluation_count(self) -> int:
		return self._evals

# 2) Pluggable components
class SimpleInitializer(SolutionInitializer[float, float]):
	def initialize(self, problem: SimpleProblem, updater: UpdateRule[float, float]) -> Population[float, float]:
		x0 = 0.0
		f0 = problem.evaluate(x0)
		return Population.from_single(x0, f0)

class GradientLikeUpdate(UpdateRule[float, float]):
	def __init__(self, problem: SimpleProblem, step: float = 0.1) -> None:
		self._problem = problem
		self._step = step

	@property
	def problem(self) -> SimpleProblem:
		return self._problem

	def set_dispatcher(self, dispatcher: EventDispatcher[float, float]) -> None:
		# Optional: capture for custom side-effects; unused here
		self._dispatcher = dispatcher

	def step(self, convergence: ConvergenceChecker[float, float]) -> tuple[float, float]:
		# finite-difference gradient
		eps = 1e-6
		# current point is last of the population
		x = convergence.last_state.current_solution  # provided by engine via population
		f = self._problem.evaluate(x)
		g = (self._problem.evaluate(x + eps) - f) / eps
		x_new = x - self._step * g
		f_new = self._problem.evaluate(x_new)
		return x_new, f_new

class MaxSteps(ConvergenceChecker[float, float]):
	def __init__(self, max_iter: int = 50) -> None:
		self._max_iter = max_iter
		self.iteration = 0
		self.last_state = None  # engine keeps population/state; used here for illustration

	def set_dispatcher(self, dispatcher: EventDispatcher[float, float]) -> None:
		self._dispatcher = dispatcher

	def reset(self) -> None:
		self.iteration = 0

	def should_continue(self, population: Population[float, float]) -> bool:
		self.iteration += 1
		# keep a handle if needed by the updater
		class _Wrapper:
			def __init__(self, pop: Population[float, float]):
				self._pop = pop
			@property
			def current_solution(self) -> float:
				return self._pop.current_solution
		self.last_state = _Wrapper(population)
		return self.iteration <= self._max_iter

	def on_run_completed(self, **kwargs) -> None:
		pass

# 3) Optional: Strategies to react to stages
class PrintStart(RunStartStrategy[float, float]):
	def on_run_start(self, initial_solution: float, initial_objective: float, evaluations: int, elapsed: float) -> None:
		print(f"Run started at x={initial_solution:.3f}, f={initial_objective:.3f}, evals={evaluations}")

class PrintIter(IterationStrategy[float, float]):
	def on_iteration(self, iteration: int, elapsed: float, current_solution: float, current_objective: float, best_solution: float, best_objective: float, evaluations: int) -> None:
		print(f"it={iteration:03d} x={current_solution:.4f} f={current_objective:.6f} best={best_objective:.6f} evals={evaluations}")

class PrintEnd(RunEndStrategy[float, float]):
	def on_run_end(self, iterations: int, elapsed: float, best_solution: float, best_objective: float, evaluations: int, success: bool, termination_reason: str) -> None:
		print(f"done in {iterations} iters, best x={best_solution:.3f}, f={best_objective:.6f}, evals={evaluations}")

# 4) Wire everything and run
problem = SimpleProblem()
updater = GradientLikeUpdate(problem)
initializer = SimpleInitializer()
convergence = MaxSteps(30)
dispatcher: EventDispatcher[float, float] = EventDispatcher()
dispatcher.add_run_start_strategy(PrintStart())
dispatcher.add_iteration_strategy(PrintIter())
dispatcher.add_run_end_strategy(PrintEnd())

from optimization.state import OptimizationState
state: OptimizationState[float, float] = OptimizationState()

engine = OptimizationEngine(initializer, updater, convergence, state, dispatcher)
result: OptimizationResult[float, float] = engine.run()
print("Result:", result.best_solution, result.best_objective)
```

Notes:
- The engine emits stage events using `dispatcher.emit_run_start`, `emit_iteration`, and `emit_run_end` so your strategies always run, and context events are emitted to listeners as well.
- The `Population` always exists; before initialization it may be empty. After `initialize()`, it contains one or more evaluated candidates.
- Iteration counting and loop control are encapsulated in your `ConvergenceChecker`.
- Evaluation counting is owned by the problem (queried via `problem.get_evaluation_count()`).

