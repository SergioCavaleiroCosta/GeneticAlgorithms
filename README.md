# Genetic Algorithms & Multi-Objective Optimization

Comprehensive optimization framework with **21 single-objective** and **5 multi-objective** benchmark problems. Features both traditional genetic algorithms and NSGA-II for multi-objective optimization.

**Key Features:**
- **Single-Objective GA**: 21 classic benchmarks (Ackley, Rosenbrock, Rastrigin, etc.)
- **Multi-Objective NSGA-II**: 5 benchmarks (ZDT1-3, DTLZ2, Schaffer N.1) with Pareto front visualization
- **Comprehensive Analysis**: Automated batch execution with statistical summaries and timing metrics
- **Enhanced Visualization**: 16pt font sizes, clean layouts, and dynamic plotting
- **Master Automation Script**: One-command execution of complete analysis pipeline
- **Strictly Typed**: Pylance-clean optimization loop with pluggable components
- **Stage-based Events**: Hooks for visualization, logging, and analysis
- **Performance Tracking**: Built-in execution timing and convergence analysis

## Recent Enhancements

### 🎯 Complete Analysis Pipeline
- **Master Script**: `run_complete_analysis.py` - One command runs entire analysis
- **2,600 Optimization Runs**: 100 runs × 26 benchmark problems  
- **Statistical Analysis**: Comprehensive performance metrics with timing
- **Automated Reporting**: CSV summaries and JSON per-example results

### 📊 Enhanced Visualization  
- **16pt Font Sizes**: Improved readability for all plots and labels
- **Clean Layouts**: Removed titles for publication-ready figures
- **Dynamic Plotting**: Real-time contours and Pareto front evolution
- **Multi-format Export**: PNG figures with high DPI for presentations

### ⚡ Performance Tracking
- **Built-in Timing**: Automatic execution time measurement
- **Convergence Analysis**: Success rates and iteration statistics  
- **Resource Monitoring**: Population sizes and evaluation counts
- **Comparative Metrics**: Cross-benchmark performance analysis

### 🚀 Batch Processing
- **Parallel Execution**: Up to 32 concurrent processes  
- **Fault Tolerance**: Graceful error handling and recovery
- **Progress Monitoring**: Real-time status updates with timing
- **Flexible Configuration**: Customizable runs and concurrency

## Architecture

Core design principles:
- Engine orchestrates initialize → iterate (update) → converge
- Convergence owns the loop condition and tracks iterations internally  
- Population abstraction supports both single- and multi-candidate flows
- Event dispatcher runs strategies per Stage (RUN_START, ITERATION, RUN_END)
- No use of `Any` - fully typed throughout

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
print("Execution time:", result.execution_time, "seconds")
print("Iterations:", result.iterations)
print("Success:", result.success)
```

**Notes:**
- The engine emits a stage signal with `dispatcher.emit(engine, stage)` at RUN_START, for each ITERATION (from its template step), and at RUN_END.
- The `Population` always exists; before initialization it may be empty. After `initialize()`, it contains one or more evaluated candidates.
- Iteration counting and loop control are encapsulated in your `ConvergenceChecker`.
- **Execution timing** is automatically tracked and available in `result.execution_time`.
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

### Additional Single-Objective Examples

**Beale (2D)** - `examples/beale/`: Classic 2D multimodal benchmark with sharp curved valley
**Rosenbrock (10D)** - `examples/rosenbrock/`: Narrow curved valley (ill-conditioned) problem  
**Ackley (10D)** - `examples/ackley/`: Non-separable multimodal benchmark with exponential and cosine terms
**Booth (2D)** - `examples/booth/`: Simple 2D quadratic with global minimum at (1, 3)
**Bukin N.6 (2D)** - `examples/bukin6/`: Highly multimodal with narrow global minimum
**Cross-in-Tray (2D)** - `examples/cross_in_tray/`: Four global minima arranged in cross pattern
**Easom (2D)** - `examples/easom/`: Flat landscape with single sharp global minimum
**Goldstein-Price (2D)** - `examples/goldstein_price/`: Multimodal with several local minima
**Griewank (10D)** - `examples/griewank/`: Product of cosines creates correlation between variables
**Himmelblau (2D)** - `examples/himmelblau/`: Four identical local minima
**Holder Table (2D)** - `examples/holder_table/`: Multiple global minima with oscillatory structure
**Lévi N.13 (2D)** - `examples/levi13/`: Multimodal with global minimum at (1, 1)
**Matyas (2D)** - `examples/matyas/`: Simple valley-shaped function
**McCormick (2D)** - `examples/mccormick/`: Asymmetric bounds and single global minimum
**Schaffer N.2 (2D)** - `examples/schaffer_n2/`: Oscillatory with global minimum at origin
**Schaffer N.4 (2D)** - `examples/schaffer_n4/`: Similar to N.2 with different parameters
**Styblinski-Tang (10D)** - `examples/styblinski_tang/`: Multimodal with many local minima
**Three-Hump Camel (2D)** - `examples/three_hump_camel/`: Three local minima, one global

## Multi-Objective Optimization (NSGA-II)

The framework includes a complete **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** implementation for multi-objective optimization, with five comprehensive benchmark problems:

### Multi-Objective Examples

**ZDT1** - `examples/zdt1/`: Convex Pareto front benchmark
- Bi-objective optimization with convex trade-off curve
- Tests basic multi-objective algorithm performance

**ZDT2** - `examples/zdt2/`: Non-convex Pareto front benchmark  
- Bi-objective with non-convex trade-off curve (f₂ = 1 - f₁²)
- More challenging than ZDT1 for maintaining diversity

**ZDT3** - `examples/zdt3/`: Disconnected Pareto front benchmark
- Bi-objective with multiple disconnected Pareto regions
- Tests algorithm's ability to maintain solutions across separate regions

**DTLZ2** - `examples/dtlz2/`: Scalable multi-objective benchmark
- Configurable number of objectives (2, 3, 4+)  
- Spherical Pareto front (quarter-circle, eighth-sphere, hyper-sphere)
- Tests algorithm scalability to many-objective optimization

**Schaffer N.1** - `examples/schaffer_n1/`: Simple single-variable multi-objective
- f₁(x) = x², f₂(x) = (x-2)²
- Perfect for understanding multi-objective concepts
- Includes both objective space and decision space visualization

### NSGA-II Features

- **Non-dominated Sorting**: Classifies solutions into Pareto fronts
- **Crowding Distance**: Maintains diversity within each front
- **Elite Selection**: Combines parent and offspring populations  
- **Tournament Selection**: Uses both rank and crowding distance
- **Enhanced Visualization**: 16pt fonts, clean layouts, color-coded Pareto ranks
- **True Pareto Front Overlay**: Shows theoretical optimal trade-offs
- **Performance Analysis**: Hypervolume, spacing, and convergence metrics

### Running Individual Examples

```bash
# Single-objective examples
uv run examples/ackley/run_ga.py
uv run examples/rosenbrock/run_ga.py
uv run examples/beale/run_ga.py
# ... (21 total single-objective benchmarks available)

# Multi-objective examples
uv run examples/zdt1/run_nsga_ii.py    # Convex front
uv run examples/zdt2/run_nsga_ii.py    # Non-convex front  
uv run examples/zdt3/run_nsga_ii.py    # Disconnected front
uv run examples/dtlz2/run_nsga_ii.py   # 2D, 3D, or modify for more objectives
uv run examples/schaffer_n1/run_nsga_ii.py  # Single variable, easy to understand
```

### Comprehensive Analysis & Batch Execution

#### Master Automation Script (Recommended)

```bash
# Complete analysis pipeline (100 runs, 22 cores) - ONE COMMAND
uv run python scripts/run_complete_analysis.py

# Quick test run (10 runs, 4 cores)
uv run python scripts/run_complete_analysis.py --runs 10 --concurrency 4

# High-performance run (100 runs, 32 cores)
uv run python scripts/run_complete_analysis.py --runs 100 --concurrency 32

# Preview execution without running
uv run python scripts/run_complete_analysis.py --dry-run

# Skip specific phases
uv run python scripts/run_complete_analysis.py --skip-cleanup --skip-single
```

The master script automatically:
1. **Cleans** existing output directories
2. **Executes** single-objective batch (21 examples × N runs)
3. **Executes** multi-objective batch (5 examples × N runs)  
4. **Generates** comprehensive analysis summaries with timing metrics

#### Manual Batch Execution

```bash
# Clean previous results
rm -r examples/*/output_*

# Run 100 cases for all single-objective examples (2100 total runs)
uv run python scripts/run_batches_multiproc.py -r 100 -c 22

# Run 100 cases for all multi-objective examples (500 total runs)  
uv run python scripts/run_multiobjective_batches.py --runs 100

# Generate analysis summaries after batch runs
uv run python scripts/analyze_all_examples.py           # Single-objective analysis
uv run python scripts/analyze_all_multiobjective.py     # Multi-objective analysis
```

#### Analysis Output

**Generated Files:**
- `scripts/all_examples_summary.csv` - Consolidated benchmark comparison
- `examples/{name}/analysis_summary.json` - Per-example statistical summary  
- `examples/{name}/analysis_runs.csv` - Per-run detailed results

**Key Metrics:**
- **Execution Times**: Min, max, mean, median, std dev, quartiles
- **Best Objectives**: Complete statistical distribution across runs
- **Convergence**: Iterations, evaluations, success rates
- **Population**: Size and diversity metrics
- **Performance**: Timing analysis and resource utilization

