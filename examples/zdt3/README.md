# ZDT3 Multi-Objective Benchmark

This example demonstrates the NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation on the ZDT3 multi-objective benchmark problem.

## Problem

The ZDT3 problem is a bi-objective benchmark with a **disconnected Pareto front**:

- **f1(x)** = x₁
- **f2(x)** = g(x) × (1 - √(x₁/g(x)) - (x₁/g(x)) × sin(10πx₁))
- **g(x)** = 1 + 9 × sum(x₂...x_m) / (m-1)

**Domain**: x_i ∈ [0, 1] for all dimensions i = 1, ..., m

**Properties**:
- **Disconnected Pareto front** with several gaps/regions
- Tests algorithm's ability to maintain diversity across separate regions
- 10 variables typically used
- Global optimum at x₁ ∈ [0,1], x₂ = x₃ = ... = x_m = 0

## Algorithm Features

The NSGA-II implementation includes:

1. **Non-dominated Sorting**: Classifies solutions into Pareto fronts
2. **Crowding Distance**: Maintains diversity within each front  
3. **Elite Selection**: Combines parent and offspring populations
4. **Multi-objective Selection**: Tournament selection using rank and crowding distance

## Running the Example

```bash
uv run examples/zdt3/run_nsga_ii.py
```

## Output

The example generates:
- Population objectives over time (`objectives.csv`)
- Solution vectors over time (`solutions.csv`) 
- Pareto front plots comparing evolved solutions to true front
- Final Pareto front visualization

## Key Characteristics

- **ZDT1**: Convex Pareto front
- **ZDT2**: Non-convex Pareto front  
- **ZDT3**: **Disconnected** Pareto front with gaps

The disconnected nature of ZDT3 makes it particularly challenging for multi-objective algorithms, as they must:
- Discover all disconnected regions
- Maintain solutions across separate regions
- Avoid converging to only one region

## Files

- `problems.py`: ZDT3 multi-objective test problem implementation
- `plotting.py`: Visualization utilities with color-coded Pareto ranks
- `run_nsga_ii.py`: Main execution script
- `README.md`: This documentation