# ZDT2 Multi-Objective Benchmark

This example demonstrates the NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation on the ZDT2 multi-objective benchmark problem.

## Problem

The ZDT2 problem is a bi-objective benchmark with a **non-convex Pareto front**:

- **f1(x)** = x₁
- **f2(x)** = g(x) × (1 - (x₁/g(x))²)
- **g(x)** = 1 + 9 × sum(x₂...x_m) / (m-1)

**Domain**: x_i ∈ [0, 1] for all dimensions i = 1, ..., m

**Properties**:
- Non-convex Pareto front (f₂ = 1 - f₁²)
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
uv run examples/zdt2/run_nsga_ii.py
```

## Output

The example generates:
- Population objectives over time (`objectives.csv`)
- Solution vectors over time (`solutions.csv`) 
- Pareto front plots comparing evolved solutions to true front
- Final Pareto front visualization

## Key Differences from ZDT1

- **ZDT1**: Convex Pareto front (f₂ = 1 - √f₁)
- **ZDT2**: **Non-convex** Pareto front (f₂ = 1 - f₁²)

The non-convex nature of ZDT2 makes it more challenging for algorithms to maintain diversity across the entire front.

## Files

- `problems.py`: ZDT2 multi-objective test problem implementation
- `plotting.py`: Visualization utilities with color-coded Pareto ranks
- `run_nsga_ii.py`: Main execution script
- `README.md`: This documentation