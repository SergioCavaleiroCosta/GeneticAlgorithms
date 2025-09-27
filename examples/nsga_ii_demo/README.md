"""Multi-Objective NSGA-II Example

This example demonstrates the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
implementation for multi-objective optimization.

## Problem

The example uses the ZDT1 test problem, a standard bi-objective benchmark:

- f1(x) = x1
- f2(x) = g(x) * (1 - sqrt(x1/g(x)))
- g(x) = 1 + 9 * sum(x2...xm) / (m-1)

Where x_i ∈ [0, 1] for all dimensions.

## Algorithm Features

The NSGA-II implementation includes:

1. **Non-dominated Sorting**: Classifies solutions into Pareto fronts
2. **Crowding Distance**: Maintains diversity within each front
3. **Elite Selection**: Combines parent and offspring populations
4. **Multi-objective Selection**: Tournament selection using rank and crowding distance

## Running the Example

```bash
python examples/nsga_ii_demo/run_nsga_ii.py
```

## Output

The example generates:
- Population objectives over time (objectives.csv)
- Solution vectors over time (solutions.csv) 
- Pareto front plots comparing evolved solutions to true front
- Final Pareto front visualization

## Files

- `problems.py`: ZDT1 and ZDT2 multi-objective test problems
- `run_nsga_ii.py`: Main execution script
- `README.md`: This documentation