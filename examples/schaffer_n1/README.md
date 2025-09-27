# Schaffer N.1 Multi-Objective Benchmark

This example demonstrates the NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation on the Schaffer N.1 multi-objective benchmark problem.

## Problem

The Schaffer N.1 problem is the **simplest multi-objective benchmark** with a **single decision variable**:

- **f₁(x)** = x²
- **f₂(x)** = (x - 2)²

**Domain**: x ∈ [-A, A] where A is typically 10³ (we use A=10 for easier visualization)

**Properties**:
- **Single decision variable** (1D)
- **Convex Pareto front**
- Easy to visualize and understand
- **Pareto optimal region**: x ∈ [0, 2]
- Global trade-offs clearly visible

## Algorithm Features

The NSGA-II implementation includes:

1. **Non-dominated Sorting**: Classifies solutions into Pareto fronts
2. **Crowding Distance**: Maintains diversity within each front  
3. **Elite Selection**: Combines parent and offspring populations
4. **Multi-objective Selection**: Tournament selection using rank and crowding distance

## Running the Example

```bash
uv run examples/schaffer_n1/run_nsga_ii.py
```

## Output

The example generates:
- Population objectives over time (`objectives.csv`)
- Solution x values over time (`solutions.csv`) 
- **Objective space plots**: Pareto front evolution with color-coded ranks
- **Decision space plots**: Population distribution showing convergence to [0, 2]

## Key Insights

This simple problem illustrates fundamental multi-objective concepts:

**Trade-offs**:
- x = 0: f₁ = 0, f₂ = 4 (minimize f₁)
- x = 2: f₁ = 4, f₂ = 0 (minimize f₂)  
- x ∈ (0, 2): Balanced trade-offs

**Pareto Optimality**:
- Any x ∈ [0, 2] is Pareto optimal
- Solutions outside [0, 2] are dominated

**Algorithm Behavior**:
- Population should converge to region [0, 2]
- Diversity maintenance keeps solutions spread across trade-off curve

## Comparison with Other Benchmarks

- **Schaffer N.1**: Single variable, convex front
- **ZDT1**: Multi-variable, convex front
- **ZDT2**: Multi-variable, non-convex front
- **ZDT3**: Multi-variable, disconnected front
- **DTLZ2**: Scalable objectives, spherical front

Schaffer N.1 is ideal for **understanding** multi-objective optimization concepts before tackling complex problems.

## Files

- `problems.py`: Schaffer N.1 multi-objective test problem implementation
- `plotting.py`: Visualization utilities (objective + decision space)
- `run_nsga_ii.py`: Main execution script
- `README.md`: This documentation