# DTLZ2 Multi-Objective Benchmark

This example demonstrates the NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation on the DTLZ2 multi-objective benchmark problem.

## Problem

The DTLZ2 problem is a **scalable multi-objective** benchmark with a **spherical Pareto front**:

**For M objectives:**
- **f₁(x)** = (1 + g(x_M)) × cos(x₁π/2) × cos(x₂π/2) × ... × cos(x_{M-1}π/2)
- **f₂(x)** = (1 + g(x_M)) × cos(x₁π/2) × cos(x₂π/2) × ... × sin(x_{M-1}π/2)
- **...**
- **f_M(x)** = (1 + g(x_M)) × sin(x₁π/2)
- **g(x_M)** = Σ(xᵢ - 0.5)² for i = M to n

**Domain**: x_i ∈ [0, 1] for all dimensions i = 1, ..., n

**Properties**:
- **Spherical Pareto front** (quarter-circle for 2D, eighth-sphere for 3D)
- **Scalable** to any number of objectives M ≥ 2
- Recommended: n = M + 10 variables
- Global optimum: x₁,...,x_{M-1} ∈ [0,1], x_M,...,x_n = 0.5

## Algorithm Features

The NSGA-II implementation includes:

1. **Non-dominated Sorting**: Classifies solutions into Pareto fronts
2. **Crowding Distance**: Maintains diversity within each front  
3. **Elite Selection**: Combines parent and offspring populations
4. **Multi-objective Selection**: Tournament selection using rank and crowding distance

## Running the Example

```bash
uv run examples/dtlz2/run_nsga_ii.py
```

**Configuring Objectives**: Edit `run_nsga_ii.py` to change:
```python
num_objectives = 2  # Try 2, 3, 4, or more
```

## Output

The example generates:
- Population objectives over time (`objectives.csv`)
- Solution vectors over time (`solutions.csv`) 
- Pareto front plots:
  - **2D**: Scatter plot with true quarter-circle front
  - **3D**: 3D scatter plot with true eighth-sphere front
  - **4D+**: Pairwise objective projections

## Key Characteristics

- **ZDT1-3**: Fixed to 2 objectives, various front shapes
- **DTLZ2**: **Scalable** to M objectives, spherical front

**Front Shapes**:
- 2 objectives: Quarter circle
- 3 objectives: Eighth of sphere  
- M objectives: Section of M-dimensional hyper-sphere

DTLZ2 is ideal for testing algorithm scalability to many-objective optimization (4+ objectives).

## Files

- `problems.py`: DTLZ2 multi-objective test problem implementation
- `plotting.py`: Visualization utilities (2D, 3D, and high-dimensional)
- `run_nsga_ii.py`: Main execution script (configurable objectives)
- `README.md`: This documentation