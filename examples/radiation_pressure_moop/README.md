# Multi-Objective Optimization: Radiation vs. Pressure Drop

## Problem Description

This example demonstrates multi-objective optimization for a fluidized bed combustor design problem with two competing objectives:

1. **Maximize radiative efficiency (η)**: Energy radiated from the flame
2. **Minimize pressure drop (ΔP)**: Energy lost to friction and turbulence

These objectives are inherently conflicting - configurations that maximize radiation often increase pressure drop and vice versa. Multi-objective optimization finds the **Pareto front**: the set of non-dominated solutions representing optimal trade-offs between the objectives.

## Decision Variables

The optimization controls 7 design parameters:

| Parameter | Symbol | Description | Bounds | Units |
|-----------|--------|-------------|--------|-------|
| phi | φ | Equivalence ratio | [0.3, 0.9] | - |
| u_avg | $u_\mathrm{avg}$ | Average velocity | [0.5, 2.4] | m/s |
| ar | - | Aspect ratio | [1.1, 5.0] | - |
| lt_0 | - | Lower tube parameter 0 | [0.005, 0.015] | m |
| lt_1 | - | Lower tube parameter 1 | [0.03, 0.08] | m |
| eps_1 | ε₁ | Emissivity parameter 1 | [0.4, 0.95] | - |
| a_1 | α₁ | Absorptivity parameter 1 | [400.0, 800.0] | 1/m |

## Objectives

### Objective 1: Maximize Radiative Efficiency
$$
\max \eta(\mathbf{x})
$$

Radiative efficiency represents the fraction of combustion energy released as thermal radiation. Higher values indicate better heat transfer to the reactor walls.

### Objective 2: Minimize Pressure Drop
$$
\min \Delta P(\mathbf{x})
$$

Pressure drop represents the energy required to pump gases through the reactor. Lower values reduce operating costs and fan power requirements.

## Constraints

The design must satisfy:
- **Flame condition**: α(x) > 0.5 (flame must be stable and established)
- **Box constraints**: All variables within specified bounds

## Mathematical Formulation

This is a **bi-objective constrained optimization problem**:

$$
\begin{align}
\min \quad & \mathbf{f}(\mathbf{x}) = [-\eta(\mathbf{x}), \Delta P(\mathbf{x})] \\
\text{s.t.} \quad & \alpha(\mathbf{x}) > 0.5 \\
& \mathbf{x}_L \leq \mathbf{x} \leq \mathbf{x}_U
\end{align}
$$

Note: η is negated in the objective function to convert maximization to minimization.

## Solution: NSGA-II Algorithm

We use **NSGA-II** (Non-dominated Sorting Genetic Algorithm II), a state-of-the-art multi-objective evolutionary algorithm:

### Key Features
- **Pareto dominance**: Compares solutions based on both objectives simultaneously
- **Non-dominated sorting**: Organizes population into Pareto fronts (layers of optimality)
- **Crowding distance**: Maintains diversity by favoring solutions in sparse regions
- **Elitism**: Preserves best solutions across generations

### Algorithm Configuration

```python
Population size: 100
Max generations: 200
Selection: Binary tournament
Crossover: Arithmetic (p = 0.9)
Mutation: Gaussian (σ = 0.10, p_gene = 0.1)
```

### Genetic Operators
- **Selection**: Binary tournament based on Pareto rank and crowding distance
- **Crossover**: Arithmetic crossover (convex combination of parents)
- **Mutation**: Gaussian mutation with clipping to bounds

## Expected Results

The optimization produces a **Pareto front** in objective space containing non-dominated solutions. Each point on the front represents a different trade-off:

- **High η, High ΔP**: Configurations optimized for radiation (rich combustion, low velocity)
- **Low η, Low ΔP**: Configurations optimized for efficiency (lean combustion, high velocity)
- **Intermediate**: Balanced designs offering moderate performance on both objectives

Decision makers can select from this front based on priorities:
- Energy efficiency priority → High η solution
- Operating cost priority → Low ΔP solution
- Balanced requirements → Intermediate solution

## Running the Optimization

### Single Run

```bash
uv run python run_nsga2.py
```

Results will be saved to `output_YYYYMMDD-HHMMSS/` containing:
- `config.json`: Run configuration
- `pareto_front.json`: Pareto-optimal solutions and objectives
- `statistics.csv`: Evolution statistics per generation
- `checkpoints/`: Population checkpoints

### Visualization

Generate plots for a completed run:

```bash
uv run python plotting.py output_YYYYMMDD-HHMMSS
```

This creates:
- `pareto_front.png`: Objective space visualization
- `parameter_distributions.png`: Decision variable distributions in Pareto set
- `tradeoff_by_*.png`: Pareto front colored by different parameters

## File Structure

```
radiation_pressure_moop/
├── README.md              # This file
├── problem.py            # Problem definition
├── run_nsga2.py          # NSGA-II execution script
├── plotting.py           # Visualization tools
└── output_*/             # Results directories (generated)
    ├── config.json
    ├── pareto_front.json
    ├── statistics.csv
    ├── checkpoints/
    └── figures/
```

## Surrogate Models

The optimization uses symbolic regression models discovered by PySR:

### Radiative Efficiency Model
```
η = tanh(sqrt(ar * (0.033107847 / sqrt(u_avg))) - ...)
```
- Complexity: 34 terms
- Training loss: 1.347188e-04

### Pressure Drop Model
```
ΔP = abs((((19.618101 / (eps_0 ^ ar)) + ...) * phi) * (...))
```
- Complexity: 35 terms
- Training loss: 7.580359e+05

### Flame Classifier Model
```
α = tanh((((phi * phi) / 0.1555563) ^ ar) * ...)
```
- Complexity: 40 terms
- Training loss: 1.570688e-03

All models are located in `exported_models/equations_only/`.

## References

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
- Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. *arXiv preprint arXiv:2305.01582*.
