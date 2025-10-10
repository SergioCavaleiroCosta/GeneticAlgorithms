# Radiative Efficiency Optimization

## Problem Description

This example optimizes the **radiative efficiency** (η_rad) in a fluidized bed combustion system## Expected Results

Since this uses a trained PySR model on real-world data with physical constraints, the optimization should:

1. **Find flame conditions**: Discover parameter combinations where α > 0.5
2. **Maximize efficiency**: Among valid flame conditions, find maximum η_rad
3. **Converge**: Within 200-400 iterations
4. **Respect bounds**: All parameter values within specified ranges
5. **Physical validity**: Solutions should represent realistic combustion conditions

### Interesting Behavior

- **Discrete transition**: Sharp boundary at α = 0.5 where efficiency drops to 0
- **Multi-objective nature**: Implicitly balancing flame stability (α) and efficiency (η_rad)
- **Parameter coupling**: phi, u_avg, ar affect both flame presence and efficiencyymbolic regression models trained with PySR (Python Symbolic Regression). 

**Critical Feature**: The optimization uses a **two-stage evaluation**:
1. **Flame Classifier (α)**: Determines if combustion conditions support a flame
2. **Radiative Efficiency**: Only computed when α > 0.5 (flame present)

This mimics real-world physics where radiative efficiency is meaningful only when combustion is occurring.

## PySR Models

### 1. Flame Classifier (α)

- **Model ID**: `20251010_155720_BXNBv9`
- **Model Type**: `flame_classifier`
- **Target Variable**: `alpha`
- **Complexity**: 30
- **Loss**: 3.469 × 10⁻²
- **Parameters**: phi, u_avg, ar
- **Purpose**: Determines flame presence (α > 0.5 = flame exists)

### 2. Radiative Efficiency (η_rad)

- **Model ID**: `20251010_162508_k1g9rP`
- **Model Type**: `flame_surrogate`
- **Target Variable**: `radiative_efficiency`
- **Complexity**: 29
- **Loss**: 2.615 × 10⁻⁴ (excellent accuracy!)
- **Parameters**: phi, u_avg, ar, lt_1, a_0
- **Purpose**: Computes efficiency when flame is present
- **Equation**: 
```
(exp(((-0.21451901 - sqrt(u_avg / ar)) / (phi * 0.7542987)) - 
     ((phi * ((phi + -0.6582989) / u_avg)) * 1.4569359)) * 1.2530051) + 
(lt_1 * (a_0 * 0.0002242238))
```

### Evaluation Logic

```python
1. Compute α = flame_classifier(phi, u_avg, ar)
2. If α ≤ 0.5:
      return 0  # No flame, no radiative efficiency
3. If α > 0.5:
      return radiative_efficiency(phi, u_avg, ar, lt_1, a_0)
```

### Physical Context

The radiative efficiency is a critical parameter in fluidized bed combustion, representing the fraction of thermal energy transferred through radiation. However, this is only meaningful when combustion is occurring (flame present). The flame classifier ensures we only optimize valid combustion conditions.

## Optimization Problem

**Objective**: Maximize radiative efficiency (η_rad)

Since genetic algorithms minimize by default, we negate the model output to convert maximization into minimization.

### Parameters and Bounds

| Parameter | Symbol | Description | Unit | Min | Max |
|-----------|--------|-------------|------|-----|-----|
| **phi** | φ | Equivalence ratio | - | 0.3 | 0.9 |
| **u_avg** | u_avg | Average inlet velocity | m/s | 0.5 | 2.4 |
| **ar** | ar | Aspect ratio | - | 1.1 | 5.0 |
| **lt_1** | lt_1 | Second layer thickness | m | 0.005 | 0.015 |
| **a_0** | a_0 | First layer extinction coefficient | 1/m | 1000.0 | 1300.0 |

### Problem Characteristics

- **Dimensionality**: 5 (all continuous parameters)
- **Objective Type**: Single-objective (minimize negative radiative efficiency)
- **Known Optimal**: Unknown (real-world problem from trained model)
- **Constraints**: 
  - Box constraints (parameter bounds)
  - Implicit constraint: α > 0.5 (flame must be present)
- **Model Type**: Two-stage evaluation with flame classifier
- **Model Accuracy**: Loss = 2.615 × 10⁻⁴ (very low, indicating high accuracy)
- **Physical Realism**: Returns 0 when no flame exists (α ≤ 0.5)

## Running the Example

### Single Run

```powershell
# From project root
uv run python examples/radiative_efficiency/run_ga.py
```

### Batch Runs (100 experiments)

```powershell
# Using the batch script
uv run python scripts/run_batches_simple.py -e radiative_efficiency -r 100
```

## Algorithm Configuration

The genetic algorithm is configured with:

- **Population Size**: 100
- **Selection**: Tournament selection (k=3)
- **Crossover**: Arithmetic crossover (prob=0.9)
- **Mutation**: Uniform mutation (scale=0.25, prob=0.3)
- **Elitism**: Top-5 individuals preserved
- **Max Iterations**: 500
- **Max Evaluations**: 50,000

## Output

Each run creates a timestamped output directory:

```
examples/radiative_efficiency/output_YYYYMMDD-HHMMSS/
├── data/
│   ├── iter_000.csv    # Population at iteration 0
│   ├── iter_001.csv    # Population at iteration 1
│   └── ...
├── figures/
│   ├── frame_000.png   # Visualization frames
│   └── ...
└── result.csv          # Final optimization results
```

### result.csv Format

```csv
best_objective,execution_time,iterations,evaluations,population_size
-45.123456,120.5,250,25000,100
```

Note: `best_objective` is negative (we minimized `-radiative_efficiency`), so negate it to get actual efficiency.

## Visualization

The plotter shows 2D contour slices of the 5D objective function:

- **Default View**: Average Velocity (`u_avg`) vs Equivalence Ratio (`phi`)
- **Fixed Dimensions**: Other parameters fixed at current best solution values
- **Population**: Current population overlaid on contour plot
- **Updates**: Every 25 iterations

## Analysis

After running batch experiments, analyze the results:

```powershell
# Analyze this example
uv run python scripts/analyze_example.py radiative_efficiency

# Analyze all examples
uv run python scripts/analyze_all_examples.py
```

This generates:
- `analysis_summary.json` - Statistical summary of all runs
- `analysis_runs.csv` - Per-run details
- Consolidated summary in `scripts/all_examples_summary.csv`

## Expected Results

Since this is a trained PySR model on real-world data, the optimal radiative efficiency depends on the training data range. Typical optimization should:

1. **Converge** within 200-400 iterations
2. **Explore** the 7D parameter space effectively
3. **Find** parameter combinations that maximize efficiency
4. **Respect** all parameter bounds

## References

- **PySR**: [github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)
- **Symbolic Regression**: Automated discovery of mathematical equations from data
- **Fluidized Bed Combustion**: Multi-parameter thermal process optimization

## Notes

- **Two-stage evaluation**: First checks flame classifier, then computes efficiency
- **Conditional logic**: Returns 0 efficiency when α ≤ 0.5 (no flame)
- **Numerical sensitivities**: sqrt, exp, and division operations require careful handling
- **Large penalty terms**: Prevent division by zero and constraint violations
- **5D visualization challenge**: We use 2D slices for visualization
- **Multiple runs recommended**: Assess solution quality and consistency
- **Physical validity**: The flame classifier ensures realistic combustion conditions
- **Model accuracy**: Very low loss (2.615 × 10⁻⁴) indicates high predictive accuracy
- **Implicit constraint**: The α > 0.5 condition acts as a feasibility constraint
