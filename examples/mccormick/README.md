# McCormick Function Optimization Example

This example demonstrates genetic algorithm optimization on the McCormick function, a 2D benchmark with asymmetric domain and trigonometric components.

## Problem Description

The McCormick function is defined as:

```
f(x, y) = sin(x + y) + (x - y)² - 1.5*x + 2.5*y + 1
```

Where:
- **Dimensions**: 2D
- **Domain**: x ∈ [-1.5, 4], y ∈ [-3, 4] (asymmetric domain)
- **Global minimum**: (x*, y*) ≈ (-0.54719, -1.54719) with f(x*, y*) ≈ -1.9133
- **Characteristics**: Mixed trigonometric and polynomial terms with asymmetric domain

## Key Features

- **Asymmetric domain**: Different bounds for x and y coordinates
- **Mixed function types**: Combination of trigonometric and polynomial terms
- **Moderate complexity**: Single global minimum with some local variation
- **Irregular landscape**: Non-uniform contour patterns

## Running the Example

```bash
uv run examples/mccormick/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the mixed landscape
  - Final optimization results

## Files

- `problem.py`: McCormick function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation