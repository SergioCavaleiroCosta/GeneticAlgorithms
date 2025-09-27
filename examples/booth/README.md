# Booth Function Optimization Example

This example demonstrates genetic algorithm optimization on the Booth function, a simple 2D quadratic benchmark problem.

## Problem Description

The Booth function is defined as:

```
f(x, y) = (x + 2y - 7)² + (2x + y - 5)²
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-10, 10]
- **Global minimum**: (x*, y*) = (1, 3) with f(x*, y*) = 0
- **Characteristics**: Convex quadratic function with a single global minimum

## Key Features

- **Simple quadratic landscape**: Ideal for testing basic GA convergence
- **2D contour visualization**: Clear visualization of the quadratic bowl shape
- **Fast convergence**: Algorithm should quickly find the global minimum
- **Analytical solution**: Easy to verify algorithm performance

## Running the Example

```bash
uv run examples/booth/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing convergence
  - Final optimization results

## Files

- `problem.py`: Booth function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation