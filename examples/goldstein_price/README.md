# Goldstein-Price Function Optimization Example

This example demonstrates genetic algorithm optimization on the Goldstein-Price function, a complex 2D polynomial benchmark with multiple local minima.

## Problem Description

The Goldstein-Price function is defined as a product of two polynomials:

```
f(x, y) = [1 + (x + y + 1)² * (19 - 14x + 3x² - 14y + 6xy + 3y²)] *
          [30 + (2x - 3y)² * (18 - 32x + 12x² + 48y - 36xy + 27y²)]
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-2, 2]
- **Global minimum**: (x*, y*) = (0, -1) with f(x*, y*) = 3
- **Characteristics**: Complex polynomial landscape with several local minima

## Key Features

- **Polynomial complexity**: High-degree polynomial creating intricate landscape
- **Multiple local minima**: Tests algorithm's ability to escape local optima
- **Steep gradients**: Sharp variations in function values
- **Classic benchmark**: Well-studied problem in optimization literature

## Running the Example

```bash
uv run examples/goldstein_price/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the polynomial landscape
  - Final optimization results

## Files

- `problem.py`: Goldstein-Price function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation