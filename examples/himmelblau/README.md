# Himmelblau Function Optimization Example

This example demonstrates genetic algorithm optimization on the Himmelblau function, a 2D polynomial benchmark with four global minima.

## Problem Description

The Himmelblau function is defined as:

```
f(x, y) = (x² + y - 11)² + (x + y² - 7)²
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-5, 5]
- **Global minima**: Four points with f = 0:
  - (3.0, 2.0)
  - (-2.805118, 3.131312)
  - (-3.779310, -3.283186)
  - (3.584428, -1.848126)
- **Characteristics**: Four equivalent global minima, polynomial landscape

## Key Features

- **Multiple global minima**: Four distinct global optima of equal value
- **Polynomial structure**: Smooth but complex landscape
- **Symmetric properties**: Interesting mathematical structure
- **Benchmark classic**: Widely used in optimization testing

## Running the Example

```bash
uv run examples/himmelblau/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing all four global minima
  - Final optimization results

## Files

- `problem.py`: Himmelblau function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation