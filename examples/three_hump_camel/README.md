# Three-Hump Camel Function Optimization Example

This example demonstrates genetic algorithm optimization on the Three-Hump Camel function, a 2D benchmark with three local minima arranged in a camel-hump pattern.

## Problem Description

The Three-Hump Camel function is defined as:

```
f(x, y) = 2*x² - 1.05*x^4 + x^6/6 + x*y + y²
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-5, 5]
- **Global minimum**: (x*, y*) = (0, 0) with f(x*, y*) = 0
- **Local minima**: Two additional local minima creating the "three humps"
- **Characteristics**: Three distinct basins with varying depths

## Key Features

- **Three-basin structure**: Distinctive camel-hump landscape with three minima
- **Polynomial complexity**: Sixth-order polynomial in x creates interesting topology
- **Moderate difficulty**: Not too many local minima, good for testing local search escape
- **Clear structure**: Easy to visualize and understand algorithm behavior

## Running the Example

```bash
uv run examples/three_hump_camel/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the three-hump structure
  - Final optimization results

## Files

- `problem.py`: Three-Hump Camel function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation