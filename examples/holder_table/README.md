# Holder Table Function Optimization Example

This example demonstrates genetic algorithm optimization on the Holder Table function, a multimodal 2D benchmark with four global minima and a table-like structure.

## Problem Description

The Holder Table function is defined as:

```
f(x, y) = -|sin(x)*cos(y)*exp(|1 - sqrt(x² + y²)/π|)|
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-10, 10]
- **Global minima**: Four symmetric points at (±8.05502, ±9.66459) with f ≈ -19.2085
- **Characteristics**: Table-like structure with four legs representing global minima

## Key Features

- **Table structure**: Distinctive flat top with four "legs" as global minima
- **Multiple global minima**: Four equivalent optimal points
- **Trigonometric complexity**: Combination of sin, cos, and exponential functions
- **Symmetric landscape**: Four-fold rotational symmetry

## Running the Example

```bash
uv run examples/holder_table/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the table structure
  - Final optimization results

## Files

- `problem.py`: Holder Table function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation