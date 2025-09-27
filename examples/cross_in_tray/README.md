# Cross-in-Tray Function Optimization Example

This example demonstrates genetic algorithm optimization on the Cross-in-Tray function, a highly multimodal 2D benchmark with four global minima.

## Problem Description

The Cross-in-Tray function is defined as:

```
f(x, y) = -0.0001 * (|sin(x)*sin(y)*exp(|100 - sqrt(x² + y²)/π|)| + 1)^0.1
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-10, 10]
- **Global minima**: Four symmetric points at (±1.3491, ±1.3491) with f ≈ -2.06261
- **Characteristics**: Highly multimodal with cross-shaped structure and four identical global minima

## Key Features

- **Multiple global minima**: Algorithm can find any of the four equivalent optima
- **Cross-shaped landscape**: Distinctive pattern with radial symmetry
- **High multimodality**: Many local minima test exploration capability
- **Symmetric structure**: Demonstrates algorithm behavior on symmetric landscapes

## Running the Example

```bash
uv run examples/cross_in_tray/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files
  - 2D contour plots showing the cross-shaped landscape
  - Final optimization results

## Files

- `problem.py`: Cross-in-Tray function implementation and problem class
- `plotting.py`: 2D contour visualization utilities
- `run_ga.py`: Main execution script
- `README.md`: This documentation