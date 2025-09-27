# Eggholder Function Optimization Example

This example demonstrates genetic algorithm optimization on the Eggholder function, one of the most challenging benchmark functions with many local minima and an irregular landscape.

## Problem Description

The Eggholder function is defined as:

```
f(x, y) = -(y + 47) * sin(sqrt(|x/2 + (y + 47)|)) - x * sin(sqrt(|x - (y + 47)|))
```

Where:
- **Dimensions**: 2D
- **Domain**: x,y ∈ [-512, 512]
- **Global minimum**: (x*, y*) = (512, 404.2319) with f(x*, y*) ≈ -959.6407
- **Characteristics**: Extremely irregular landscape with hundreds of local minima

## Key Features

- **Most challenging benchmark**: Known as one of the hardest 2D optimization problems
- **Irregular landscape**: Egg-carton like surface with many peaks and valleys
- **Advanced logging**: Includes specialized logging strategies for detailed analysis
- **Comprehensive visualization**: Detailed contour plots showing the complex landscape

## Running the Example

```bash
uv run examples/eggholder/run_ga.py
```

## Output Files

- `analysis_runs.csv`: Statistical summary of multiple runs
- `analysis_summary.json`: Detailed analysis results
- `output_YYYYMMDD-HHMMSS/`: Timestamped run directory containing:
  - Population data CSV files with detailed logging
  - 2D contour plots showing the irregular landscape
  - Final optimization results

## Files

- `problem.py`: Eggholder function implementation and problem class
- `plotting.py`: Specialized 2D contour visualization utilities
- `logging_strategies.py`: Advanced logging strategies for detailed analysis
- `run_ga.py`: Main execution script
- `README.md`: This documentation