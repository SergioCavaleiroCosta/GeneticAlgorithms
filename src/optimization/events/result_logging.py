"""Result logging strategies for capturing final optimization outcomes.

Provides a ResultLogger that captures final optimization results including
execution time, iterations, population size, and best objective values to
CSV files for statistical analysis across multiple runs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import OptimizationStageStrategy, Stage
from ..types import ST, OT

if TYPE_CHECKING:
    from ..optimization_engine import OptimizationEngine

@dataclass
class ResultLoggerConfig:
    """Configuration for result logging."""
    filename: str = "result.csv"
    write_header: bool = True
    append_mode: bool = False
    flush_on_write: bool = True


class ResultLogger(OptimizationStageStrategy[ST, OT]):
    """Logs final optimization results to CSV files.
    
    Captures key metrics including execution time, iterations, population size,
    and best objective values at the end of each optimization run.
    """
    
    def __init__(self, out_dir: Path, *, config: ResultLoggerConfig | None = None) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or ResultLoggerConfig()
        self._result_file = self._out_dir / self._config.filename
        self._header_written = False
        
        # Initialize file if not in append mode
        if not self._config.append_mode and self._result_file.exists():
            self._result_file.unlink()
    
    def execute(self, engine: "OptimizationEngine[ST, OT]", stage: Stage) -> None:
        """Execute result logging strategy."""
        if stage != Stage.RUN_END:
            return
        
        convergence = engine._convergence
        population = engine.population
        problem = engine.problem
        
        # Extract key metrics
        execution_time = getattr(engine, '_run_start_time', None)
        if execution_time is not None:
            from time import perf_counter
            execution_time = perf_counter() - execution_time
        else:
            execution_time = None
        
        iterations = convergence.iteration
        population_size = getattr(population, 'size', len(population.candidates) if hasattr(population, 'candidates') else 0)
        best_objective = population.best_objective
        evaluations = problem.evaluation_count if hasattr(problem, 'evaluation_count') else getattr(problem, 'get_evaluation_count', lambda: 0)()
        
        # Prepare row data
        row_data = {
            'execution_time': execution_time,
            'iterations': iterations,
            'population_size': population_size,
            'best_objective': float(best_objective) if best_objective is not None else None,
            'evaluations': evaluations,
        }
        
        # Write to CSV
        mode = 'a' if self._config.append_mode or self._header_written else 'w'
        with self._result_file.open(mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            # Write header if needed
            if self._config.write_header and (not self._header_written or mode == 'w'):
                writer.writeheader()
                self._header_written = True
            
            # Write data row
            writer.writerow(row_data)
            
            if self._config.flush_on_write:
                f.flush()


__all__ = ["ResultLogger", "ResultLoggerConfig"]