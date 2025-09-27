"""Plotting utilities for NSGA-II DTLZ2 multi-objective optimization visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

# Make project src/ importable
import sys
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nsga_ii.pareto_utils import non_dominated_sort
from .problems import DTLZ2Problem


def plot_pareto_front(objectives: list[Sequence[float]], output_dir: Path, iteration: int = -1, num_objectives: int = 2) -> None:
    """Plot the current population and true Pareto front with color-coded ranks."""
    if num_objectives == 2:
        _plot_2d_pareto_front(objectives, output_dir, iteration)
    elif num_objectives == 3:
        _plot_3d_pareto_front(objectives, output_dir, iteration)
    else:
        # For 4+ objectives, plot pairwise projections
        _plot_high_dim_pareto_front(objectives, output_dir, iteration, num_objectives)


def _plot_2d_pareto_front(objectives: list[Sequence[float]], output_dir: Path, iteration: int = -1) -> None:
    """Plot 2D Pareto front."""
    # Extract objective values
    f1_vals = [obj[0] for obj in objectives]
    f2_vals = [obj[1] for obj in objectives]
    
    # Perform non-dominated sorting to get ranks
    fronts = non_dominated_sort(objectives)
    
    # Create labels showing the generation number
    title = f'NSGA-II on DTLZ2 Problem - Generation {iteration}'
    
    # Define colors for the first 4 ranks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    rank_labels = ['Rank 1 (Best)', 'Rank 2', 'Rank 3', 'Rank 4']
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot each rank with different colors
    for rank, front in enumerate(fronts[:4]):  # Only plot first 4 ranks
        if not front:  # Skip empty fronts
            continue
            
        # Extract coordinates for this rank
        rank_f1 = [f1_vals[i] for i in front]
        rank_f2 = [f2_vals[i] for i in front]
        
        plt.scatter(rank_f1, rank_f2, 
                   alpha=0.7, s=40, 
                   color=colors[rank], 
                   label=f'{rank_labels[rank]} ({len(front)} points)',
                   edgecolors='black', linewidth=0.5)
    
    # Plot remaining ranks (5+) in gray if they exist
    if len(fronts) > 4:
        remaining_indices = []
        for front in fronts[4:]:
            remaining_indices.extend(front)
        
        if remaining_indices:
            remaining_f1 = [f1_vals[i] for i in remaining_indices]
            remaining_f2 = [f2_vals[i] for i in remaining_indices]
            plt.scatter(remaining_f1, remaining_f2, 
                       alpha=0.4, s=20, 
                       color='gray', 
                       label=f'Ranks 5+ ({len(remaining_indices)} points)')
    
    # Plot true Pareto front for DTLZ2 (quarter circle)
    problem = DTLZ2Problem(num_objectives=2)
    true_front = problem.get_true_pareto_front(100)
    plt.plot(true_front[:, 0], true_front[:, 1], 'r-', linewidth=2, label='True Pareto Front')
    
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if iteration >= 0:
        filename = f"pareto_front_gen_{iteration:03d}.png"
    else:
        filename = "final_pareto_front.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_3d_pareto_front(objectives: list[Sequence[float]], output_dir: Path, iteration: int = -1) -> None:
    """Plot 3D Pareto front."""
    # Extract objective values
    f1_vals = [obj[0] for obj in objectives]
    f2_vals = [obj[1] for obj in objectives]
    f3_vals = [obj[2] for obj in objectives]
    
    # Perform non-dominated sorting to get ranks
    fronts = non_dominated_sort(objectives)
    
    # Create labels showing the generation number
    title = f'NSGA-II on DTLZ2 Problem (3D) - Generation {iteration}'
    
    # Define colors for the first 4 ranks
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    rank_labels = ['Rank 1 (Best)', 'Rank 2', 'Rank 3', 'Rank 4']
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each rank with different colors
    for rank, front in enumerate(fronts[:4]):
        if not front:
            continue
            
        rank_f1 = [f1_vals[i] for i in front]
        rank_f2 = [f2_vals[i] for i in front]
        rank_f3 = [f3_vals[i] for i in front]
        
        ax.scatter(rank_f1, rank_f2, rank_f3,
                  alpha=0.7, s=40, 
                  color=colors[rank], 
                  label=f'{rank_labels[rank]} ({len(front)} points)')
    
    # Plot true Pareto front (eighth of sphere)
    problem = DTLZ2Problem(num_objectives=3)
    true_front = problem.get_true_pareto_front(400)
    ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], 
              c='red', s=1, alpha=0.6, label='True Pareto Front')
    
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_title(title)
    ax.legend()
    
    # Save plot
    if iteration >= 0:
        filename = f"pareto_front_gen_{iteration:03d}.png"
    else:
        filename = "final_pareto_front.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_high_dim_pareto_front(objectives: list[Sequence[float]], output_dir: Path, iteration: int = -1, num_objectives: int = 4) -> None:
    """Plot high-dimensional Pareto front using pairwise projections."""
    # For 4+ objectives, create subplot matrix of pairwise projections
    n_pairs = min(6, num_objectives * (num_objectives - 1) // 2)  # Limit to 6 pairs
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Perform non-dominated sorting
    fronts = non_dominated_sort(objectives)
    
    # Get pairs of objectives to plot
    pairs = []
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            pairs.append((i, j))
            if len(pairs) >= 6:  # Limit to 6 pairs
                break
        if len(pairs) >= 6:
            break
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        
        # Extract objective values for this pair
        fi_vals = [obj[i] for obj in objectives]
        fj_vals = [obj[j] for obj in objectives]
        
        # Plot first rank only for clarity
        if fronts:
            front = fronts[0]
            rank_fi = [fi_vals[k] for k in front]
            rank_fj = [fj_vals[k] for k in front]
            ax.scatter(rank_fi, rank_fj, alpha=0.7, s=20, color=colors[0])
        
        ax.set_xlabel(f'f{i+1}')
        ax.set_ylabel(f'f{j+1}')
        ax.set_title(f'f{i+1} vs f{j+1}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(pairs), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'NSGA-II on DTLZ2 Problem ({num_objectives}D) - Generation {iteration}')
    plt.tight_layout()
    
    # Save plot
    if iteration >= 0:
        filename = f"pareto_front_gen_{iteration:03d}.png"
    else:
        filename = "final_pareto_front.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


class ParetoPlotter:
    """Event strategy for plotting Pareto fronts during optimization."""
    
    def __init__(self, output_dir: Path, plot_every: int = 10, num_objectives: int = 2) -> None:
        self.output_dir = output_dir
        self.plot_every = plot_every
        self.num_objectives = num_objectives
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, engine, stage) -> None:
        iteration = getattr(engine._convergence, 'iteration', 0)
        
        if iteration % self.plot_every == 0 or iteration == 0:
            objectives = engine.population.objectives
            plot_pareto_front(objectives, self.figures_dir, iteration, self.num_objectives)


class MultiObjectivePopulationLogger:
    """Logger for multi-objective population data."""
    
    def __init__(self, output_dir: Path, num_objectives: int = 2) -> None:
        self.output_dir = output_dir
        self.num_objectives = num_objectives
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.objectives_file = self.output_dir / "objectives.csv"
        self.solutions_file = self.output_dir / "solutions.csv"
        
        # Initialize files with headers
        obj_headers = ",".join([f"f{i+1}" for i in range(num_objectives)])
        with open(self.objectives_file, 'w') as f:
            f.write(f"iteration,individual,{obj_headers}\n")
        with open(self.solutions_file, 'w') as f:
            f.write("iteration,individual,solution\n")
    
    def execute(self, engine, stage) -> None:
        iteration = getattr(engine._convergence, 'iteration', 0)
        population = engine.population
        
        # Log objectives
        with open(self.objectives_file, 'a') as f:
            for i, objectives in enumerate(population.objectives):
                obj_str = ",".join(f"{obj:.6f}" for obj in objectives)
                f.write(f"{iteration},{i},{obj_str}\n")
        
        # Log solutions (first 5 dimensions only to avoid huge files)
        with open(self.solutions_file, 'a') as f:
            for i, solution in enumerate(population.candidates):
                sol_str = ",".join(f"{x:.6f}" for x in solution[:5])
                f.write(f"{iteration},{i},{sol_str}\n")


__all__ = ["plot_pareto_front", "ParetoPlotter", "MultiObjectivePopulationLogger"]