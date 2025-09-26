"""Reusable population logging strategies.

Provides a generic PopulationLogger that records (per iteration) the full
population with both normalized and real-domain parameter values when
parameter normalizers are provided via the engine.

Features
--------
- Logs one CSV per iteration for easy post-processing / streaming analysis.
- Column naming: <alias> (or name) used; normalized columns get *_n suffix.
- Configurable file naming pattern and inclusion of headers each file.
- Optional master index file (disabled by default to avoid huge single file).

Design notes
------------
- We rely on the engine.parameters exposing ParameterSpec with a .normalizer
  implementing to_real().
- Engine.population must provide .candidates (Sequence[NDArrayFloat]) and
  .objectives (Sequence[float]).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List, Any
import csv

from . import OptimizationStageStrategy, Stage
from ..types import ST, OT

################################################################################
# NOTE: We previously used Protocol-based structural types here. They were
# removed to reduce noise—everything needed is available directly via the
# engine:
#   engine.parameters -> iterable of parameter specs (with .alias/.name/.normalizer)
#   engine.population.candidates / .objectives / .size
# This keeps the logger lean and avoids pyright complaints about extraneous
# protocols.
################################################################################

@dataclass
class PopulationLoggerConfig:
    per_iteration_dir: str = "data"
    filename_pattern: str = "iter_{iteration}.csv"  # available key: iteration
    write_header_each_file: bool = True
    create_master_index: bool = False
    master_filename: str = "population_all.csv"
    flush_on_write: bool = False  # Set True if tail -f style monitoring needed

class PopulationLogger(OptimizationStageStrategy[ST, OT]):  # type: ignore[type-arg]
    def __init__(self, out_dir: Path, *, config: PopulationLoggerConfig | None = None) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or PopulationLoggerConfig()
        self._data_dir = self._out_dir / self._config.per_iteration_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._iteration = -1
        self._aliases: list[str] | None = None
        self._master_file = None
        if self._config.create_master_index:
            mpath = self._out_dir / self._config.master_filename
            self._master_file = mpath.open("w", newline="")
            self._master_writer = csv.writer(self._master_file)
        else:
            self._master_writer = None

    # --------------------------------------------------------------
    def _ensure_aliases(self, params: Sequence[Any]) -> list[str]:
        if self._aliases is not None:
            return self._aliases
        aliases: list[str] = []
        for p in params:
            alias = getattr(p, "alias", None) or getattr(p, "name", None) or f"p{len(aliases)}"
            aliases.append(str(alias))
        self._aliases = aliases
        return aliases

    def _denorm_population(self, candidates: Sequence[Any], params: Sequence[Any]) -> list[list[float]]:
        out: list[list[float]] = []
        for cand in candidates:
            try:
                row = [float(p.normalizer.to_real(float(cand[i]))) for i, p in enumerate(params)]
            except Exception:  # fall back gracefully if unexpected shape
                row = [float(cand[i]) for i in range(min(len(cand), len(params)))]  # type: ignore[index]
            out.append(row)
        return out

    def _build_header(self, aliases: Sequence[str]) -> list[str]:
        return ["iteration", "candidate_index", "objective"] + [f"{a}_n" for a in aliases] + list(aliases)

    def _write_rows(
        self,
        writer: Any,
        iteration: int,
        candidates: Sequence[Any],
        objectives: Sequence[Any],
        denorm: Sequence[Sequence[float]],
        *,
        write_header: bool,
        header: Sequence[str],
    ) -> None:
        if write_header:
            writer.writerow(header)
        for idx, (cand, obj, real_vals) in enumerate(zip(candidates, objectives, denorm)):
            row: List[float | int] = [iteration, idx, float(obj)]
            row.extend(float(x) for x in cand)  # normalized
            row.extend(float(v) for v in real_vals)  # real
            writer.writerow(row)

    # --------------------------------------------------------------
    def execute(self, engine, stage: Stage) -> None:  # type: ignore[override]
        params = list(engine.parameters)
        pop = engine.population
        if stage == Stage.RUN_START:
            self._iteration = 0
        elif stage == Stage.ITERATION:
            self._iteration += 1
        elif stage == Stage.RUN_END:
            # keep same iteration value (log final state)
            pass
        else:
            return

        if getattr(pop, "size", 0) == 0:
            return
        aliases = self._ensure_aliases(params)
        header = self._build_header(aliases)
        denorm = self._denorm_population(pop.candidates, params)
        fname = self._config.filename_pattern.format(iteration=self._iteration)
        iter_path = self._data_dir / fname
        with iter_path.open("w", newline="") as f:
            writer = csv.writer(f)
            self._write_rows(
                writer,
                self._iteration,
                pop.candidates,
                pop.objectives,
                denorm,
                write_header=self._config.write_header_each_file,
                header=header,
            )
            if self._config.flush_on_write:
                try:
                    f.flush()
                except Exception:
                    pass
        if self._master_writer is not None:
            # Write header only once at first iteration
            if self._iteration == 0:
                self._master_writer.writerow(header)
            self._write_rows(
                self._master_writer,
                self._iteration,
                pop.candidates,
                pop.objectives,
                denorm,
                write_header=False,
                header=header,
            )
            if self._config.flush_on_write:
                try:
                    self._master_file.flush()  # type: ignore[union-attr]
                except Exception:
                    pass

    # --------------------------------------------------------------
    def __del__(self) -> None:  # best-effort close
        try:
            if self._master_file is not None:
                self._master_file.close()
        except Exception:
            pass

__all__ = ["PopulationLogger", "PopulationLoggerConfig"]
