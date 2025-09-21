# Copilot Context & Contribution Rules

This repository contains Python code for optimization algorithms (e.g., genetic algorithms) under `src/`. Use these instructions to guide code generation and edits.

## Quality Gates (must pass)
- Pylance diagnostics: zero errors across the workspace after changes.
- No syntax errors. Code must import and type-check cleanly.
- Keep diffs minimal: do not reformat unrelated code or change public APIs unless necessary.
- No new uses of `Any`; existing `Any` must be removed or wrapped with precise types.

## Python typing & design guidelines
- Use full type hints everywhere. Avoid `Any` at all cost. Do not introduce it; if an external API forces `Any`, isolate it at boundaries and immediately narrow to precise types (via parsing, validation, or typed wrappers).
- Prefer Protocols for interfaces. For optimizers, follow the patterns in `src/optimization/optimization_protocol.py`:
  - Use type variables with clear variance:
    - `ST` for solution type (invariant)
    - `OT` for objective type (invariant, numeric)
    - `OT_co` for problem objective type (covariant, numeric)
  - Use `numpy.typing.NDArray[Any]` for NumPy arrays instead of `np.ndarray`.
  - Annotate tuples precisely, e.g., `Tuple[float, float]` not `tuple`.
- Keep docstrings concise and explain inputs/outputs and return types.

## Coding style
- Follow existing code style and structure. Use small, focused changes.
- Co-locate new modules under `src/genetic_algorithms/` or `src/optimization/` as appropriate.
- If you change a Protocol or public interface, update all usages in the repo in the same change.
- Prefer one class/protocol per file. If a file contains multiple classes, split them unless there is a compelling reason not to.

## When adding features
- Include a tiny example or usage snippet in a docstring or README when feasible.
- Prefer small, composable functions. Keep algorithms deterministic where possible or make randomness injectable.

## Definition of Done
- You’ve run a quick pass to ensure:
  - No Pylance errors remain.
  - Types are accurate (especially generics and Protocols).
  - Files import without runtime errors.
- Changes are documented briefly (inline or in README) if behavior is user-visible.

## Notes
- If `pyproject.toml` configures linting/formatting, follow it. If absent, keep current formatting and avoid mass reformatting.
- For population-based algorithms, respect the interfaces in `optimization_protocol.py` (population size, selection, crossover, mutation, etc.).