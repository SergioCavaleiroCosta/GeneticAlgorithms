"""Protocol for initializing the starting population and seeding the updater."""
from typing import Protocol
from abc import abstractmethod
from .types import ST, OT
from .optimization_problem import OptimizationProblem
from .update_rule import UpdateRule
from .population import Population


class SolutionInitializer(Protocol[ST, OT]):
    """Create the initial population and seed the update rule.

    Implementations should compute objectives via the provided problem,
    build a Population (size >= 1 recommended), choose the initial
    current candidate convention, and seed the updater accordingly.
    """

    @abstractmethod
    def initialize(
        self,
        problem: OptimizationProblem[ST, OT],
        updater: UpdateRule[ST, OT],
    ) -> Population[ST, OT]:
        """Return an initial population and seed the updater."""
        ...


__all__ = ["SolutionInitializer"]
