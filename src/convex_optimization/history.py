"""Per-iteration convergence history for the optimization methods.

Every solver returns a ``Result`` carrying a full ``History``: the objective
value and a problem-appropriate residual at every iterate, starting from the
initial point. Nothing is only printed; the history is a data structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class History:
    """Objective value and residual recorded at every iterate."""

    objectives: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)

    def append(self, objective: float, residual: float) -> None:
        self.objectives.append(float(objective))
        self.residuals.append(float(residual))

    def __len__(self) -> int:
        return len(self.objectives)

    def tail(self, n: int = 5) -> list[tuple[int, float, float]]:
        """The last ``n`` recorded rows as (iteration, objective, residual)."""
        start = max(0, len(self.objectives) - n)
        return [
            (i, self.objectives[i], self.residuals[i]) for i in range(start, len(self.objectives))
        ]


@dataclass
class Result:
    """Outcome of one solver run."""

    x: np.ndarray
    history: History
    n_iter: int  # number of updates applied (len(history) - 1)
    converged: bool
    tol: float

    def final_objective(self) -> float:
        return self.history.objectives[-1]

    def final_residual(self) -> float:
        return self.history.residuals[-1]

    def final_objective_gap(self, f_star: float) -> float:
        """f(x_final) - f(x_star); a negative value means x_star is worse."""
        return self.final_objective() - float(f_star)
