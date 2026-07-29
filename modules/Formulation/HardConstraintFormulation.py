import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Formulation.EntropyFormulation import EntropyFormulation


class HardConstraintFormulation(EntropyFormulation):
    """
    Hard entropy constraint: E_t >= a_t enforced exactly for every t.

    Adds no penalty term to the objective; instead adds one inequality
    constraint per time step.  The feasible set {μ : E_t(μ) >= a_t} is
    convex (E_t is concave), so the full problem remains convex and CVXPY
    can solve it directly.
    """

    def objective_term(self, complexity_exprs: list, schedule: np.ndarray):
        return 0

    def extra_constraints(self, complexity_exprs: list, schedule: np.ndarray) -> list:
        return [E_t >= float(a_t) for E_t, a_t in zip(complexity_exprs, schedule)]
