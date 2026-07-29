import numpy as np
from abc import ABC, abstractmethod


class EntropyFormulation(ABC):
    """
    Abstract encoding of the entropy constraint E_t >= a_t in the SB objective.

    The SBProblem builds CVXPY expressions for E_t and passes them here.
    Each subclass decides whether to enforce the constraint exactly (hard),
    as a one-sided soft penalty (hinge), or as a two-sided squared penalty (CCCP).

    Interface contract
    ------------------
    SBProblem calls both methods and adds their outputs to the CVXPY problem:
      - objective_term  → added to the minimisation objective
      - extra_constraints → appended to the constraint list

    For CCCP formulations (requires_cccp = True), SBProblem passes linearised
    expressions L_t in place of the true concave E_t expressions.
    """

    @abstractmethod
    def objective_term(self, complexity_exprs: list, schedule: np.ndarray):
        """
        CVXPY expression to add to the objective, or 0 if none.

        Parameters
        ----------
        complexity_exprs : list of cp.Expression
            [E_0, ..., E_T] (or linearised L_t for CCCP formulations).
        schedule : np.ndarray of shape (T+1,)
            Target lower bounds a_t.
        """

    @abstractmethod
    def extra_constraints(self, complexity_exprs: list, schedule: np.ndarray) -> list:
        """
        List of CVXPY constraints to add, or [] if none.

        Parameters
        ----------
        complexity_exprs : list of cp.Expression
        schedule : np.ndarray of shape (T+1,)
        """

    @property
    def requires_cccp(self) -> bool:
        """True if this formulation needs the outer CCCP linearisation loop."""
        return False
