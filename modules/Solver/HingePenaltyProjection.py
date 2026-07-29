import numpy as np

from .JointEntropyProjection import JointEntropyProjection


class HingePenaltyProjection(JointEntropyProjection):
    """
    Proximal step for the one-sided hinge penalty on the joint entropy:

        p* = argmin_p  KL(p || q) + beta * (a - H(p))_+ ,   p in simplex.

    This replaces the hard projection onto {H(p) >= a} in the IPF sweep, giving
    the penalized formulation of the entropy-constrained bridge.  Its practical
    virtue is that it stays bounded when the requested level is unreachable: the
    hard projection has no solution once a exceeds the reachable envelope (see
    Solver.Reachability) and drives the potentials divergent, whereas this prox
    just tempers by a fixed amount and stops.

    Closed form
    -----------
    The penalty is piecewise, so the solution has three branches.  Writing
    p_lambda for the temperature scaling q^{1/(1+lambda)}/Z (inherited from
    JointEntropyProjection), and using d(-H)/dp_x = log p_x + 1:

      1. H(q) >= a — penalty inactive, nothing to do:      p* = q
      2. H(p_beta) <= a — penalty active and smooth; the
         stationarity condition log(p/q) + 1 + beta(log p + 1) + eta = 0
         gives exactly a fixed temperature lambda = beta: p* = p_beta
      3. otherwise the optimum sits on the kink H(p) = a, which is the hard
         projection: bisect lambda in [0, beta] for H(p_lambda) = a

    Since H(p_lambda) increases in lambda, these three cases are exhaustive and
    mutually exclusive, and beta -> infinity recovers the hard projection.

    Parameters
    ----------
    beta : float
        Penalty weight > 0.  Larger values track the hard constraint more
        closely; smaller values tolerate more violation.
    """

    SCHEDULE_FUNCTIONAL = "joint"

    def __init__(self, beta: float, bisect_tol: float = 1e-10,
                 bisect_max_iter: int = 60):
        if not beta > 0:
            raise ValueError("beta must be positive")
        super().__init__(bisect_tol=bisect_tol, bisect_max_iter=bisect_max_iter)
        self.beta = float(beta)

    def project(self, q: np.ndarray, a: float, phase_space=None) -> np.ndarray:
        """
        Prox step at level a.  Signature matches the hard projections so the
        two are interchangeable inside IPFDykstraSolver.
        """
        q = np.asarray(q, dtype=float)
        q = np.clip(q, 0.0, None)
        q = q / q.sum()

        if a <= 0.0:
            return q.copy()

        # branch 1: penalty inactive
        if self._entropy(q) >= a - 1e-12:
            return q.copy()

        # branch 2: penalty active and smooth — fixed temperature lambda = beta
        p_beta = self._p_lam(q, self.beta)
        if self._entropy(p_beta) <= a:
            return p_beta

        # branch 3: optimum on the kink H(p) = a — the hard projection
        return self._bisect(q, a)
