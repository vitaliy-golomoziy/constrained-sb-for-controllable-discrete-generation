import numpy as np


class JointEntropyProjection:
    """
    KL projection onto {p : H(p) >= a} where H is the joint Shannon entropy.

        H(p) = -sum_s p_s log p_s

    Solved via temperature scaling (Gibbs tilting):

        p*(lambda) propto p^{1/(1+lambda)},  lambda >= 0

    H(p*(lambda)) is strictly increasing in lambda, so the constraint
    H(p*(lambda)) = a is located by bisection.  Each evaluation is O(S)
    regardless of the number of tokens K, making this feasible for S=65536.

    This is the tractable proxy used when K is large enough that the
    conditional-entropy projection (SLSQP over S variables) is too slow.

    Parameters
    ----------
    bisect_tol : float
    bisect_max_iter : int
    """

    # Which functional the schedule levels are expressed in; read by
    # IPFDykstraSolver.check_feasibility to pick the right reachable envelope.
    SCHEDULE_FUNCTIONAL = "joint"

    def __init__(self, bisect_tol: float = 1e-10, bisect_max_iter: int = 60):
        self._bisect_tol = bisect_tol
        self._bisect_max_iter = bisect_max_iter

    def project(self, q: np.ndarray, a: float, phase_space=None) -> np.ndarray:
        """
        Project q onto {p : H(p) >= a}.

        Parameters
        ----------
        q : np.ndarray of shape (S,)
        a : float — joint entropy lower bound (nats)
        phase_space : unused, accepted for API compatibility

        Returns
        -------
        np.ndarray of shape (S,) — projected distribution p*
        """
        q = np.asarray(q, dtype=float)
        q = np.clip(q, 0.0, None)
        q /= q.sum()

        if a <= 0.0:
            return q.copy()

        if self._entropy(q) >= a - 1e-12:
            return q.copy()

        return self._bisect(q, a)

    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p = p[p > 1e-300]
        return float(-np.sum(p * np.log(p)))

    def _p_lam(self, q: np.ndarray, lam: float) -> np.ndarray:
        log_p = np.log(np.maximum(q, 1e-300)) / (1.0 + lam)
        log_p -= log_p.max()
        p = np.exp(log_p)
        return p / p.sum()

    def _bisect(self, q: np.ndarray, a: float) -> np.ndarray:
        # find upper bound: lambda large enough that H(p*(lambda)) >= a
        lam_hi = 1.0
        for _ in range(30):
            if self._entropy(self._p_lam(q, lam_hi)) >= a - 1e-12:
                break
            lam_hi *= 4.0

        lam_lo = 0.0
        for _ in range(self._bisect_max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            if self._entropy(self._p_lam(q, lam_mid)) < a:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
            if lam_hi - lam_lo < self._bisect_tol:
                break

        return self._p_lam(q, 0.5 * (lam_lo + lam_hi))
