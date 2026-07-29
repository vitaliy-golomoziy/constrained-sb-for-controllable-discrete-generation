import numpy as np
import scipy.optimize
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace


class EntropyProjection:
    """
    KL projection onto the entropy constraint set  {p : H(p) >= a},
    where H is the average conditional entropy:

        H(p) = (1/K) * sum_{k=1}^{K} H(X_k | X_{-k})

    Solves: p* = argmin_{p : H(p) >= a, sum(p)=1, p>=0}  KL(p || q)

    Two cases handled separately:
    - K=1 (single token):  H(p) = plain Shannon entropy; p* is a temperature
      scaling of q, found by bisection on a 1-D monotone equation (fast).
    - K>1:  uses scipy.optimize for the inner fixed-λ problem, with outer
      bisection on the Lagrange multiplier λ.

    In both cases, if H(q) >= a the constraint is already satisfied and q is
    returned unchanged (no projection needed).

    Parameters
    ----------
    bisect_tol : float
        Convergence tolerance for the outer bisection on λ.
    bisect_max_iter : int
        Maximum bisection iterations.
    inner_tol : float
        Tolerance passed to scipy for the inner optimisation (K>1 case).
    """

    # Which functional the schedule levels are expressed in; read by
    # IPFDykstraSolver.check_feasibility to pick the right reachable envelope.
    SCHEDULE_FUNCTIONAL = "erasure"

    def __init__(
        self,
        bisect_tol: float = 1e-10,
        bisect_max_iter: int = 60,
        inner_tol: float = 1e-12,
    ):
        self._bisect_tol = bisect_tol
        self._bisect_max_iter = bisect_max_iter
        self._inner_tol = inner_tol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        q: np.ndarray,
        a: float,
        phase_space: DiscretePhaseSpace,
    ) -> np.ndarray:
        """
        Project q onto {p : H(p) >= a}.

        Parameters
        ----------
        q : np.ndarray of shape (num_states,)  — distribution to project.
        a : float — entropy lower bound.
        phase_space : DiscretePhaseSpace

        Returns
        -------
        np.ndarray of shape (num_states,) — the projected distribution p*.
        """
        q = np.asarray(q, dtype=float)
        q = np.clip(q, 0.0, None)
        q /= q.sum()

        if a <= 0.0:
            return q.copy()

        H_q = self._complexity(q, phase_space)
        if H_q >= a - 1e-12:
            return q.copy()

        if phase_space.n == 1:
            return self._project_k1(q, a)
        else:
            return self._project_kn(q, a, phase_space)

    # ------------------------------------------------------------------
    # Complexity functional and its gradient
    # ------------------------------------------------------------------

    def _complexity(self, p: np.ndarray, ps: DiscretePhaseSpace) -> float:
        """H(p) = (1/n) * sum_k H(X_k | X_{-k})."""
        p = np.maximum(p, 0.0)
        A, n = ps.alphabet_size, ps.n
        config_tensor = p.reshape([A] * n)

        def _ent(v):
            v = v[v > 0]
            return float(-np.sum(v * np.log(v)))

        h_full = _ent(config_tensor.ravel())
        total = 0.0
        for k in range(n):
            p_mk = config_tensor.sum(axis=k).ravel()
            total += h_full - _ent(p_mk)
        return total / n

    def _grad_complexity(self, p: np.ndarray, ps: DiscretePhaseSpace) -> np.ndarray:
        """
        Gradient of H(p) with respect to p.

            grad_x H(p) = -log(p_x) + (1/K) * sum_k log( p_{-k}(x) )

        where p_{-k}(x) is the marginal of p summed over token k, evaluated
        at x's reduced index.
        """
        A, n = ps.alphabet_size, ps.n
        p = np.maximum(p, 1e-300)
        config_tensor = p.reshape([A] * n)
        configs = ps.all_configurations()

        grad = -np.log(p)  # -log p_x  (shape: num_states,)

        if n > 1:
            for s, x in enumerate(configs):
                acc = 0.0
                for k in range(n):
                    p_mk = config_tensor.sum(axis=k)  # shape (A,)*(n-1)
                    idx_mk = tuple(int(x[j]) for j in range(n) if j != k)
                    acc += np.log(p_mk[idx_mk] + 1e-300)
                grad[s] += acc / n
        # for n==1 the (1/K)*sum_k log p_{-k}(x) term is 0 (empty marginal=1)

        return grad

    # ------------------------------------------------------------------
    # K=1 projection: temperature scaling + bisection
    # ------------------------------------------------------------------

    def _project_k1(self, q: np.ndarray, a: float) -> np.ndarray:
        """
        For K=1,  p*(λ) = q^{1/(1+λ)} / Z(λ).
        H(p*(λ)) is strictly increasing in λ >= 0.
        Bisect to find λ with H(p*(λ)) = a.
        """
        def p_lam(lam):
            log_p = np.log(np.maximum(q, 1e-300)) / (1.0 + lam)
            log_p -= np.max(log_p)  # shift for numerical stability
            p = np.exp(log_p)
            return p / p.sum()

        def H_lam(lam):
            p = p_lam(lam)
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)))

        # Find an upper bound where H(p*(λ_hi)) >= a
        lam_hi = 1.0
        for _ in range(30):
            if H_lam(lam_hi) >= a - 1e-12:
                break
            lam_hi *= 4.0

        lam_lo = 0.0
        for _ in range(self._bisect_max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            if H_lam(lam_mid) < a:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
            if lam_hi - lam_lo < self._bisect_tol:
                break

        return p_lam(0.5 * (lam_lo + lam_hi))

    # ------------------------------------------------------------------
    # K>1 projection: scipy inner solve + bisection on λ
    # ------------------------------------------------------------------

    def _project_kn(
        self, q: np.ndarray, a: float, ps: DiscretePhaseSpace
    ) -> np.ndarray:
        """
        For K>1, minimise  KL(p||q) - λ·H(p)  subject to  Σp=1, p≥0
        for a sequence of λ values via outer bisection on H(p*(λ)) = a.
        """
        S = ps.num_states

        def _inner_solve(lam: float, p0: np.ndarray) -> np.ndarray:
            """Minimise KL(p||q) - λ·H(p) over the probability simplex."""
            log_q = np.log(np.maximum(q, 1e-300))

            def obj(p):
                p = np.maximum(p, 1e-300)
                kl = float(np.sum(p * (np.log(p) - log_q)))
                h = self._complexity(p, ps)
                return kl - lam * h

            def jac(p):
                p = np.maximum(p, 1e-300)
                g_kl = np.log(p) - log_q + 1.0
                g_h = self._grad_complexity(p, ps)
                return g_kl - lam * g_h

            res = scipy.optimize.minimize(
                obj,
                x0=np.maximum(p0, 1e-12),
                jac=jac,
                method="SLSQP",
                bounds=[(1e-15, None)] * S,
                constraints={"type": "eq", "fun": lambda p: p.sum() - 1.0},
                options={"ftol": self._inner_tol, "maxiter": 500},
            )
            p_opt = np.clip(res.x, 0.0, None)
            return p_opt / p_opt.sum()

        # ------ outer bisection on λ to achieve H(p*(λ)) = a ------
        # λ=0 → p*=q with H(q) < a  (checked by caller)
        lam_lo, lam_hi = 0.0, 1.0
        p_lo = q.copy()

        # grow lam_hi until H(p*(lam_hi)) >= a
        for _ in range(30):
            p_hi = _inner_solve(lam_hi, p_lo)
            if self._complexity(p_hi, ps) >= a - 1e-10:
                break
            lam_lo, p_lo = lam_hi, p_hi
            lam_hi *= 4.0

        # bisect
        for _ in range(self._bisect_max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            p_mid = _inner_solve(lam_mid, p_lo)
            if self._complexity(p_mid, ps) < a:
                lam_lo, p_lo = lam_mid, p_mid
            else:
                lam_hi = lam_mid
                p_hi = p_mid
            if lam_hi - lam_lo < self._bisect_tol:
                break

        return _inner_solve(0.5 * (lam_lo + lam_hi), p_lo)
