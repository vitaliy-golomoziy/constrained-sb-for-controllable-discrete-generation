import numpy as np
try:
    import scipy.sparse
    _HAS_SPARSE = True
except ImportError:
    _HAS_SPARSE = False


class DegenerateMarginalError(RuntimeError):
    """
    Raised when a marginal cannot be computed because the potential-weighted
    path measure has collapsed to zero mass everywhere at some time step.

    This happens when the log-potentials develop an unbounded dynamic range,
    which in practice means the constraint set is infeasible (see
    Solver.Reachability) or the schedule is far outside the reachable envelope.

    It is deliberately an error and not a silently-substituted default: an
    earlier version of this class fell back to the uniform distribution here,
    which is indistinguishable from a genuine maximum-entropy solution and
    produced a spurious fixed point that passed the convergence test.
    """


def _log_matvec_csr(indptr, indices, log_data, log_v, S):
    """
    log( M @ exp(log_v) ) row-wise, for a CSR matrix M with log-data log_data.

    Computed with a per-row max shift, so no term can overflow or underflow:
    the result is exact up to floating-point rounding even when the entries of
    log_v span thousands of nats.
    """
    terms = log_data + log_v[indices]                  # (nnz,)
    row_lens = np.diff(indptr)
    nonempty = row_lens > 0
    out = np.full(S, -np.inf)
    if not np.any(nonempty):
        return out

    starts = indptr[:-1][nonempty]
    rmax = np.maximum.reduceat(terms, starts)
    finite = np.isfinite(rmax)
    shift = np.where(finite, rmax, 0.0)
    shift_per_nnz = np.repeat(shift, row_lens[nonempty])
    ex = np.exp(terms - shift_per_nnz)
    ssum = np.add.reduceat(ex, starts)
    with np.errstate(divide="ignore"):
        out[nonempty] = np.where(finite, shift + np.log(ssum), -np.inf)
    return out


def _log_matvec_dense(log_M, log_v):
    """log( M @ exp(log_v) ) row-wise for a dense matrix with log-entries log_M."""
    terms = log_M + log_v[None, :]
    rmax = terms.max(axis=1)
    finite = np.isfinite(rmax)
    shift = np.where(finite, rmax, 0.0)
    ex = np.exp(terms - shift[:, None])
    with np.errstate(divide="ignore"):
        return np.where(finite, shift + np.log(ex.sum(axis=1)), -np.inf)


class ForwardBackward:
    """
    Forward-backward algorithm for marginal computation under the potential
    factorisation  P(x_{0:N}) = R(x_{0:N}) * prod_t phi_t(x_t).

    Given N+1 potentials {phi_t}, the marginal at each time step is:

        P_t(x_t)  ∝  phi_t(x_t) * alpha_t(x_t) * beta_t(x_t)

    where the forward (alpha) and backward (beta) messages are:

        alpha_0(x_0)  = R_0(x_0)
        alpha_t(x_t)  = sum_{x_{t-1}} K(x_{t-1}, x_t) phi_{t-1}(x_{t-1}) alpha_{t-1}(x_{t-1})

        beta_N(x_N)   = 1
        beta_t(x_t)   = sum_{x_{t+1}} K(x_t, x_{t+1}) phi_{t+1}(x_{t+1}) beta_{t+1}(x_{t+1})

    Everything is carried in log-space with per-row max shifts.  This matters:
    once a schedule binds hard, the log-potentials routinely span several
    hundred nats, and forming the product phi*alpha*beta in linear space
    underflows to zero in *every* coordinate.  Per-time-step constant shifts
    cancel in the normalised marginal, so the shifts are free.

    Memory and time: O(N * S) and O(N * nnz) per call, where S = |S|^K.
    This is exponentially cheaper than materialising the full path measure
    (O(S^{N+1})), which is the key advantage over the direct solver.

    Parameters
    ----------
    R0 : np.ndarray of shape (S,)
        Initial distribution of the reference process.
    kernel : np.ndarray or scipy.sparse matrix of shape (S, S)
        Transition matrix K[s, s'] = P(X_{t+1}=s' | X_t=s).
    """

    def __init__(self, R0: np.ndarray, kernel):
        self._R0 = np.asarray(R0, dtype=float)
        self._S = len(self._R0)
        with np.errstate(divide="ignore"):
            self.log_R0 = np.log(self._R0)

        self._sparse = _HAS_SPARSE and scipy.sparse.issparse(kernel)
        if self._sparse:
            self._K = kernel
            # backward pass needs K @ v, forward pass needs K^T @ v
            fwd = scipy.sparse.csr_matrix(kernel.T)
            bwd = scipy.sparse.csr_matrix(kernel)
            with np.errstate(divide="ignore"):
                self._fwd = (fwd.indptr, fwd.indices, np.log(fwd.data))
                self._bwd = (bwd.indptr, bwd.indices, np.log(bwd.data))
        else:
            self._K = np.asarray(kernel, dtype=float)
            with np.errstate(divide="ignore"):
                self._log_K = np.log(self._K)
                self._log_KT = np.log(self._K.T)

    # ------------------------------------------------------------------
    # Single kernel applications (log-space)
    # ------------------------------------------------------------------

    def log_forward_step(self, log_v: np.ndarray) -> np.ndarray:
        """log( K^T @ exp(log_v) ) — one step of the alpha recursion."""
        if self._sparse:
            ip, ix, ld = self._fwd
            return _log_matvec_csr(ip, ix, ld, log_v, self._S)
        return _log_matvec_dense(self._log_KT, log_v)

    def log_backward_step(self, log_v: np.ndarray) -> np.ndarray:
        """log( K @ exp(log_v) ) — one step of the beta recursion."""
        if self._sparse:
            ip, ix, ld = self._bwd
            return _log_matvec_csr(ip, ix, ld, log_v, self._S)
        return _log_matvec_dense(self._log_K, log_v)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def marginals(self, potentials: list) -> list:
        """
        Compute all T+1 marginals given a list of N+1 potentials (linear space).

        Kept for backward compatibility; internally delegates to the log-space
        implementation.

        Returns
        -------
        list of np.ndarray, each shape (S,), each summing to 1.
        """
        with np.errstate(divide="ignore"):
            log_pot = [np.log(np.asarray(p, dtype=float)) for p in potentials]
        return self.marginals_log(log_pot)

    def marginals_log(self, log_potentials: list) -> list:
        """Compute all T+1 marginals from log-potentials."""
        log_alpha, log_beta = self.messages_log(log_potentials)
        return self.combine_log(log_potentials, log_alpha, log_beta)

    def marginal_at(self, potentials: list, t: int) -> np.ndarray:
        """
        Compute the marginal at a single time step t.
        Runs a full forward-backward pass; prefer marginals() when all
        time steps are needed.
        """
        return self.marginals(potentials)[t]

    def messages_log(self, log_potentials: list):
        """
        Both message passes in log-space.

        Returns
        -------
        (log_alpha, log_beta) : each a list of N+1 arrays of shape (S,).
            Each message is shifted to have max 0; the per-step constants
            cancel in combine_log.
        """
        N = len(log_potentials) - 1
        log_alpha = [None] * (N + 1)
        log_alpha[0] = self.log_R0.copy()
        for t in range(1, N + 1):
            v = log_alpha[t - 1] + log_potentials[t - 1]
            a = self.log_forward_step(v)
            log_alpha[t] = self._shift(a, f"alpha_{t}")

        log_beta = [None] * (N + 1)
        log_beta[N] = np.zeros(self._S)
        for t in range(N - 1, -1, -1):
            v = log_beta[t + 1] + log_potentials[t + 1]
            b = self.log_backward_step(v)
            log_beta[t] = self._shift(b, f"beta_{t}")
        return log_alpha, log_beta

    def combine_log(self, log_potentials: list, log_alpha: list,
                    log_beta: list) -> list:
        """Form the normalised marginals from log-potentials and log-messages."""
        out = []
        for t, (lp, la, lb) in enumerate(zip(log_potentials, log_alpha, log_beta)):
            out.append(self.marginal_from_parts(lp, la, lb, t))
        return out

    def marginal_from_parts(self, log_phi_t, log_alpha_t, log_beta_t,
                            t=None) -> np.ndarray:
        """Normalised P_t from log phi_t + log alpha_t + log beta_t."""
        log_m = log_phi_t + log_alpha_t + log_beta_t
        mx = log_m.max()
        if not np.isfinite(mx):
            raise DegenerateMarginalError(
                f"marginal at t={t} has zero total mass: "
                f"log(phi*alpha*beta) is -inf in all {self._S} states. "
                "The potentials have diverged, which usually means the entropy "
                "schedule is outside the reachable envelope."
            )
        m = np.exp(log_m - mx)
        total = m.sum()
        if not (total > 0.0) or not np.isfinite(total):
            raise DegenerateMarginalError(
                f"marginal at t={t} did not normalise (sum={total!r})."
            )
        return m / total

    # ------------------------------------------------------------------

    def _shift(self, log_msg: np.ndarray, what: str) -> np.ndarray:
        mx = log_msg.max()
        if not np.isfinite(mx):
            raise DegenerateMarginalError(
                f"message {what} is -inf in all {self._S} states; the "
                "reference process cannot connect the prescribed endpoints."
            )
        return log_msg - mx
