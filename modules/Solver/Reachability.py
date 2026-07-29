"""
Reachable entropy envelope for entropy-constrained Schrödinger bridges.

Any path measure P with KL(P || R) < infinity satisfies
supp(P_{t,t+1}) subset supp(R_{t,t+1}).  Writing

    s_max = max_y |{x : K(x, y) > 0}|

for the reverse fan-in of the reference kernel, the chain rule gives

    H(P_t) <= H(P_t, P_{t+1}) = H(P_{t+1}) + H(X_t | X_{t+1})
           <= H(P_{t+1}) + log s_max,

and symmetrically H(P_{t+1}) <= H(P_t) + log s_max.  Iterating from both
endpoints bounds the joint entropy at every interior time:

    H(P_t) <= min( K log A,
                   H(p_T) + (N - t) log s_max,
                   H(p_0) + t       log s_max ).

A schedule that asks for more than this is infeasible for *any* finite-KL
path, no matter how the solver is implemented.  For a one-flip Gibbs kernel on
K binary tokens, s_max = K + 1, so the envelope collapses hard near a
concentrated endpoint: with H(p_T) = 1.97 nats and K = 16, H(P_{N-1}) can
never exceed 1.97 + log 17 = 4.80 nats, however aggressive the schedule.

For the erasure (average conditional) entropy the same envelope applies after
dividing by K, since H(p) = (1/K) sum_k H(X_k | X_{-k}) <= H(p) / K by the
chain rule (conditioning on X_{-k} conditions on at least X_{1..k-1}).
That bound is valid but loose; it is capped at log A.

This condition is necessary, not sufficient: a schedule inside the envelope
may still be infeasible.  Its purpose is to catch the schedules that provably
cannot be met before the solver silently diverges trying.
"""
import numpy as np

try:
    import scipy.sparse
    _HAS_SPARSE = True
except ImportError:
    _HAS_SPARSE = False


def joint_entropy(p: np.ndarray) -> float:
    """Joint Shannon entropy in nats."""
    p = np.asarray(p, dtype=float).ravel()
    pos = p[p > 1e-300]
    return float(-np.sum(pos * np.log(pos)))


def kernel_fan_in(kernel) -> int:
    """
    s_in = max_y |{x : K(x, y) > 0}| — how many predecessors a state can have.
    Bounds H(X_t | X_{t+1}), hence the backward recursion from p_T.

    For a kernel that resamples one of K tokens per step this is K*(A-1)+1;
    for the binary one-flip Gibbs kernel, K+1.
    """
    if _HAS_SPARSE and scipy.sparse.issparse(kernel):
        coo = kernel.tocoo()
        keep = coo.data > 0
        if not np.any(keep):
            return 1
        counts = np.bincount(coo.col[keep], minlength=kernel.shape[1])
        return int(counts.max())
    K = np.asarray(kernel, dtype=float)
    return int((K > 0).sum(axis=0).max())


def kernel_fan_out(kernel) -> int:
    """
    s_out = max_x |{y : K(x, y) > 0}| — how many successors a state can have.
    Bounds H(X_{t+1} | X_t), hence the forward recursion from p_0.

    Equal to the fan-in whenever the kernel's support is symmetric, which holds
    for every Gibbs reference in this implementation; the two are kept separate because
    the two directions of the envelope are bounded by different constants.
    """
    if _HAS_SPARSE and scipy.sparse.issparse(kernel):
        coo = kernel.tocoo()
        keep = coo.data > 0
        if not np.any(keep):
            return 1
        counts = np.bincount(coo.row[keep], minlength=kernel.shape[0])
        return int(counts.max())
    K = np.asarray(kernel, dtype=float)
    return int((K > 0).sum(axis=1).max())


def entropy_envelope(h_p0, h_pT, num_steps, s_max, num_tokens,
                     alphabet_size, functional="joint", s_out=None):
    """
    Upper bound on the constrained functional at each time t = 0..num_steps.

    Parameters
    ----------
    h_p0, h_pT : float
        Joint Shannon entropies of the prescribed endpoint marginals.
    num_steps : int
        N, so there are N+1 marginals.
    s_max : int
        Reverse fan-in of the reference kernel (see kernel_fan_in); bounds the
        backward recursion from p_T.
    num_tokens, alphabet_size : int
        K and A.
    functional : {"joint", "erasure"}
        Which functional the schedule is expressed in.
    s_out : int, optional
        Forward fan-out (see kernel_fan_out); bounds the forward recursion from
        p_0.  Defaults to s_max, which is exact for symmetric-support kernels.

    Returns
    -------
    np.ndarray of shape (num_steps+1,)
    """
    log_s = np.log(max(s_max, 1))
    log_s_out = np.log(max(s_out if s_out is not None else s_max, 1))
    log_card = num_tokens * np.log(alphabet_size)
    t = np.arange(num_steps + 1)
    env = np.minimum.reduce([
        np.full(num_steps + 1, log_card),
        h_pT + (num_steps - t) * log_s,
        h_p0 + t * log_s_out,
    ])
    if functional == "joint":
        return env
    if functional == "erasure":
        return np.minimum(env / num_tokens, np.log(alphabet_size))
    raise ValueError(f"unknown functional {functional!r}")


def check_schedule(schedule, envelope, atol=1e-9):
    """
    Interior time steps whose requested level exceeds the envelope.

    Returns
    -------
    list of (t, requested, max_feasible)
    """
    schedule = np.asarray(schedule, dtype=float)
    N = len(schedule) - 1
    return [(t, float(schedule[t]), float(envelope[t]))
            for t in range(1, N)
            if schedule[t] > envelope[t] + atol]


def format_violations(violations) -> str:
    lines = [
        f"  t={t}: requested a_t={req:.4f} nats, "
        f"max feasible {ub:.4f} nats (excess {req - ub:.4f})"
        for t, req, ub in violations
    ]
    return "\n".join(lines)
