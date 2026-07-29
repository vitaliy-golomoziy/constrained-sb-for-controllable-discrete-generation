import numpy as np
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from Distribution.DiscreteDistribution import DiscreteDistribution
from ReferenceProcess.ReferenceProcess import ReferenceProcess
from Functional.ComplexityFunctional import ComplexityFunctional
from Solver.ForwardBackward import ForwardBackward
from Solver.EntropyProjection import EntropyProjection
from Solver import Reachability


class InfeasibleScheduleError(ValueError):
    """Raised when the entropy schedule provably exceeds the reachable envelope."""


class IPFDykstraSolver:
    """
    IPF+Dykstra solver for the entropy-constrained Schrödinger Bridge problem.

    Minimises KL(P || R) over path measures subject to:
      - P_0 = p0  (fixed initial marginal)
      - P_N = pT  (fixed terminal marginal)
      - H(P_t) >= a_t  for 1 <= t <= N-1  (entropy schedule)

    Algorithm (Benamou et al. 2015)
    -------------------------------
    The path measure is represented implicitly via N+1 potentials:
        P(x_{0:N}) = R(x_{0:N}) * prod_t phi_t(x_t)
    where each phi_t : S^K -> R_+ lives in |S|^K space rather than the
    full |S|^{K(N+1)} trajectory space.  Marginals are computed via the
    forward-backward algorithm on the Markov reference.

    The constraint sets are
        C_0 = {P : P_0 = p0},   C_N = {P : P_N = pT},         (affine)
        C_t = {P : H(P_t) >= a_t},  1 <= t <= N-1.            (convex, NOT affine)

    Cyclic Bregman projections converge to the KL-projection onto the
    intersection only when every set is affine.  The entropy sets are convex
    but not affine — that is exactly why the problem is convex at all — so the
    interior projections carry Dykstra corrections z_t:

        w      = y * z_t          (re-apply the stored correction)
        y_new  = Pi_{C_t}(w)      (project)
        z_t    = w / y_new        (refresh the correction)

    Without them the iteration still converges, but to an arbitrary *feasible*
    path measure rather than the KL-minimising one: as soon as H(P_t) >= a_t,
    the projection is the identity, so every feasible P = R * prod phi_t is a
    fixed point.  The endpoint sets are affine and need no correction, so they
    keep the classical Sinkhorn rescaling.

    Each cycle:
      1. One forward-backward pass with the current potentials, giving the
         backward messages for the sweep and the marginals for the
         convergence test.
      2. A sweep t = 0..N: form P_t from (phi_t * z_t, alpha_t, beta_t),
         project onto the constraint for time t, update phi_t and z_t, and
         advance alpha incrementally.

    Advancing alpha inside the sweep (rather than recomputing the whole
    forward-backward for every t) is a pure optimisation: alpha_t depends only
    on phi_{<t}, which the sweep has already updated, and beta_t only on
    phi_{>t}, which it has not — precisely the messages a full recomputation
    would produce.

    Memory:  O(N * |S|^K)  — exponentially smaller than the direct solver.
    Per-cycle cost:  O(N * nnz(K))  forward-backward
                   + the numerical entropy-projection cost (SLSQP, K>1)

    Parameters
    ----------
    phase_space : DiscretePhaseSpace
    reference : ReferenceProcess
        Must expose ._initial (shape S,) and ._kernel (shape S,S).
    functional : ComplexityFunctional
        Used for post-solve analysis only; the entropy constraint is carried
        by the projection object.
    projection : EntropyProjection, optional
        Custom projection solver; defaults to EntropyProjection().
    use_dykstra : bool
        Apply Dykstra corrections to the interior constraints (default True).
        Setting this False recovers plain cyclic Bregman projections, which
        converge to a feasible but generally suboptimal point; it exists only
        so the difference can be measured.
    on_infeasible : {"warn", "raise", "ignore"}
        What to do when the schedule exceeds the reachable envelope
        (Solver.Reachability).  Default "warn".

    Attributes set after solve()
    ----------------------------
    n_cycles : int
    converged : bool          — whether the residual actually reached tol
    final_residual : float
    convergence_history : list of float
    schedule_violations : list of (t, requested, max_feasible)
    """

    def __init__(
        self,
        phase_space: DiscretePhaseSpace,
        reference: ReferenceProcess,
        functional: ComplexityFunctional,
        projection: EntropyProjection | None = None,
        use_dykstra: bool = True,
        on_infeasible: str = "warn",
    ):
        self.phase_space = phase_space
        self.reference = reference
        self.functional = functional
        self.use_dykstra = use_dykstra
        if on_infeasible not in ("warn", "raise", "ignore"):
            raise ValueError(f"on_infeasible must be warn/raise/ignore, "
                             f"got {on_infeasible!r}")
        self.on_infeasible = on_infeasible

        self._fb = ForwardBackward(reference._initial, reference._kernel)
        self._proj = projection if projection is not None else EntropyProjection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        p0: np.ndarray,
        pT: np.ndarray,
        schedule: np.ndarray,
        max_cycles: int = 500,
        tol: float = 1e-7,
        verbose: bool = False,
    ) -> DiscreteDistribution:
        """
        Run IPF+Dykstra and return the optimal path measure as a
        DiscreteDistribution.

        For problems where |S|^{K(N+1)} is large the materialization step
        (building the full trajectory array) dominates; use
        solve_marginals_only() instead when the full path measure is not needed.

        Parameters
        ----------
        p0, pT   : np.ndarray of shape (num_states,)
        schedule : np.ndarray of shape (T+1,) — entropy lower bounds a_t
        max_cycles : int — maximum number of full sweeps
        tol : float — stop when max L-inf change in any marginal < tol
        verbose : bool — print per-cycle convergence info

        Returns
        -------
        DiscreteDistribution — the optimal path measure μ*
        """
        self._run(p0, pT, schedule, max_cycles, tol, verbose)
        return self._build_distribution(self._log_potentials)

    def solve_marginals_only(
        self,
        p0: np.ndarray,
        pT: np.ndarray,
        schedule: np.ndarray,
        max_cycles: int = 500,
        tol: float = 1e-7,
        verbose: bool = False,
    ) -> list:
        """
        Run IPF+Dykstra and return only the T+1 marginals.

        Does not materialise the full trajectory distribution, so it is
        feasible even when |S|^{K(N+1)} is too large to store.

        Returns
        -------
        list of np.ndarray, each of shape (num_states,) — marginals P*_t.
        """
        marginals, _, _ = self._run(p0, pT, schedule, max_cycles, tol, verbose)
        return marginals

    # ------------------------------------------------------------------
    # Feasibility screening
    # ------------------------------------------------------------------

    def check_feasibility(self, p0, pT, schedule):
        """
        Compare the schedule against the reachable entropy envelope.

        Returns (violations, envelope); see Solver.Reachability.
        """
        ps = self.phase_space
        s_max = Reachability.kernel_fan_in(self.reference._kernel)
        s_out = Reachability.kernel_fan_out(self.reference._kernel)
        which = getattr(self._proj, "SCHEDULE_FUNCTIONAL", "erasure")
        envelope = Reachability.entropy_envelope(
            Reachability.joint_entropy(p0),
            Reachability.joint_entropy(pT),
            ps.num_steps,
            s_max,
            ps.n,
            ps.alphabet_size,
            functional=which,
            s_out=s_out,
        )
        return Reachability.check_schedule(schedule, envelope), envelope

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run(self, p0, pT, schedule, max_cycles, tol, verbose):
        ps = self.phase_space
        N = ps.num_steps
        S = ps.num_states
        p0 = np.asarray(p0, dtype=float)
        pT = np.asarray(pT, dtype=float)
        schedule = np.asarray(schedule, dtype=float)

        violations, envelope = self.check_feasibility(p0, pT, schedule)
        self.schedule_violations = violations
        self.envelope = envelope
        if violations and self.on_infeasible != "ignore":
            msg = ("entropy schedule exceeds the reachable envelope "
                   "(no finite-KL path can satisfy it):\n"
                   + Reachability.format_violations(violations))
            if self.on_infeasible == "raise":
                raise InfeasibleScheduleError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

        with np.errstate(divide="ignore"):
            log_p0 = np.log(p0)
            log_pT = np.log(pT)

        # Work in log-space throughout.  Interior constraint sets are convex
        # but not affine, so each carries a Dykstra correction log_z[t];
        # the affine endpoint sets do not need one.
        log_phi = [np.zeros(S) for _ in range(N + 1)]
        log_z = [np.zeros(S) for _ in range(N + 1)]

        prev_marginals = None
        history = []
        curr_marginals = None
        delta = np.inf
        cycles_done = 0

        for cycle in range(max_cycles):
            # ---- 1. messages + marginals under the current potentials ----
            log_alpha, log_beta = self._fb.messages_log(log_phi)
            curr_marginals = self._fb.combine_log(log_phi, log_alpha, log_beta)

            if prev_marginals is not None:
                delta = max(
                    np.max(np.abs(curr_marginals[t] - prev_marginals[t]))
                    for t in range(N + 1)
                )
                history.append(delta)
                if verbose:
                    print(f"  cycle {cycle:4d}  max_Δmarginal = {delta:.3e}")
                if delta < tol:
                    self._log_potentials = log_phi
                    self.n_cycles = cycles_done
                    self.converged = True
                    self.final_residual = float(delta)
                    self.convergence_history = history
                    return curr_marginals, cycles_done, history

            prev_marginals = [m.copy() for m in curr_marginals]

            # ---- 2. sweep t = 0..N, advancing alpha incrementally ----
            la = self._fb.log_R0.copy()
            for t in range(N + 1):
                interior = 0 < t < N
                trial = log_phi[t] + log_z[t] if (interior and self.use_dykstra) \
                    else log_phi[t]

                m_t = self._fb.marginal_from_parts(trial, la, log_beta[t], t)

                if t == 0:
                    log_star = log_p0
                elif t == N:
                    log_star = log_pT
                else:
                    p_star = self._proj.project(m_t, float(schedule[t]), ps)
                    with np.errstate(divide="ignore"):
                        log_star = np.log(np.maximum(p_star, 0.0))

                with np.errstate(divide="ignore", invalid="ignore"):
                    log_m = np.log(m_t)
                    log_ratio = np.where(m_t > 0.0, log_star - log_m, -np.inf)

                log_phi[t] = trial + log_ratio
                if interior and self.use_dykstra:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        log_z[t] = np.where(m_t > 0.0, log_m - log_star, 0.0)

                if t < N:
                    la = self._fb._shift(
                        self._fb.log_forward_step(la + log_phi[t]), f"alpha_{t+1}"
                    )
            cycles_done = cycle + 1

        # exhausted max_cycles — report the final state honestly
        log_alpha, log_beta = self._fb.messages_log(log_phi)
        curr_marginals = self._fb.combine_log(log_phi, log_alpha, log_beta)
        if prev_marginals is not None:
            delta = max(np.max(np.abs(curr_marginals[t] - prev_marginals[t]))
                        for t in range(N + 1))
            history.append(delta)
        self._log_potentials = log_phi
        self.n_cycles = cycles_done
        self.converged = bool(delta < tol)
        self.final_residual = float(delta)
        self.convergence_history = history
        if not self.converged:
            warnings.warn(
                f"IPF did not converge: after {cycles_done} cycles the max "
                f"marginal change is {delta:.3e} (tol={tol:.1e}).",
                RuntimeWarning, stacklevel=3,
            )
        return curr_marginals, cycles_done, history

    # ------------------------------------------------------------------
    # Materialise full path measure from potentials
    # ------------------------------------------------------------------

    def _build_distribution(self, log_potentials: list) -> DiscreteDistribution:
        """
        Construct P(x_{0:N}) = R(x_{0:N}) * prod_t phi_t(x_t).

        Accumulated in log-space to avoid overflow.  Only feasible when
        the trajectory space |S|^{N+1} fits in memory (toy problems).

        Structural zeros of the reference are preserved as exact zeros rather
        than floored at 1e-300: a floor leaks mass onto trajectories that R
        forbids, which makes KL(P || R) evaluate to +inf.

        The tensor is indexed in C-order: P[s_0, s_1, ..., s_N] matches
        DiscreteDistribution's encode_trajectory convention.
        """
        K = self.reference._kernel
        try:
            import scipy.sparse
            if scipy.sparse.issparse(K):
                K = K.toarray()
        except ImportError:
            pass
        K = np.asarray(K, dtype=float)

        with np.errstate(divide="ignore"):
            log_K = np.log(K)
            log_P = np.log(self.reference._initial) + log_potentials[0]
        for t in range(1, self.phase_space.num_marginals):
            log_P = log_P[..., np.newaxis] + log_K + log_potentials[t]

        finite = np.isfinite(log_P)
        if not np.any(finite):
            raise ValueError("path measure has no support under the reference")
        log_P = log_P - log_P[finite].max()
        probs = np.where(np.isfinite(log_P), np.exp(log_P), 0.0).ravel()
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
        return DiscreteDistribution(self.phase_space, probs)
