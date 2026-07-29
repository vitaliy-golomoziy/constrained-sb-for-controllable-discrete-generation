import numpy as np
import cvxpy as cp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from Distribution.DiscreteDistribution import DiscreteDistribution
from ReferenceProcess.ReferenceProcess import ReferenceProcess
from Functional.ComplexityFunctional import ComplexityFunctional
from Formulation.EntropyFormulation import EntropyFormulation


class SBProblem:
    """
    Entropy-constrained Schrödinger Bridge problem.

    Minimises KL(μ || R) over path measures μ subject to:
      - fixed endpoint marginals:  μ_0 = p0,  μ_T = pT
      - entropy schedule:          E_t(μ) ~ a_t  (exact form depends on formulation)

    The complexity functional E_t and the constraint encoding are injected,
    so swapping either requires no changes to this class.

    Parameters
    ----------
    phase_space : DiscretePhaseSpace
    reference : ReferenceProcess
    complexity_functional : ComplexityFunctional
        Defines E_t for CCCP gradient computation and post-solve analysis.
        Note: the CVXPY constraint expression is always conditional entropy
        (hardcoded via rel_entr for DCP compliance). True CVXPY pluggability
        would require subclasses to supply a cp.Expression — future work.
    entropy_formulation : EntropyFormulation
        Encodes how the schedule is enforced (hard / hinge / squared-CCCP).
    """

    def __init__(
        self,
        phase_space: DiscretePhaseSpace,
        reference: ReferenceProcess,
        complexity_functional: ComplexityFunctional,
        entropy_formulation: EntropyFormulation,
    ):
        self.phase_space = phase_space
        self.reference = reference
        self.complexity_functional = complexity_functional
        self.entropy_formulation = entropy_formulation

        R_probs = reference.joint_distribution().full_probabilities()
        self._support = R_probs > 0
        self._log_R = np.where(self._support, np.log(np.maximum(R_probs, 1e-300)), 0.0)

        self._M = self._build_marginal_matrices()
        self._S = self._build_token_sum_matrices()
        self._B = self._build_broadcast_matrices()

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _build_marginal_matrices(self) -> list:
        """M[t]: (num_states, num_trajectories) — extracts the t-step marginal."""
        ps = self.phase_space
        Ms = [np.zeros((ps.num_states, ps.num_trajectories)) for _ in range(ps.num_marginals)]
        for i, traj in enumerate(ps.all_trajectories()):
            for t in range(ps.num_marginals):
                Ms[t][ps.encode_configuration(traj[t]), i] = 1.0
        return Ms

    def _build_token_sum_matrices(self) -> list:
        """S[k]: (A^(n-1), A^n) — sums out token k from a configuration distribution."""
        ps = self.phase_space
        A, n = ps.alphabet_size, ps.n
        Ss = []
        for k in range(n):
            n_reduced = A ** (n - 1)
            S_k = np.zeros((n_reduced, ps.num_states))
            for s, config in enumerate(ps.all_configurations()):
                reduced = np.delete(config, k)
                j = int(np.ravel_multi_index(reduced, [A] * (n - 1))) if n > 1 else 0
                S_k[j, s] = 1.0
            Ss.append(S_k)
        return Ss

    def _build_broadcast_matrices(self) -> list:
        """
        B[k]: (num_states, num_states) — (B[k] @ μ_t)[s] = p(x_{-k}^s).

        For each configuration s and token k, row s of B[k] is the row of S[k]
        that corresponds to s's reduced index. This lets us write the conditional
        entropy as -cp.sum(cp.rel_entr(μ_t, B[k] @ μ_t)), which CVXPY recognises
        as concave (jointly convex rel_entr negated).
        """
        Bs = []
        for k in range(self.phase_space.n):
            S_k = self._S[k]  # (A^(n-1), num_states)
            B_k = np.zeros((self.phase_space.num_states, self.phase_space.num_states))
            for s in range(self.phase_space.num_states):
                j = int(np.argmax(S_k[:, s]))  # unique j with S_k[j, s] = 1
                B_k[s, :] = S_k[j, :]
            Bs.append(B_k)
        return Bs

    # ------------------------------------------------------------------
    # CVXPY expression builders
    # ------------------------------------------------------------------

    def _kl_objective(self, mu: cp.Variable) -> cp.Expression:
        """KL(μ || R) = -H(μ) - log_R @ μ  (convex in μ)."""
        return -cp.sum(cp.entr(mu)) - self._log_R @ mu

    def _complexity_expr(self, mu: cp.Variable, t: int) -> cp.Expression:
        """
        CVXPY expression for E_t = (1/n) Σ_k H(X_{t,k} | X_{t,-k}).

        Uses H(X_k | X_{-k}) = -Σ_s rel_entr(μ_t[s], p(x_{-k}^s))
                               = -cp.sum(cp.rel_entr(μ_t, B[k] @ μ_t))

        rel_entr is jointly convex, so its negation is concave, and CVXPY
        recognises the full expression as concave — DCP-compliant as a
        constraint lower bound or as the target of a hinge penalty.
        """
        ps = self.phase_space
        mu_t = self._M[t] @ mu
        cond_entropies = sum(
            -cp.sum(cp.rel_entr(mu_t, self._B[k] @ mu_t)) for k in range(ps.n)
        )
        return cond_entropies / ps.n

    def _base_constraints(self, mu: cp.Variable, p0: np.ndarray, pT: np.ndarray) -> list:
        constraints = [
            cp.sum(mu) == 1,
            self._M[0] @ mu == p0,
            self._M[self.phase_space.num_steps] @ mu == pT,
        ]
        zero_idx = np.where(~self._support)[0]
        if zero_idx.size:
            constraints.append(mu[zero_idx] == 0)
        return constraints

    # ------------------------------------------------------------------
    # Gradient of E_t (used by CCCP loop)
    # ------------------------------------------------------------------

    def _complexity_gradient(self, mu_val: np.ndarray, t: int) -> np.ndarray:
        """
        Gradient ∂E_t/∂μ at a given point mu_val (numpy).

        ∂E_t/∂μ_t[s] = (1/n) Σ_k log(μ_{t,-k}[s_{-k}] / μ_t[s])

        This equals -(1/n) Σ_k log p(x_k | x_{-k}) — the negative average
        conditional log-probability under the current distribution.
        """
        ps = self.phase_space
        A, n = ps.alphabet_size, ps.n
        mu_t = self._M[t] @ mu_val
        config_tensor = mu_t.reshape([A] * n)

        grad_mu_t = np.zeros(ps.num_states)
        for s, config in enumerate(ps.all_configurations()):
            total = 0.0
            for k in range(n):
                mu_minus_k = config_tensor.sum(axis=k)
                idx_mk = tuple(int(config[j]) for j in range(n) if j != k)
                p_mk = mu_minus_k[idx_mk] if n > 1 else 1.0
                if mu_t[s] > 1e-300 and p_mk > 1e-300:
                    total += np.log(p_mk / mu_t[s])
            grad_mu_t[s] = total / n

        return self._M[t].T @ grad_mu_t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, p0: np.ndarray, pT: np.ndarray, schedule: np.ndarray) -> cp.Problem:
        """
        Assemble and return the CVXPY problem (non-CCCP formulations only).

        Parameters
        ----------
        p0 : np.ndarray of shape (num_states,)  — initial marginal
        pT : np.ndarray of shape (num_states,)  — terminal marginal
        schedule : np.ndarray of shape (T+1,)   — entropy lower bounds a_t
        """
        if self.entropy_formulation.requires_cccp:
            raise ValueError(
                "This formulation requires CCCP; call solve() instead of build()."
            )
        p0, pT, schedule = map(np.asarray, (p0, pT, schedule))

        mu = cp.Variable(self.phase_space.num_trajectories, nonneg=True)
        complexity_exprs = [
            self._complexity_expr(mu, t) for t in range(self.phase_space.num_marginals)
        ]

        penalty = self.entropy_formulation.objective_term(complexity_exprs, schedule)
        form_constraints = self.entropy_formulation.extra_constraints(complexity_exprs, schedule)

        objective = cp.Minimize(
            self._kl_objective(mu) + (penalty if not isinstance(penalty, int) else 0)
        )
        constraints = self._base_constraints(mu, p0, pT) + form_constraints
        return cp.Problem(objective, constraints)

    def solve(
        self,
        p0: np.ndarray,
        pT: np.ndarray,
        schedule: np.ndarray,
        cccp_max_iter: int = 50,
        cccp_tol: float = 1e-6,
        **solver_kwargs,
    ) -> DiscreteDistribution:
        """
        Solve the SB problem and return the optimal path measure μ*.

        For hard-constraint and hinge formulations this is a single CVXPY solve.
        For squared-penalty (requires_cccp=True) this runs the CCCP outer loop:
          1. Initialise μ^(0) = uniform over R's support.
          2. Linearise E_t around μ^(k): L_t(μ) = E_t(μ^k) + ∇E_t(μ^k)^T (μ - μ^k).
          3. Solve the convex sub-problem with (L_t - a_t)² in the objective.
          4. Repeat until ||μ^(k+1) - μ^(k)||_2 < cccp_tol.

        Parameters
        ----------
        p0, pT : np.ndarray of shape (num_states,)
        schedule : np.ndarray of shape (T+1,)
        cccp_max_iter : int
        cccp_tol : float
        **solver_kwargs : passed to cp.Problem.solve()
        """
        p0 = np.asarray(p0, dtype=float)
        pT = np.asarray(pT, dtype=float)
        schedule = np.asarray(schedule, dtype=float)

        if not self.entropy_formulation.requires_cccp:
            return self._solve_direct(p0, pT, schedule, **solver_kwargs)
        else:
            return self._solve_cccp(p0, pT, schedule, cccp_max_iter, cccp_tol, **solver_kwargs)

    def _solve_direct(self, p0, pT, schedule, **solver_kwargs) -> DiscreteDistribution:
        prob = self.build(p0, pT, schedule)
        prob.solve(**solver_kwargs)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Solver status: {prob.status}")
        return self._extract_solution(prob.variables()[0].value)

    def _solve_cccp(self, p0, pT, schedule, max_iter, tol, **solver_kwargs) -> DiscreteDistribution:
        ps = self.phase_space

        # Initialise at uniform over support of R
        mu_val = np.where(self._support, 1.0, 0.0).astype(float)
        mu_val /= mu_val.sum()

        for _ in range(max_iter):
            mu_var = cp.Variable(ps.num_trajectories, nonneg=True)

            # Linearised complexity expressions
            linear_exprs = []
            for t in range(ps.num_marginals):
                E_t = self.complexity_functional(self._M[t] @ mu_val, ps)
                grad_t = self._complexity_gradient(mu_val, t)
                linear_exprs.append(E_t + grad_t @ (mu_var - mu_val))

            penalty = self.entropy_formulation.objective_term(linear_exprs, schedule)
            constraints = self._base_constraints(mu_var, p0, pT)
            prob = cp.Problem(cp.Minimize(self._kl_objective(mu_var) + penalty), constraints)
            prob.solve(**solver_kwargs)

            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"CCCP sub-problem status: {prob.status}")

            mu_new = np.clip(mu_var.value, 0, None)
            mu_new /= mu_new.sum()

            if np.linalg.norm(mu_new - mu_val) < tol:
                mu_val = mu_new
                break
            mu_val = mu_new

        return self._extract_solution(mu_val)

    def _extract_solution(self, raw: np.ndarray) -> DiscreteDistribution:
        probs = np.clip(raw, 0, None)
        probs[~self._support] = 0.0  # zero out solver residuals outside R's support
        probs /= probs.sum()
        return DiscreteDistribution(self.phase_space, probs)
