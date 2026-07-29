import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace


class DiscreteDistribution:
    """
    A probability distribution over full trajectories (x_0, ..., x_T).

    The distribution is stored as a flat array `probs` of length num_trajectories,
    indexed by encode_trajectory(). All methods work by reshaping / summing axes
    of the equivalent tensor of shape (num_states,) * (T+1).

    Parameters
    ----------
    phase_space : DiscretePhaseSpace
    probs : array-like of length num_trajectories
        Must be non-negative and sum to 1.
    """

    def __init__(self, phase_space: DiscretePhaseSpace, probs):
        self.phase_space = phase_space
        probs = np.asarray(probs, dtype=float)
        assert probs.shape == (phase_space.num_trajectories,), (
            f"probs must have length {phase_space.num_trajectories}"
        )
        assert np.all(probs >= -1e-12), "probs must be non-negative"
        assert abs(probs.sum() - 1.0) < 1e-9, "probs must sum to 1"
        probs = np.clip(probs, 0, None)
        self._probs = probs / probs.sum()

    # ------------------------------------------------------------------
    # Raw access
    # ------------------------------------------------------------------

    def full_probabilities(self) -> np.ndarray:
        """Flat array of length num_trajectories."""
        return self._probs.copy()

    def _tensor(self) -> np.ndarray:
        """Joint table reshaped as (num_states,)^(T+1)."""
        S = self.phase_space.num_states
        M = self.phase_space.num_marginals
        return self._probs.reshape([S] * M)

    # ------------------------------------------------------------------
    # Marginals
    # ------------------------------------------------------------------

    def time_marginal(self, t: int) -> np.ndarray:
        """
        Marginal distribution over configurations at time t.

        Returns
        -------
        np.ndarray of shape (num_states,)
            probs[s] = P(X_t = s)
        """
        assert 0 <= t < self.phase_space.num_marginals
        tensor = self._tensor()
        axes_to_sum = tuple(i for i in range(self.phase_space.num_marginals) if i != t)
        return tensor.sum(axis=axes_to_sum)

    def joint_time_marginal(self, time_indices) -> np.ndarray:
        """
        Joint marginal over a subset of time steps.

        Parameters
        ----------
        time_indices : sequence of ints, e.g. [0, 2]

        Returns
        -------
        np.ndarray of shape (num_states,) * len(time_indices)
            Axes are in the order given by time_indices.
        """
        time_indices = list(time_indices)
        M = self.phase_space.num_marginals
        tensor = self._tensor()
        axes_to_sum = tuple(i for i in range(M) if i not in time_indices)
        result = tensor.sum(axis=axes_to_sum)
        # axes of result correspond to time_indices sorted; reorder if needed
        sorted_kept = sorted(range(M), key=lambda i: i not in time_indices)
        kept_in_order = [i for i in range(M) if i in time_indices]
        desired_order = [kept_in_order.index(t) for t in time_indices]
        return np.transpose(result, desired_order)

    # ------------------------------------------------------------------
    # Entropy functional  E = (1/n) * sum_k H(X_k | X_{-k})
    # ------------------------------------------------------------------

    def conditional_entropy(self) -> float:
        """
        Average conditional entropy E = (1/n) * sum_{k=1}^{n} H(X_k | X_{-k}).

        Here X = (X_1,...,X_n) is the configuration at a *single* time step,
        and this expectation is taken under the time-averaged configuration marginal.

        The formula used is H(X_k | X_{-k}) = H(X) - H(X_{-k}).
        We compute this on the marginal over a single time step.

        Note: Since E is an *average* over tokens, this method returns E
        evaluated on the time-marginal average (mean over t).
        Use `conditional_entropy_at(t)` for a specific time step.
        """
        M = self.phase_space.num_marginals
        return np.mean([self.conditional_entropy_at(t) for t in range(M)])

    def conditional_entropy_at(self, t: int) -> float:
        """
        E_t = (1/n) * sum_{k=1}^{n} H(X_{t,k} | X_{t,-k})
        evaluated on the marginal distribution at time t.

        Uses H(X_{t,k} | X_{t,-k}) = H(X_t) - H(X_{t,-k}).
        """
        ps = self.phase_space
        marg = self.time_marginal(t)  # shape (num_states,)
        # Reshape to (A, A, ..., A) with n axes
        config_tensor = marg.reshape([ps.alphabet_size] * ps.n)

        def _entropy(p_flat):
            p = p_flat[p_flat > 0]
            return -np.sum(p * np.log(p))

        h_full = _entropy(config_tensor.ravel())

        total = 0.0
        for k in range(ps.n):
            # marginal over all tokens except k: sum over axis k
            p_minus_k = config_tensor.sum(axis=k).ravel()
            h_minus_k = _entropy(p_minus_k)
            total += h_full - h_minus_k

        return total / ps.n
