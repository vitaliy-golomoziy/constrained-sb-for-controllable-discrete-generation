import numpy as np
from abc import ABC, abstractmethod
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from Distribution.DiscreteDistribution import DiscreteDistribution


class ReferenceProcess(ABC):
    """
    Abstract base class for Markov reference processes over discrete trajectories.

    A reference process R over trajectories (x_0,...,x_T) factors as:
        R(x_0,...,x_T) = initial(x_0) * prod_{t=0}^{T-1} K(x_t, x_{t+1})

    Subclasses implement _build_kernel() to define their transition dynamics.
    The initial distribution, trajectory scoring, and joint distribution
    construction are shared here.

    Parameters
    ----------
    phase_space : DiscretePhaseSpace
    initial_probs : array-like of shape (num_states,)
        Initial distribution over configurations at t=0.
    """

    def __init__(self, phase_space: DiscretePhaseSpace, initial_probs):
        self.phase_space = phase_space
        self._initial = self._validated_probs(initial_probs, phase_space.num_states)
        self._kernel = self._build_kernel()

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_kernel(self) -> np.ndarray:
        """
        Build and return the transition matrix K of shape (num_states, num_states).

        K[s, s'] = P(X_{t+1} = s' | X_t = s).
        Must have non-negative entries with rows summing to 1.
        Called once during __init__; result is stored as self._kernel.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validated_probs(probs, expected_len: int) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        assert p.shape == (expected_len,), f"probs must have length {expected_len}"
        assert np.all(p >= -1e-12), "probs must be non-negative"
        assert abs(p.sum() - 1.0) < 1e-9, "probs must sum to 1"
        p = np.clip(p, 0, None)
        return p / p.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transition_kernel(self) -> np.ndarray:
        """
        Transition matrix K of shape (num_states, num_states).

        K[s, s'] = P(X_{t+1} = s' | X_t = s).
        """
        return self._kernel.copy()

    def trajectory_log_prob(self, trajectory) -> float:
        """
        Log-probability of a trajectory under this reference process.

        R(x_0,...,x_T) = initial(x_0) * prod_{t=0}^{T-1} K(x_t, x_{t+1})

        Parameters
        ----------
        trajectory : array-like of shape (T+1, n)

        Returns
        -------
        float  (-inf for zero-probability trajectories)
        """
        ps = self.phase_space
        trajectory = np.asarray(trajectory, dtype=int)
        assert trajectory.shape == (ps.num_marginals, ps.n)

        s0 = ps.encode_configuration(trajectory[0])
        p0 = self._initial[s0]
        if p0 == 0:
            return -np.inf
        log_p = np.log(p0)

        for t in range(ps.num_steps):
            s = ps.encode_configuration(trajectory[t])
            s_next = ps.encode_configuration(trajectory[t + 1])
            k = self._kernel[s, s_next]
            if k == 0:
                return -np.inf
            log_p += np.log(k)

        return float(log_p)

    def trajectory_prob(self, trajectory) -> float:
        """Probability of a trajectory under this reference process."""
        return float(np.exp(self.trajectory_log_prob(trajectory)))

    def joint_distribution(self) -> DiscreteDistribution:
        """
        Full joint distribution R over all trajectories.

        Built iteratively: R tensor grows one time axis per step via
        R[..., s_t, s_{t+1}] = R[..., s_t] * K[s_t, s_{t+1}].

        Returns
        -------
        DiscreteDistribution with probs[i] = R(trajectory_i)
        """
        ps = self.phase_space
        R = self._initial.copy()

        for _ in range(ps.num_steps):
            R = R[..., np.newaxis] * self._kernel

        probs = np.clip(R.ravel(), 0, None)
        probs /= probs.sum()
        return DiscreteDistribution(ps, probs)
