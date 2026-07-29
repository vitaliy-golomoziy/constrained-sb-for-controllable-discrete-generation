import numpy as np
from abc import ABC, abstractmethod
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace


class ComplexityFunctional(ABC):
    """
    Abstract complexity functional E(p) for a configuration distribution p.

    Operates on the marginal at a single time step, returning a non-negative
    scalar that measures how "structured" or "complex" the distribution is.
    The SB entropy constraint is E_t >= a_t, where E_t = self(dist.time_marginal(t)).

    Subclasses encode different notions of complexity, e.g.:
      - ConditionalEntropyFunctional: E = (1/n) sum_k H(X_k | X_{-k})
      - DeltaEntropyFunctional:       E = (1/n) sum_k H(X_k | X_{N(k)})  [spatial window]

    Swapping the functional requires no changes to the SB solver — just pass a
    different ComplexityFunctional instance.
    """

    @abstractmethod
    def __call__(self, config_probs: np.ndarray, phase_space: DiscretePhaseSpace) -> float:
        """
        Evaluate the complexity of a single-time configuration distribution.

        Parameters
        ----------
        config_probs : np.ndarray of shape (num_states,)
            Marginal distribution over A^n at one time step.
        phase_space : DiscretePhaseSpace

        Returns
        -------
        float  (>= 0)
        """

    def trajectory_complexities(self, dist) -> np.ndarray:
        """
        Evaluate E_t for every t in {0, ..., T}.

        Parameters
        ----------
        dist : DiscreteDistribution

        Returns
        -------
        np.ndarray of shape (T+1,)
            complexities[t] = self(dist.time_marginal(t), dist.phase_space)
        """
        ps = dist.phase_space
        return np.array([
            self(dist.time_marginal(t), ps)
            for t in range(ps.num_marginals)
        ])

    def constraint_slack(self, dist, schedule: np.ndarray) -> np.ndarray:
        """
        Compute E_t - a_t for each t.  Negative values indicate violated constraints.

        Parameters
        ----------
        dist : DiscreteDistribution
        schedule : np.ndarray of shape (T+1,)
            Required lower bounds a_t.

        Returns
        -------
        np.ndarray of shape (T+1,)
        """
        schedule = np.asarray(schedule, dtype=float)
        assert schedule.shape == (dist.phase_space.num_marginals,), (
            f"schedule must have length {dist.phase_space.num_marginals}"
        )
        return self.trajectory_complexities(dist) - schedule
