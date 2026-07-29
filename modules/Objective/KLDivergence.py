import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Distribution.DiscreteDistribution import DiscreteDistribution
from ReferenceProcess.ReferenceProcess import ReferenceProcess


class KLDivergence:
    """
    KL divergence computations for the static and dynamic SB problems.

    All methods are stateless class methods built on the base `kl` computation.
    To extend to other divergences (f-divergences, etc.), subclass and override `kl`.

    Static vs dynamic:
      - Dynamic SB objective:  KL(μ || R) over the full path space
      - Static SB objective:   KL(π || R_{t0,t1}) over a two-time coupling
      - Marginal diagnostic:   KL(μ_t || ν_t) at a single time step
    """

    @staticmethod
    def kl(p: np.ndarray, q: np.ndarray) -> float:
        """
        KL(p || q) for two probability arrays of the same shape.

        Returns +inf if q[i] = 0 for any i where p[i] > 0.

        Parameters
        ----------
        p, q : np.ndarray  (will be flattened; must have the same number of elements)
        """
        p = np.asarray(p, dtype=float).ravel()
        q = np.asarray(q, dtype=float).ravel()
        assert p.shape == q.shape, "p and q must have the same number of elements"
        assert abs(p.sum() - 1.0) < 1e-9, "p must be normalised"
        assert abs(q.sum() - 1.0) < 1e-9, "q must be normalised"

        support_violation = (p > 0) & (q == 0)
        if np.any(support_violation):
            return np.inf

        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    @classmethod
    def dynamic(cls, mu: DiscreteDistribution, reference: ReferenceProcess) -> float:
        """
        Dynamic SB objective: KL(μ || R) over the full trajectory space.

        KL(μ || R) = sum_{trajectories} μ(traj) * log(μ(traj) / R(traj))

        Parameters
        ----------
        mu : DiscreteDistribution
            Path measure to evaluate.
        reference : ReferenceProcess
            Reference process R.
        """
        p = mu.full_probabilities()
        q = reference.joint_distribution().full_probabilities()
        return cls.kl(p, q)

    @classmethod
    def static(cls, coupling: np.ndarray, reference_coupling: np.ndarray) -> float:
        """
        Static SB objective: KL(π || R_{t0,t1}) between two coupling matrices.

        Both inputs are probability arrays over pairs of configurations,
        e.g. obtained via DiscreteDistribution.joint_time_marginal([t0, t1]).

        Parameters
        ----------
        coupling : np.ndarray
            The transport plan π, shape (num_states, num_states) or flat.
        reference_coupling : np.ndarray
            The reference marginal R_{t0,t1}, same shape.
        """
        return cls.kl(
            np.asarray(coupling, dtype=float).ravel(),
            np.asarray(reference_coupling, dtype=float).ravel(),
        )

    @classmethod
    def marginal(cls, mu: DiscreteDistribution, nu: DiscreteDistribution, t: int) -> float:
        """
        KL(μ_t || ν_t) between the time-t marginals of two distributions.

        Useful as a diagnostic: how far is μ_t from the reference marginal R_t?

        Parameters
        ----------
        mu : DiscreteDistribution
        nu : DiscreteDistribution
        t : int
            Time index in {0, ..., T}.
        """
        return cls.kl(mu.time_marginal(t), nu.time_marginal(t))
