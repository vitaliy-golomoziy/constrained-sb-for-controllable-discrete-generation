import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from Functional.ComplexityFunctional import ComplexityFunctional


class ConditionalEntropyFunctional(ComplexityFunctional):
    """
    Average conditional entropy complexity functional.

    E(p) = (1/n) * sum_{k=1}^{n} H(X_k | X_{-k})

    Using the identity H(X_k | X_{-k}) = H(X) - H(X_{-k}), this is:

    E(p) = (1/n) * sum_k [ H(X) - H(X_{-k}) ]

    where H(X) is the joint entropy of the configuration and H(X_{-k}) is
    the entropy of the (n-1)-token marginal obtained by summing out token k.

    This is the primary complexity measure for the SB entropy constraint E_t >= a_t.
    It is concave in p, so the constraint set {p : E(p) >= a} is convex.
    """

    def __call__(self, config_probs: np.ndarray, phase_space: DiscretePhaseSpace) -> float:
        """
        Parameters
        ----------
        config_probs : np.ndarray of shape (num_states,)
        phase_space : DiscretePhaseSpace

        Returns
        -------
        float  (>= 0)
        """
        config_probs = np.asarray(config_probs, dtype=float)
        assert config_probs.shape == (phase_space.num_states,)

        A = phase_space.alphabet_size
        n = phase_space.n
        config_tensor = config_probs.reshape([A] * n)

        def _entropy(p_flat):
            p = p_flat[p_flat > 0]
            return float(-np.sum(p * np.log(p)))

        h_full = _entropy(config_tensor.ravel())

        total = 0.0
        for k in range(n):
            p_minus_k = config_tensor.sum(axis=k).ravel()
            total += h_full - _entropy(p_minus_k)

        return total / n
