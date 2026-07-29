import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from ReferenceProcess.ReferenceProcess import ReferenceProcess


class GibbsReferenceProcess(ReferenceProcess):
    """
    Gibbs sampler reference process for the SB toy problem.

    At each step, one token index k is chosen uniformly at random and
    resampled as X_k ~ p_T(X_k | X_{-k}), leaving X_{-k} unchanged.
    The stationary distribution is p_T and the process is reversible
    (detailed balance holds w.r.t. p_T).

    Parameters
    ----------
    phase_space : DiscretePhaseSpace
    stationary_probs : array-like of shape (num_states,)
        Target distribution p_T. Used both as the initial distribution
        and to define the conditional resampling kernels.
    """

    def __init__(self, phase_space: DiscretePhaseSpace, stationary_probs):
        self._stationary_probs = stationary_probs  # stored before super().__init__
        super().__init__(phase_space, stationary_probs)

    def _stationary_tensor(self) -> np.ndarray:
        """p_T as tensor of shape (A,)*n."""
        ps = self.phase_space
        return self._initial.reshape([ps.alphabet_size] * ps.n)

    def _build_kernel(self) -> np.ndarray:
        """
        Averaged Gibbs kernel K of shape (num_states, num_states).

        K[s, s'] = (1/n) * sum_k  p_T(x'_k | x_{-k}) * 1[x'_{-k} = x_{-k}]
        """
        ps = self.phase_space
        p_tensor = self._stationary_tensor()
        S = ps.num_states
        A = ps.alphabet_size
        n = ps.n
        K = np.zeros((S, S))

        all_configs = ps.all_configurations()

        for s, x in enumerate(all_configs):
            for k in range(n):
                p_minus_k = p_tensor.sum(axis=k)
                idx_minus_k = tuple(int(x[j]) for j in range(n) if j != k)
                p_xmk = p_minus_k[idx_minus_k]

                for a in range(A):
                    x_prime = x.copy()
                    x_prime[k] = a
                    s_prime = ps.encode_configuration(x_prime)

                    if p_xmk > 0:
                        idx_prime = tuple(int(x_prime[j]) for j in range(n))
                        p_joint = p_tensor[idx_prime]
                        K[s, s_prime] += p_joint / p_xmk
                    else:
                        K[s, s_prime] += 1.0 / A

        K /= n
        return K
