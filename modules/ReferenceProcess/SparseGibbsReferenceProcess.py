import numpy as np
import scipy.sparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace


class SparseGibbsReferenceProcess:
    """
    Gibbs sampler reference process with uniform stationary, stored as a
    sparse transition matrix.

    For a K-token binary configuration (alphabet size 2), the averaged Gibbs
    kernel with stationary π = Uniform({0,1}^K) has the closed form:

        K(x, x)  = 1/2          (self-loop: chosen token keeps its value)
        K(x, y)  = 1/(2K)       if Hamming(x, y) = 1  (one token flips)
        K(x, y)  = 0            otherwise

    Each row sums to 1: 1/2 + K × 1/(2K) = 1.

    The stationary distribution is Uniform, which coincides with the
    initial distribution stored in _initial.

    This class exposes only _initial and _kernel — the two attributes
    consumed by IPFDykstraSolver and ForwardBackward.

    Parameters
    ----------
    phase_space : DiscretePhaseSpace
        Must have alphabet_size == 2.
    """

    def __init__(self, phase_space: DiscretePhaseSpace):
        assert phase_space.alphabet_size == 2, "SparseGibbsReferenceProcess requires binary alphabet"
        self.phase_space = phase_space

        K = phase_space.n
        S = phase_space.num_states  # 2^K

        self._initial = np.full(S, 1.0 / S)
        self._kernel = self._build_sparse_kernel(K, S)

    @staticmethod
    def _build_sparse_kernel(K: int, S: int):
        """
        Build the sparse Gibbs transition matrix analytically.

        Token k occupies bit position (K-1-k) of the state integer
        (big-endian, matching DiscretePhaseSpace.encode_configuration).
        Flipping token k = XOR with 2^(K-1-k).
        """
        s_idx = np.arange(S, dtype=np.int64)

        # --- self-loops ---
        row_self = s_idx
        col_self = s_idx
        data_self = np.full(S, 0.5)

        # --- one-flip neighbours ---
        # flip_masks[k] = 2^(K-1-k) for k = 0, ..., K-1
        flip_masks = np.int64(1) << (np.arange(K - 1, -1, -1, dtype=np.int64))

        # broadcast: (S, 1) XOR (1, K) → (S, K)
        neighbors = s_idx[:, None] ^ flip_masks[None, :]   # shape (S, K)

        row_flip = np.repeat(s_idx, K)          # (S*K,)
        col_flip = neighbors.ravel()             # (S*K,)
        data_flip = np.full(S * K, 1.0 / (2.0 * K))

        rows = np.concatenate([row_self, row_flip])
        cols = np.concatenate([col_self, col_flip])
        data = np.concatenate([data_self, data_flip])

        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(S, S), dtype=np.float64)
