import numpy as np
from itertools import product


class DiscretePhaseSpace:
    """
    Phase space for the toy SB problem.

    A state at one time step is a configuration x in A^n (n tokens, each in {0,...,A-1}).
    A trajectory is a sequence of T+1 configurations: (x_0, x_1, ..., x_T).

    Parameters
    ----------
    n : int
        Number of tokens per configuration.
    alphabet_size : int
        Size of the alphabet |A|.
    num_steps : int
        Number of time steps T. There are T+1 marginals: t=0,1,...,T.
    """

    def __init__(self, n: int, alphabet_size: int, num_steps: int):
        self.n = n
        self.alphabet_size = alphabet_size
        self.num_steps = num_steps  # T

    @property
    def num_marginals(self) -> int:
        """T+1: number of time checkpoints."""
        return self.num_steps + 1

    @property
    def num_states(self) -> int:
        """Number of configurations at a single time step: |A|^n."""
        return self.alphabet_size ** self.n

    @property
    def num_trajectories(self) -> int:
        """Number of full trajectories: |A|^(n*(T+1))."""
        return self.alphabet_size ** (self.n * self.num_marginals)

    # ------------------------------------------------------------------
    # Configuration encode / decode  (single time step)
    # ------------------------------------------------------------------

    def encode_configuration(self, config) -> int:
        """Map a configuration (length-n tuple/array over A) to a flat index."""
        config = np.asarray(config, dtype=int)
        assert config.shape == (self.n,), f"Expected length-{self.n} config"
        idx = 0
        for tok in config:
            idx = idx * self.alphabet_size + int(tok)
        return idx

    def decode_configuration(self, index: int) -> np.ndarray:
        """Inverse of encode_configuration. Returns length-n array."""
        tokens = []
        for _ in range(self.n):
            tokens.append(index % self.alphabet_size)
            index //= self.alphabet_size
        return np.array(tokens[::-1], dtype=int)

    # ------------------------------------------------------------------
    # Trajectory encode / decode  (T+1 configurations)
    # ------------------------------------------------------------------

    def encode_trajectory(self, trajectory) -> int:
        """
        Map a trajectory — sequence of T+1 configurations — to a flat index.

        Parameters
        ----------
        trajectory : array-like of shape (T+1, n)
        """
        trajectory = np.asarray(trajectory, dtype=int)
        assert trajectory.shape == (self.num_marginals, self.n), (
            f"Expected shape ({self.num_marginals}, {self.n})"
        )
        idx = 0
        for config in trajectory:
            idx = idx * self.num_states + self.encode_configuration(config)
        return idx

    def decode_trajectory(self, index: int) -> np.ndarray:
        """
        Inverse of encode_trajectory.

        Returns
        -------
        np.ndarray of shape (T+1, n)
        """
        configs = []
        for _ in range(self.num_marginals):
            configs.append(self.decode_configuration(index % self.num_states))
            index //= self.num_states
        return np.array(configs[::-1], dtype=int)

    # ------------------------------------------------------------------
    # Enumeration helpers
    # ------------------------------------------------------------------

    def all_configurations(self) -> np.ndarray:
        """
        All configurations in lexicographic order.

        Returns
        -------
        np.ndarray of shape (num_states, n)
        """
        return np.array(
            list(product(range(self.alphabet_size), repeat=self.n)), dtype=int
        )

    def all_trajectories(self) -> np.ndarray:
        """
        All trajectories in lexicographic order.

        Returns
        -------
        np.ndarray of shape (num_trajectories, T+1, n)
        """
        configs = self.all_configurations()
        traj_indices = product(range(self.num_states), repeat=self.num_marginals)
        return np.array(
            [np.stack([configs[i] for i in traj]) for traj in traj_indices], dtype=int
        )
