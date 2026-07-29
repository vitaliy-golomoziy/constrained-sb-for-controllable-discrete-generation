import copy
import torch
import torch.nn.functional as F
import numpy as np

from modules.MaskedTokenModel import MaskedTokenModel


class SBPath:
    """
    Holds T+1 masked token models used by the amortized MNIST heuristic.

    Each M_t is a MaskedTokenModel.  A per-model temperature > 1 flattens
    the conditional distributions (raises entropy); < 1 sharpens them.

    These models define sampling laws Q_t. They are not claimed to equal the
    exact bridge marginals P_t.
    """

    def __init__(self, T: int, reference_model: MaskedTokenModel,
                 device: torch.device):
        self.T      = T
        self.device = device

        # Clone the reference model for each timestep
        self.models      = [copy.deepcopy(reference_model).to(device)
                            for _ in range(T + 1)]
        self.temperatures = np.ones(T + 1, dtype=np.float32)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def gibbs_sweep(self, x: torch.Tensor, t: int,
                    n_sweeps: int = 1) -> torch.Tensor:
        """Run n_sweeps Gibbs sweeps through model M_t at temperature tau_t."""
        tau = float(self.temperatures[t])
        for _ in range(n_sweeps):
            x = self.models[t].gibbs_sweep(x, temperature=tau)
        return x

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_marginal(self, t: int, x_init: torch.Tensor,
                        n_sweeps: int = 5) -> torch.Tensor:
        """Advance the heuristic Q_t sampler from x_init by n_sweeps."""
        return self.gibbs_sweep(x_init, t, n_sweeps)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def conditional_entropy(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Evaluate the predictive-entropy proxy in equation (17)."""
        return self.models[t].conditional_entropy(x)

    # ------------------------------------------------------------------
    def set_temperature(self, t: int, tau: float) -> None:
        self.temperatures[t] = tau

    # ------------------------------------------------------------------
    def find_temperature(self, t: int, x: torch.Tensor,
                         target_H: float,
                         tau_lo: float = 0.5, tau_hi: float = 10.0,
                         n_iter: int = 15) -> float:
        """
        Bisect on temperature tau such that mean H(M_t^tau) >= target_H.

        Higher tau flattens the conditional → higher entropy.
        Updates self.temperatures[t] in place.
        Returns the found temperature.
        """
        @torch.no_grad()
        def mean_H(tau):
            self.temperatures[t] = tau
            return self.conditional_entropy(x, t).mean().item()

        # If already satisfied at tau_lo, no adjustment needed
        if mean_H(tau_lo) >= target_H:
            return tau_lo

        # Expand upper bound if needed
        while mean_H(tau_hi) < target_H and tau_hi < 50.0:
            tau_hi *= 2.0

        for _ in range(n_iter):
            tau_mid = 0.5 * (tau_lo + tau_hi)
            if mean_H(tau_mid) < target_H:
                tau_lo = tau_mid
            else:
                tau_hi = tau_mid

        tau = 0.5 * (tau_lo + tau_hi)
        self.temperatures[t] = tau
        return tau

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save all model state dicts and temperatures."""
        torch.save({
            'T':            self.T,
            'temperatures': self.temperatures.tolist(),
            'model_states': [m.state_dict() for m in self.models],
            'model_cfg':    dict(
                vocab_size  = self.models[0].vocab_size,
                seq_len     = self.models[0].seq_len,
                d_model     = self.models[0].token_emb.embedding_dim - 0,
                n_heads     = self.models[0].transformer.layers[0].self_attn.num_heads,
                n_layers    = len(self.models[0].transformer.layers),
            ),
        }, path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, device: torch.device) -> 'SBPath':
        data     = torch.load(path, map_location=device, weights_only=False)
        T        = data['T']
        cfg      = data['model_cfg']

        ref      = MaskedTokenModel(**cfg)
        sb_path  = cls(T, ref, device)

        for t, state in enumerate(data['model_states']):
            sb_path.models[t].load_state_dict(state)
        sb_path.temperatures = np.array(data['temperatures'], dtype=np.float32)
        return sb_path

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoints(cls, ckpt_paths: list[str],
                         device: torch.device) -> 'SBPath':
        """
        Build an SBPath from a list of T+1 per-timestep MTM checkpoints.

        Each checkpoint is a dict with keys 'model_cfg' and 'model_state',
        as produced by train_noise_mtms.py.

        Args:
            ckpt_paths: list of T+1 paths, ckpt_paths[t] is the checkpoint
                        for timestep t (t=0 = full noise, t=T = clean MNIST)
            device    : target device
        """
        T   = len(ckpt_paths) - 1
        ref = None
        models = []
        for t, path in enumerate(ckpt_paths):
            data  = torch.load(path, map_location=device, weights_only=False)
            model = MaskedTokenModel(**data['model_cfg'])
            model.load_state_dict(data['model_state'])
            model.to(device).eval()
            models.append(model)
            if ref is None:
                ref = model   # just used to initialise cls

        # build SBPath shell then replace models list
        sb_path         = cls(T, ref, device)
        sb_path.models  = models
        return sb_path
