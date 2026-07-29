"""
Token corruption utilities for the masking-schedule reference process.

At noise level sigma in [0, 1], each token is independently replaced by a
uniform-random token with probability sigma.  sigma=0 → clean MNIST,
sigma=1 → all tokens uniform random.

For T timesteps the schedule is: sigma_t = 1 - t/T
  t=0: sigma=1  (all random — source P_0)
  t=T: sigma=0  (clean MNIST — target P_T)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def corrupt_tokens(tokens: torch.Tensor, sigma: float,
                   vocab_size: int = 16) -> torch.Tensor:
    """
    Corrupt a batch of token sequences.

    Args:
        tokens    : (B, K) long — clean token sequences
        sigma     : corruption probability in [0, 1]
        vocab_size: number of codebook entries
    Returns:
        corrupted : (B, K) long — tokens replaced uniformly at random with
                    probability sigma, original value kept with prob 1-sigma
    """
    if sigma <= 0.0:
        return tokens.clone()
    if sigma >= 1.0:
        return torch.randint(0, vocab_size, tokens.shape,
                             dtype=tokens.dtype, device=tokens.device)

    mask      = torch.rand_like(tokens, dtype=torch.float) < sigma
    random_tk = torch.randint(0, vocab_size, tokens.shape,
                              dtype=tokens.dtype, device=tokens.device)
    return torch.where(mask, random_tk, tokens)


def noise_level(t: int, T: int) -> float:
    """sigma_t = 1 - t/T."""
    return 1.0 - t / T


class CorruptedTokenDataset(Dataset):
    """
    Wraps a numpy token array and applies random corruption on-the-fly.

    Each __getitem__ call applies a fresh independent corruption, so the
    effective dataset is much larger than the raw token count.
    """

    def __init__(self, tokens_npy: np.ndarray, sigma: float,
                 vocab_size: int = 16):
        """
        Args:
            tokens_npy: (N, K) int16/int64 array of clean MNIST tokens
            sigma      : corruption probability (noise level)
            vocab_size : codebook size
        """
        self.tokens     = torch.from_numpy(tokens_npy.astype(np.int64))
        self.sigma      = sigma
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx]                             # (K,)
        corrupted = corrupt_tokens(x.unsqueeze(0), self.sigma,
                                   self.vocab_size).squeeze(0)
        return (corrupted,)                              # tuple — matches TensorDataset API


def make_corrupted_loader(tokens_npy: np.ndarray, sigma: float,
                          batch_size: int, shuffle: bool = True,
                          vocab_size: int = 16) -> DataLoader:
    ds = CorruptedTokenDataset(tokens_npy, sigma, vocab_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)
