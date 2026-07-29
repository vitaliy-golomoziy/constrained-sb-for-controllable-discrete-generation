import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


class VQVAETrainer:
    """Manages training and dataset encoding for a VQVAE model."""

    def __init__(self, model, cfg: dict, device: torch.device):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        total = recon_sum = vq_sum = 0.0
        for x, _ in loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            _, loss, info = self.model(x)
            loss.backward()
            self.optimizer.step()
            total     += loss.item()
            recon_sum += info['recon_loss']
            vq_sum    += info['vq_loss']
        n = len(loader)
        return total / n, recon_sum / n, vq_sum / n

    @torch.no_grad()
    def eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for x, _ in loader:
            x = x.to(self.device)
            _, loss, _ = self.model(x)
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def encode_dataset(self, loader: DataLoader):
        """Encode all images in loader to flat token arrays.

        Returns:
            tokens: np.ndarray of shape (N, K) int16 — one row per image,
                K = 49 token indices in [0, num_embeddings).
            labels: np.ndarray of shape (N,) uint8 — the corresponding labels
                in the loader's iteration order. Returning labels here (rather
                than reading them from the dataset separately) keeps tokens and
                labels aligned row-for-row even when the loader uses shuffle=True.
        """
        self.model.eval()
        tok_parts, lbl_parts = [], []
        for x, y in loader:
            x = x.to(self.device)
            idx = self.model.encode(x)        # (B, 7, 7)
            tok_parts.append(idx.cpu().numpy().reshape(len(x), -1))
            lbl_parts.append(np.asarray(y).reshape(-1))
        tokens = np.concatenate(tok_parts, axis=0).astype(np.int16)
        labels = np.concatenate(lbl_parts, axis=0).astype(np.uint8)
        return tokens, labels
