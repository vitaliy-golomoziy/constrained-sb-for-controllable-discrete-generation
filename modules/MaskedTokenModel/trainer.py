import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MaskedTokenTrainer:
    """Manages MLM training for MaskedTokenModel."""

    def __init__(self, model, cfg: dict, device: torch.device):
        self.model  = model.to(device)
        self.cfg    = cfg
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg['mtm_learning_rate'],
            weight_decay=cfg.get('mtm_weight_decay', 1e-4),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['mtm_num_epochs'],
            eta_min=cfg['mtm_learning_rate'] / 10,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def make_loader(tokens_npy: str | np.ndarray, batch_size: int,
                    shuffle: bool = True) -> DataLoader:
        """Build a DataLoader from a (N, K) int16 token array or .npy path."""
        if isinstance(tokens_npy, str):
            tokens_npy = np.load(tokens_npy)
        t = torch.from_numpy(tokens_npy.astype(np.int64))
        return DataLoader(TensorDataset(t), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=False)

    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader,
                    mask_prob: float | None = None) -> float:
        self.model.train()
        total = 0.0
        for (tokens,) in loader:
            tokens = tokens.to(self.device)
            self.optimizer.zero_grad()
            _, loss = self.model.mlm_loss(tokens, mask_prob=mask_prob)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        self.scheduler.step()
        return total / len(loader)

    @torch.no_grad()
    def eval_loss(self, loader: DataLoader,
                  mask_prob: float | None = None) -> float:
        self.model.eval()
        total = 0.0
        for (tokens,) in loader:
            tokens = tokens.to(self.device)
            _, loss = self.model.mlm_loss(tokens, mask_prob=mask_prob)
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def eval_accuracy(self, loader: DataLoader,
                      mask_prob: float | None = None) -> float:
        """Fraction of masked positions predicted correctly."""
        self.model.eval()
        if mask_prob is None:
            mask_prob = self.model.mask_prob
        correct = total = 0
        for (tokens,) in loader:
            tokens = tokens.to(self.device)
            B, K   = tokens.shape
            mask   = torch.rand(B, K, device=self.device) < mask_prob
            if not mask.any():
                continue
            masked          = tokens.clone()
            masked[mask]    = self.model.MASK
            logits          = self.model(masked)
            preds           = logits[mask].argmax(dim=-1)
            correct        += (preds == tokens[mask]).sum().item()
            total          += mask.sum().item()
        return correct / total if total > 0 else 0.0
