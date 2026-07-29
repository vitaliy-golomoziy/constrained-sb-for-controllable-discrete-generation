"""Train a VQ-VAE on MNIST and save token sequences for downstream SB training.

Usage:
    python mnist/train_vqvae.py
    python mnist/train_vqvae.py /path/to/config.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.VQVAE import VQVAE, VQVAETrainer


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    exp_dir  = Path(config_path).parent
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    device = select_device()
    print(f"Device : {device}")

    # ------------------------------------------------------------------ data
    data_dir  = exp_dir / cfg['data_dir']
    transform = transforms.ToTensor()
    train_ds  = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds   = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=False)

    # ----------------------------------------------------------------- model
    model = VQVAE(
        hidden_channels=cfg['hidden_channels'],
        embedding_dim=cfg['embedding_dim'],
        num_embeddings=cfg['num_embeddings'],
        commitment_cost=cfg['commitment_cost'],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : VQVAE  params={n_params:,}  "
          f"K={model.K}  A={model.num_embeddings}")

    trainer = VQVAETrainer(model, cfg, device)

    # --------------------------------------------------------------- training
    best_loss = float('inf')
    for epoch in range(1, cfg['num_epochs'] + 1):
        train_loss, recon, vq = trainer.train_epoch(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            test_loss = trainer.eval_loss(test_loader)
            print(f"Epoch {epoch:3d}/{cfg['num_epochs']}  "
                  f"train={train_loss:.4f}  (recon={recon:.4f}  vq={vq:.4f})  "
                  f"test={test_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            model.save(ckpt_dir / 'best.pt')

    model.save(ckpt_dir / 'final.pt')
    print(f"\nCheckpoints saved to {ckpt_dir}")

    # ----------------------------------------------- encode and save tokens
    print("Encoding training set with best checkpoint...")
    model = VQVAE.load(ckpt_dir / 'best.pt', map_location=device)
    trainer = VQVAETrainer(model, cfg, device)

    # encode_dataset returns labels in the loader's iteration order, so tokens
    # and labels stay aligned row-for-row even though train_loader shuffles.
    train_tokens, train_labels = trainer.encode_dataset(train_loader)  # (60000, 49)
    test_tokens,  test_labels  = trainer.encode_dataset(test_loader)   # (10000, 49)

    np.save(exp_dir / 'train_tokens.npy', train_tokens)
    np.save(exp_dir / 'test_tokens.npy',  test_tokens)
    np.save(exp_dir / 'train_labels.npy', train_labels)
    np.save(exp_dir / 'test_labels.npy',  test_labels)

    print(f"Saved train_tokens {train_tokens.shape}, test_tokens {test_tokens.shape}")
    print(f"Saved train_labels {train_labels.shape}, test_labels {test_labels.shape}")
    print(f"Token range: [{train_tokens.min()}, {train_tokens.max()}]")

    # -------------------------------------------- quick codebook usage stats
    counts = np.bincount(train_tokens.ravel(), minlength=cfg['num_embeddings'])
    usage  = (counts > 0).sum()
    print(f"Codebook usage: {usage}/{cfg['num_embeddings']} entries active")
    print(f"  per-entry counts: {counts.tolist()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?',
                        default=str(Path(__file__).resolve().parent / 'config.yaml'))
    args = parser.parse_args()
    main(args.config)
