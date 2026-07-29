"""Train the masked token model (reference Gibbs kernel) on MNIST tokens.

Requires train_tokens.npy produced by train_vqvae.py.

Usage:
    python mnist/train_masked_model.py
    python mnist/train_masked_model.py /path/to/config.yaml
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

from modules.MaskedTokenModel import MaskedTokenModel, MaskedTokenTrainer


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
    train_tokens = np.load(exp_dir / 'train_tokens.npy')  # (60000, 49)
    test_tokens  = np.load(exp_dir / 'test_tokens.npy')   # (10000, 49)

    train_loader = MaskedTokenTrainer.make_loader(
        train_tokens, cfg['mtm_batch_size'], shuffle=True)
    test_loader  = MaskedTokenTrainer.make_loader(
        test_tokens,  cfg['mtm_batch_size'], shuffle=False)

    # ----------------------------------------------------------------- model
    model = MaskedTokenModel(
        vocab_size=cfg['num_embeddings'],
        seq_len=49,
        d_model=cfg['mtm_d_model'],
        n_heads=cfg['mtm_n_heads'],
        n_layers=cfg['mtm_n_layers'],
        dropout=cfg['mtm_dropout'],
        mask_prob=cfg['mtm_mask_prob'],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : MaskedTokenModel  params={n_params:,}  "
          f"vocab={cfg['num_embeddings']}  seq_len=49")

    trainer = MaskedTokenTrainer(model, cfg, device)

    # --------------------------------------------------------------- training
    best_loss = float('inf')
    for epoch in range(1, cfg['mtm_num_epochs'] + 1):
        train_loss = trainer.train_epoch(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            test_loss = trainer.eval_loss(test_loader)
            acc       = trainer.eval_accuracy(test_loader)
            print(f"Epoch {epoch:3d}/{cfg['mtm_num_epochs']}  "
                  f"train={train_loss:.4f}  test={test_loss:.4f}  "
                  f"acc={acc:.3f}")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({'model_state': model.state_dict(),
                        'model_cfg':   dict(
                            vocab_size=cfg['num_embeddings'], seq_len=49,
                            d_model=cfg['mtm_d_model'], n_heads=cfg['mtm_n_heads'],
                            n_layers=cfg['mtm_n_layers'], dropout=cfg['mtm_dropout'],
                            mask_prob=cfg['mtm_mask_prob'])},
                       ckpt_dir / 'mtm_best.pt')

    torch.save({'model_state': model.state_dict(),
                'model_cfg':   dict(
                    vocab_size=cfg['num_embeddings'], seq_len=49,
                    d_model=cfg['mtm_d_model'], n_heads=cfg['mtm_n_heads'],
                    n_layers=cfg['mtm_n_layers'], dropout=cfg['mtm_dropout'],
                    mask_prob=cfg['mtm_mask_prob'])},
               ckpt_dir / 'mtm_final.pt')
    print(f"\nCheckpoints saved to {ckpt_dir}")

    # ------------------------------------------ quick Gibbs sanity check
    print("\nGibbs sampling sanity check (5 sweeps from random tokens)...")
    model.eval()
    rng    = np.random.default_rng(0)
    sample = torch.from_numpy(
        rng.integers(0, cfg['num_embeddings'], size=(8, 49)).astype(np.int64)
    ).to(device)

    for _ in range(5):
        sample = model.gibbs_sweep(sample)

    H = model.conditional_entropy(sample).cpu().numpy()
    print(f"  Conditional entropy after 5 sweeps: "
          f"mean={H.mean():.3f}  min={H.min():.3f}  max={H.max():.3f}  "
          f"(log A = {np.log(cfg['num_embeddings']):.3f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?',
                        default=str(Path(__file__).resolve().parent / 'config.yaml'))
    args = parser.parse_args()
    main(args.config)
