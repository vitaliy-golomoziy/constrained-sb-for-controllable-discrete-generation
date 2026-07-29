"""
Pre-train one MTM per noise level for the masking-schedule SB reference.

For T timesteps we need T+1 models at sigma_t = 1 - t/T:
  t=T : sigma=0  → clean MNIST       (reuse mtm_best.pt, skip training)
  t=0 : sigma=1  → all random        (skip — uniform prediction is exact)
  t=1..T-1: fine-tune from reference MTM on corrupted MNIST

Fine-tuning rather than training from scratch is ~5-10× faster and works
well because all noise levels share the same token vocabulary and structure.

Usage:
    python mnist/train_noise_mtms.py
    python mnist/train_noise_mtms.py /path/to/config.yaml
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, random, shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from modules.MaskedTokenModel import MaskedTokenModel, MaskedTokenTrainer
from modules.SBTrainer.corruption import make_corrupted_loader, noise_level


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_reference_mtm(ckpt_path: Path, device: torch.device) -> tuple[MaskedTokenModel, dict]:
    data  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MaskedTokenModel(**data['model_cfg'])
    model.load_state_dict(data['model_state'])
    return model.to(device), data['model_cfg']


def save_mtm(model: MaskedTokenModel, model_cfg: dict, path: Path) -> None:
    torch.save({'model_state': model.state_dict(), 'model_cfg': model_cfg}, path)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    exp_dir  = Path(config_path).parent
    ckpt_dir = exp_dir / 'checkpoints'
    out_dir  = ckpt_dir / 'noise_mtms'
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Device : {device}")

    T           = cfg['sb_T']
    vocab_size  = cfg['num_embeddings']
    train_tokens = np.load(exp_dir / 'train_tokens.npy')
    test_tokens  = np.load(exp_dir / 'test_tokens.npy')
    print(f"Tokens : {len(train_tokens):,} train  /  {len(test_tokens):,} test")
    print(f"T={T}  vocab={vocab_size}")
    print()

    # Load reference MTM (trained on clean MNIST)
    ref_ckpt        = ckpt_dir / 'mtm_best.pt'
    ref_model, mcfg = load_reference_mtm(ref_ckpt, device)
    print(f"Reference MTM loaded from {ref_ckpt}")

    # ------------------------------------------------------------------ t=T
    # t=T: sigma=0 (clean MNIST) — copy the reference checkpoint directly
    dst_T = out_dir / f'mtm_t{T:02d}.pt'
    shutil.copy(ref_ckpt, dst_T)
    print(f"t={T:2d}  sigma={noise_level(T,T):.3f}  → copied reference  [{dst_T.name}]")

    # ------------------------------------------------------------------ t=0
    # t=0: sigma=1 (all random) — skip, caller will use uniform sampling
    dst_0 = out_dir / f'mtm_t00.pt'
    save_mtm(ref_model, mcfg, dst_0)   # save architecture for loading, weights don't matter
    print(f"t= 0  sigma={noise_level(0,T):.3f}  → saved (uniform — weights irrelevant)  [{dst_0.name}]")
    print()

    # ------------------------------------------------------------------ t=1..T-1
    finetune_epochs = cfg.get('sb_noise_mtm_epochs', 20)
    finetune_lr     = cfg.get('sb_noise_mtm_lr',     5e-4)
    batch_size      = cfg['mtm_batch_size']

    for t in range(1, T):
        sigma = noise_level(t, T)
        print(f"t={t:2d}  sigma={sigma:.3f}  ({int(sigma*100):3d}% tokens corrupted)")

        # fresh copy of reference for fine-tuning
        import copy
        model = copy.deepcopy(ref_model)
        model.train()

        train_loader = make_corrupted_loader(train_tokens, sigma, batch_size,
                                             shuffle=True,  vocab_size=vocab_size)
        test_loader  = make_corrupted_loader(test_tokens,  sigma, batch_size,
                                             shuffle=False, vocab_size=vocab_size)

        ft_cfg = {'mtm_learning_rate': finetune_lr,
                  'mtm_weight_decay':  cfg.get('mtm_weight_decay', 1e-4),
                  'mtm_num_epochs':    finetune_epochs}
        trainer = MaskedTokenTrainer(model, ft_cfg, device)

        best_loss = float('inf')
        for epoch in range(1, finetune_epochs + 1):
            train_loss = trainer.train_epoch(train_loader)
            if epoch % 5 == 0 or epoch == finetune_epochs:
                test_loss = trainer.eval_loss(test_loader)
                acc       = trainer.eval_accuracy(test_loader)
                print(f"  epoch {epoch:3d}/{finetune_epochs}  "
                      f"train={train_loss:.4f}  test={test_loss:.4f}  acc={acc:.3f}")
            if train_loss < best_loss:
                best_loss = train_loss
                dst = out_dir / f'mtm_t{t:02d}.pt'
                save_mtm(model, mcfg, dst)

        # measure conditional entropy on corrupted test samples
        model.eval()
        test_loader2 = make_corrupted_loader(test_tokens, sigma, 256,
                                              shuffle=False, vocab_size=vocab_size)
        Hs = []
        with torch.no_grad():
            for (batch,) in test_loader2:
                batch = batch.to(device)
                Hs.append(model.conditional_entropy(batch).mean().item())
                if len(Hs) >= 8:
                    break
        H_mean = float(np.mean(Hs))
        print(f"  → saved  loss={best_loss:.4f}  H={H_mean:.3f}  [{dst.name}]")
        print()

    print(f"All noise-level MTMs saved to: {out_dir}")
    print("Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        nargs='?',
        default=str(Path(__file__).resolve().parent / 'config.yaml'),
    )
    args = parser.parse_args()
    main(args.config)
