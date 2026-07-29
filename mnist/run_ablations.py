"""
Ablation experiments validating that the entropy schedule drives coarse-to-fine emergence.

Ablation 1 — MaskGIT baseline:
    All T+1 models = same reference MTM (trained on clean MNIST).
    Gibbs sampling from this model at every step starting from random tokens.
    Expected: rapid collapse to MNIST-like images with no gradual emergence,
    since the model always targets clean MNIST regardless of context.

Ablation 2 — Entropy schedule contrast:
    Per-noise-level MTMs, four conditions compared side-by-side:
    a) Natural  — no enforcement (tau=1, our main result)
    b) Fast     — entropy drops to MNIST level by t=4 (early crystallization)
    c) Slow     — entropy stays near-maximum until t=6 (late crystallization)
    d) Pulsed   — crystallize, dissolve, and re-crystallize

Saves:
    mnist/output/ablation_maskgit.png
    mnist/output/ablation_schedules.png
    mnist/output/ablation_entropy_profiles.png
    mnist/output/results.json

Usage:
    python mnist/run_ablations.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.VQVAE import VQVAE
from modules.MaskedTokenModel import MaskedTokenModel
from modules.SBTrainer import SBPath


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_ref_model(ckpt_path: Path, device: torch.device) -> MaskedTokenModel:
    data  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MaskedTokenModel(**data['model_cfg'])
    model.load_state_dict(data['model_state'])
    return model.to(device).eval()


@torch.no_grad()
def find_output_temperature(
        sb_path: SBPath, t: int, x: torch.Tensor, target_H: float,
        n_gibbs: int, calibration_sweeps: int = 5,
        n_iters: int = 8) -> float:
    """
    Bisect on tau so that Gibbs sweeps at step t produce output H ≈ target_H.

    Calibrates on the full batch x using cal_gibbs = max(5, n_gibbs//4) sweeps
    (fast but sufficient for tau selection).  Sets sb_path.temperatures[t] to
    the found tau and returns it.
    """
    cal_gibbs = calibration_sweeps

    def output_H(tau: float) -> float:
        sb_path.temperatures[t] = tau
        x_out = sb_path.sample_marginal(t, x, n_sweeps=cal_gibbs)
        return sb_path.conditional_entropy(x_out, t).mean().item()

    H_nat = output_H(1.0)
    if abs(H_nat - target_H) < 0.05:
        return 1.0

    if H_nat > target_H:
        # Need to concentrate → search tau in (tau_min, 1)
        tau_lo, tau_hi = 0.05, 1.0
        for _ in range(n_iters):
            tau_mid = (tau_lo + tau_hi) / 2.0
            if output_H(tau_mid) > target_H:
                tau_hi = tau_mid
            else:
                tau_lo = tau_mid
    else:
        # Need to spread → search tau in (1, tau_max)
        tau_lo, tau_hi = 1.0, 2.0
        while tau_hi < 50.0 and output_H(tau_hi) < target_H:
            tau_hi *= 2.0
        for _ in range(n_iters):
            tau_mid = (tau_lo + tau_hi) / 2.0
            if output_H(tau_mid) < target_H:
                tau_lo = tau_mid
            else:
                tau_hi = tau_mid

    tau = (tau_lo + tau_hi) / 2.0
    sb_path.temperatures[t] = tau
    return tau


@torch.no_grad()
def generate(sb_path: SBPath, vocab_size: int, n_samples: int,
             n_gibbs: int, device: torch.device,
             entropy_targets: list[float] | None = None,
             calibration_sweeps: int = 5,
             bisection_iterations: int = 8) -> tuple[list, list]:
    """
    Forward generation t=0..T.

    entropy_targets: if provided, bisect on OUTPUT H at each step (not input H)
                     so the actual post-Gibbs samples hit the target.
                     None = no enforcement (tau=1 throughout).

    Returns (frames, Hs).
    """
    K = sb_path.models[0].seq_len
    x = torch.randint(0, vocab_size, (n_samples, K), dtype=torch.long, device=device)
    frames = [x.clone()]
    Hs     = [sb_path.conditional_entropy(x, 0).mean().item()]

    for t in range(1, sb_path.T + 1):
        if entropy_targets is not None:
            tau = find_output_temperature(
                sb_path,
                t,
                x,
                entropy_targets[t],
                n_gibbs,
                calibration_sweeps,
                bisection_iterations,
            )
        x = sb_path.sample_marginal(t, x, n_sweeps=n_gibbs)
        frames.append(x.clone())
        H_out = sb_path.conditional_entropy(x, t).mean().item()
        Hs.append(H_out)
        if entropy_targets is not None:
            print(f"    t={t:2d}  tau={tau:.3f}  H={H_out:.3f}  (target={entropy_targets[t]:.2f})")
            sb_path.temperatures[t] = 1.0

    sb_path.temperatures[:] = 1.0
    return frames, Hs


@torch.no_grad()
def decode(vqvae: VQVAE, tokens: torch.Tensor, device: torch.device) -> np.ndarray:
    imgs = vqvae.decode(tokens.to(device)).squeeze(1).cpu().numpy()
    return np.clip(imgs, 0.0, 1.0)


def save_panel(frames_imgs: list, Hs: list, title: str,
               n_rows: int, out_path: Path) -> None:
    """Save a grid: n_rows sample rows × (T+1) timestep columns, with H labels."""
    T1 = len(frames_imgs)
    fig, axes = plt.subplots(n_rows, T1, figsize=(1.5 * T1, 1.5 * n_rows + 0.6))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    for t in range(T1):
        for r in range(n_rows):
            axes[r, t].imshow(frames_imgs[t][r], cmap='gray', vmin=0, vmax=1)
            axes[r, t].axis('off')
        axes[0, t].set_title(f't={t}\nH={Hs[t]:.2f}', fontsize=7)
    fig.suptitle(title, fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_schedule_comparison(all_frames: list, all_Hs: list, all_labels: list,
                              T: int, out_path: Path,
                              n_per: int = 2) -> None:
    """Multi-row figure: one block of rows per schedule, showing 2 samples each."""
    n_schedules = len(all_frames)
    fig, axes = plt.subplots(n_schedules * n_per, T + 1,
                             figsize=(1.5 * (T + 1), 1.5 * n_schedules * n_per + 1.0))

    for s, (frames_imgs, Hs, label) in enumerate(zip(all_frames, all_Hs, all_labels)):
        for r in range(n_per):
            row = s * n_per + r
            for t in range(T + 1):
                axes[row, t].imshow(frames_imgs[t][r], cmap='gray', vmin=0, vmax=1)
                axes[row, t].axis('off')
            if r == 0:
                axes[row, 0].set_ylabel(label, fontsize=8, rotation=90,
                                        labelpad=4, va='center')
        # column headers on top row
        if s == 0:
            for t in range(T + 1):
                axes[0, t].set_title(f't={t}\nH={Hs[t]:.2f}', fontsize=7)

    fig.suptitle('Entropy schedule contrast — same noise-MTM path, different targets',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_entropy_plot(all_Hs: list, all_labels: list, all_styles: list,
                      T: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for Hs, label, style in zip(all_Hs, all_labels, all_styles):
        ax.plot(range(T + 1), Hs, style, label=label, linewidth=1.8, markersize=5)
    ax.set_xlabel('timestep t')
    ax.set_ylabel('predictive-entropy proxy (nats)')
    ax.set_title('Predictive-entropy profiles under different schedules')
    ax.legend(fontsize=8)
    ax.set_ylim(0, np.log(16) * 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(
    config_path: str,
    *,
    checkpoint_dir: Path | None = None,
    output_dir: Path | None = None,
    seed_override: int | None = None,
    n_samples_override: int | None = None,
    n_gibbs_override: int | None = None,
    calibration_sweeps_override: int | None = None,
    bisection_iterations_override: int | None = None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_dir = Path(config_path).resolve().parent
    ckpt_dir = checkpoint_dir or (exp_dir / 'checkpoints')
    out_dir = output_dir or (exp_dir / 'output')
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get('seed', 0) if seed_override is None else seed_override)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device     = select_device()
    T          = cfg['sb_T']
    vocab_size = cfg['num_embeddings']
    N_SAMPLES = int(
        cfg.get('ablation_n_samples', 16)
        if n_samples_override is None
        else n_samples_override
    )
    N_GIBBS = int(
        cfg.get('ablation_gibbs_sweeps', 20)
        if n_gibbs_override is None
        else n_gibbs_override
    )
    N_CALIBRATION = int(
        cfg.get('ablation_calibration_sweeps', 5)
        if calibration_sweeps_override is None
        else calibration_sweeps_override
    )
    N_BISECTION = int(
        cfg.get('ablation_bisection_iterations', 8)
        if bisection_iterations_override is None
        else bisection_iterations_override
    )

    print(
        f"Device: {device}  seed={seed}  T={T}  vocab={vocab_size}  "
        f"n_gibbs={N_GIBBS}\n"
    )

    # ------------------------------------------------------------------ VQ-VAE
    vqvae = VQVAE.load(str(ckpt_dir / 'best.pt'), map_location=device).to(device).eval()

    # ---------------------------------------------------------- noise-MTM path
    noise_dir  = ckpt_dir / 'noise_mtms'
    ckpt_paths = [str(noise_dir / f'mtm_t{t:02d}.pt') for t in range(T + 1)]
    print("Loading per-noise-level MTMs ...")
    sb_noise = SBPath.from_checkpoints(ckpt_paths, device)

    # ------------------------------------------------------ reference MTM path
    print("Loading reference MTM (MaskGIT baseline) ...")
    ref_model = load_ref_model(ckpt_dir / 'mtm_best.pt', device)
    sb_ref    = SBPath(T=T, reference_model=ref_model, device=device)

    # ============================================================ ABLATION 1
    print("\n=== Ablation 1: MaskGIT-like baseline (reference MTM everywhere) ===")
    frames_ref, Hs_ref = generate(sb_ref, vocab_size, N_SAMPLES, N_GIBBS, device)
    frames_ref_imgs = [decode(vqvae, x, device) for x in frames_ref]
    for t, H in enumerate(Hs_ref):
        print(f"  t={t:2d}  H={H:.3f}")
    save_panel(frames_ref_imgs, Hs_ref,
               title='Ablation 1 — MaskGIT-like baseline (reference MTM at all steps)',
               n_rows=min(8, N_SAMPLES), out_path=out_dir / 'ablation_maskgit.png')

    # ============================================================ ABLATION 2
    print("\n=== Ablation 2: Entropy schedule contrast ===")

    # Natural: no enforcement
    print("\n  [Natural — no enforcement]")
    frames_nat, Hs_nat = generate(sb_noise, vocab_size, N_SAMPLES, N_GIBBS, device)
    frames_nat_imgs = [decode(vqvae, x, device) for x in frames_nat]
    for t, H in enumerate(Hs_nat):
        print(f"  t={t:2d}  H={H:.3f}")

    # Fast crystallization: entropy drops to MNIST level by t=4
    # Force early structure — lower entropy at t=1..4 to concentrate distribution
    fast_schedule = [
        1.64,   # t=0 (random source, not enforced but noted)
        2.00,   # t=1: concentrate slightly vs natural 2.75
        1.50,   # t=2: concentrate more
        1.10,   # t=3: near-MNIST
        0.90,   # t=4: MNIST-like early
        0.82,   # t=5
        0.78,   # t=6
        0.75,   # t=7
        0.45,   # t=8: well below natural endpoint (~0.58), force concentrate
    ]
    print("\n  [Fast crystallization — entropy drops to MNIST level by t=4]")
    frames_fast, Hs_fast = generate(sb_noise, vocab_size, N_SAMPLES, N_GIBBS, device,
                                    entropy_targets=fast_schedule,
                                    calibration_sweeps=N_CALIBRATION,
                                    bisection_iterations=N_BISECTION)
    frames_fast_imgs = [decode(vqvae, x, device) for x in frames_fast]

    # Slow crystallization: entropy stays near-maximum until t=6
    slow_schedule = [
        1.64,   # t=0
        2.72,   # t=1: stay near-maximum
        2.65,   # t=2
        2.55,   # t=3
        2.60,   # t=4: push above natural (2.41)
        2.55,   # t=5: well above natural (2.12) — requires tau >> 1
        2.40,   # t=6: well above natural (1.74) — images still diffuse
        2.00,   # t=7: well above natural (1.27) — delayed crystallization
        0.74,   # t=8: crystallize
    ]
    print("\n  [Slow crystallization — entropy stays high until t=6]")
    frames_slow, Hs_slow = generate(sb_noise, vocab_size, N_SAMPLES, N_GIBBS, device,
                                    entropy_targets=slow_schedule,
                                    calibration_sweeps=N_CALIBRATION,
                                    bisection_iterations=N_BISECTION)
    frames_slow_imgs = [decode(vqvae, x, device) for x in frames_slow]

    # Pulsed: natural up to t=4, concentrate to partial digits at t=5,
    # dissolve back to chaos at t=6, re-crystallize at t=7-8.
    # Relies on natural structure built up by t=4 so that concentrated t=5
    # sharpens real digit context rather than collapsing to background.
    pulsed_schedule = [
        1.64,   # t=0
        2.72,   # t=1: near natural
        2.65,   # t=2: near natural
        2.55,   # t=3: near natural
        2.40,   # t=4: near natural — build up spatial structure first
        1.58,   # t=5: concentrate hard → partial digits emerge
        2.30,   # t=6: spread back to chaos → digits dissolve
        0.85,   # t=7: concentrate → digits re-emerge
        0.45,   # t=8: final clean digits
    ]
    print("\n  [Pulsed — crystallize at t=5, dissolve at t=6, re-crystallize at t=7]")
    frames_pulsed, Hs_pulsed = generate(sb_noise, vocab_size, N_SAMPLES, N_GIBBS, device,
                                        entropy_targets=pulsed_schedule,
                                        calibration_sweeps=N_CALIBRATION,
                                        bisection_iterations=N_BISECTION)
    frames_pulsed_imgs = [decode(vqvae, x, device) for x in frames_pulsed]

    # ------------------------------------------------------------ save outputs
    save_schedule_comparison(
        all_frames =[frames_nat_imgs, frames_fast_imgs, frames_slow_imgs, frames_pulsed_imgs],
        all_Hs     =[Hs_nat,         Hs_fast,          Hs_slow,          Hs_pulsed],
        all_labels =['Natural (no enforcement)',
                     'Fast crystallization',
                     'Slow crystallization',
                     'Pulsed (crystallize → dissolve → re-crystallize)'],
        T=T,
        out_path=out_dir / 'ablation_schedules.png',
        n_per=min(2, N_SAMPLES),
    )

    panel_rows = min(8, N_SAMPLES)
    save_panel(frames_nat_imgs,    Hs_nat,    'Natural (no enforcement)',                    panel_rows, out_dir / 'ablation_natural.png')
    save_panel(frames_fast_imgs,   Hs_fast,   'Fast crystallization schedule',               panel_rows, out_dir / 'ablation_fast.png')
    save_panel(frames_slow_imgs,   Hs_slow,   'Slow crystallization schedule',               panel_rows, out_dir / 'ablation_slow.png')
    save_panel(frames_pulsed_imgs, Hs_pulsed, 'Pulsed (crystallize → dissolve → re-cryst)', panel_rows, out_dir / 'ablation_pulsed.png')

    save_entropy_plot(
        all_Hs    =[Hs_ref, Hs_nat, Hs_fast, Hs_slow, Hs_pulsed],
        all_labels=['Single clean MTM', 'Natural', 'Fast', 'Slow', 'Pulsed'],
        all_styles=['s--', 'o-', '^:', 'v-.', 'D-'],
        T=T,
        out_path=out_dir / 'ablation_entropy_profiles.png',
    )

    results = {
        'description': (
            'Section 7 amortized MNIST heuristic. Values are the predictive-'
            'entropy proxy of equation (17), not exact erasure entropies of '
            'Schrodinger bridge marginals.'
        ),
        'seed': seed,
        'device': str(device),
        'num_samples': N_SAMPLES,
        'num_steps': T,
        'gibbs_sweeps_per_timestep': N_GIBBS,
        'calibration_sweeps': N_CALIBRATION,
        'bisection_iterations': N_BISECTION,
        'profiles': {
            'single_clean_mtm': Hs_ref,
            'natural': Hs_nat,
            'fast': Hs_fast,
            'slow': Hs_slow,
            'pulsed': Hs_pulsed,
        },
        'targets': {
            'fast': fast_schedule,
            'slow': slow_schedule,
            'pulsed': pulsed_schedule,
        },
    }
    (out_dir / 'results.json').write_text(
        json.dumps(results, indent=2) + '\n',
        encoding='utf-8',
    )
    print("\nAll ablations complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        nargs='?',
        default=str(Path(__file__).resolve().parent / 'config.yaml'),
    )
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n-samples', type=int)
    parser.add_argument('--n-gibbs', type=int)
    parser.add_argument('--calibration-sweeps', type=int)
    parser.add_argument('--bisection-iterations', type=int)
    args = parser.parse_args()
    main(
        args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        seed_override=args.seed,
        n_samples_override=args.n_samples,
        n_gibbs_override=args.n_gibbs,
        calibration_sweeps_override=args.calibration_sweeps,
        bisection_iterations_override=args.bisection_iterations,
    )
