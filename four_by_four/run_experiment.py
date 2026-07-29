#!/usr/bin/env python3
"""Reproduce the three binary 4x4 experiments reported in Section 6.

Runs
----
3C  Unconstrained Schrödinger bridge from Blank to Block.
3E  Hard joint-entropy schedule chosen below the reachable envelope.
3G  Hinge-penalized bridge for a deliberately unreachable 6-nat schedule.

The exact state space has 2^16 states. The sparse Gibbs reference resamples one
uniformly selected bit from Bernoulli(1/2) at each time step; its transition
matrix has 17 nonzeros per row (the 16 one-bit changes plus the self-loop).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "modules"))

from Functional.ConditionalEntropyFunctional import ConditionalEntropyFunctional
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from ReferenceProcess.SparseGibbsReferenceProcess import SparseGibbsReferenceProcess
from Solver import Reachability
from Solver.HingePenaltyProjection import HingePenaltyProjection
from Solver.IPFDykstraSolver import IPFDykstraSolver
from Solver.JointEntropyProjection import JointEntropyProjection


def product_bernoulli(theta: np.ndarray) -> np.ndarray:
    """Return a product-Bernoulli law in lexicographic binary-state order."""
    theta = np.asarray(theta, dtype=float)
    num_tokens = theta.size
    states = np.arange(2**num_tokens, dtype=np.int64)
    bit_positions = np.arange(num_tokens - 1, -1, -1, dtype=np.int64)
    bits = ((states[:, None] >> bit_positions[None, :]) & 1).astype(float)
    log_p = bits @ np.log(theta) + (1.0 - bits) @ np.log1p(-theta)
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def joint_entropy(p: np.ndarray) -> float:
    positive = p[p > 0]
    return float(-positive @ np.log(positive))


def erasure_entropy(p: np.ndarray, num_tokens: int) -> float:
    tensor = p.reshape([2] * num_tokens)
    total = num_tokens * joint_entropy(p)
    for k in range(num_tokens):
        total -= joint_entropy(tensor.sum(axis=k).ravel())
    return float(total / num_tokens)


def pixel_means(p: np.ndarray, num_tokens: int) -> list[float]:
    states = np.arange(p.size, dtype=np.int64)
    bit_positions = np.arange(num_tokens - 1, -1, -1, dtype=np.int64)
    bits = ((states[:, None] >> bit_positions[None, :]) & 1).astype(float)
    return (bits.T @ p).tolist()


def solve(
    phase_space: DiscretePhaseSpace,
    reference: SparseGibbsReferenceProcess,
    initial: np.ndarray,
    terminal: np.ndarray,
    schedule: np.ndarray,
    *,
    projection: Any,
    max_cycles: int,
    tolerance: float,
    on_infeasible: str,
) -> dict[str, Any]:
    solver = IPFDykstraSolver(
        phase_space,
        reference,
        ConditionalEntropyFunctional(),
        projection=projection,
        on_infeasible=on_infeasible,
    )
    started = time.perf_counter()
    marginals = solver.solve_marginals_only(
        initial,
        terminal,
        schedule,
        max_cycles=max_cycles,
        tol=tolerance,
    )
    elapsed = time.perf_counter() - started

    num_tokens = phase_space.n
    entropies = [joint_entropy(m) for m in marginals]
    erasure_entropies = [erasure_entropy(m, num_tokens) for m in marginals]
    fan_in = Reachability.kernel_fan_in(reference._kernel)
    increments = np.abs(np.diff(entropies))
    deficits = [
        [t, float(schedule[t] - entropies[t])]
        for t in range(1, phase_space.num_steps)
        if entropies[t] < schedule[t] - 1e-6
    ]

    return {
        "elapsed_seconds": elapsed,
        "cycles": solver.n_cycles,
        "converged": bool(solver.converged),
        "final_residual": float(solver.final_residual),
        "convergence_history": [float(x) for x in solver.convergence_history],
        "joint_entropy": entropies,
        "erasure_entropy": erasure_entropies,
        "pixel_means": [pixel_means(m, num_tokens) for m in marginals],
        "schedule_deficits": deficits,
        "maximum_joint_entropy_increment": float(increments.max()),
        "increment_bound_log_fan_in": float(np.log(fan_in)),
        "increment_bound_satisfied": bool(
            increments.max() <= np.log(fan_in) + 1e-9
        ),
        "endpoint_maximum_absolute_error": [
            float(np.max(np.abs(marginals[0] - initial))),
            float(np.max(np.abs(marginals[-1] - terminal))),
        ],
    }


def monotone_schedule(start: float, cap: float, num_steps: int) -> np.ndarray:
    schedule = np.zeros(num_steps + 1)
    schedule[1:num_steps] = np.linspace(start, cap, num_steps - 1)
    return schedule


def make_figure(
    free: dict[str, Any],
    hard: dict[str, Any],
    hard_schedule: np.ndarray,
    display_steps: list[int],
    rows: int,
    cols: int,
    output_dir: Path,
) -> None:
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(display_steps)
    fig = plt.figure(figsize=(1.7 * ncols + 1.5, 7.1))
    grid = fig.add_gridspec(
        3,
        ncols,
        height_ratios=[1, 1, 1.55],
        hspace=0.35,
        wspace=0.08,
    )
    for row, (label, result) in enumerate(
        [("3C: unconstrained", free), ("3E: hard schedule", hard)]
    ):
        for col, t in enumerate(display_steps):
            ax = fig.add_subplot(grid[row, col])
            image = np.asarray(result["pixel_means"][t]).reshape(rows, cols)
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$t={t}$", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)

    ax = fig.add_subplot(grid[2, :])
    times = np.arange(len(free["joint_entropy"]))
    ax.plot(times, free["joint_entropy"], "o-", label="3C: unconstrained")
    ax.plot(times, hard["joint_entropy"], "s--", label="3E: hard schedule")
    ax.step(times, hard_schedule, where="mid", linestyle=":", label="schedule")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("$H(P_t)$ [nats]")
    ax.set_xticks(times)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle(
        r"Blank$\to$Block ($K=16$, $A=2$): free vs. binding schedule",
        fontsize=11,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"blank_to_block.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)


def run(config: dict[str, Any]) -> dict[str, Any]:
    rows = int(config["grid_rows"])
    cols = int(config["grid_cols"])
    num_tokens = rows * cols
    num_steps = int(config["num_steps"])
    phase_space = DiscretePhaseSpace(
        n=num_tokens,
        alphabet_size=2,
        num_steps=num_steps,
    )
    reference = SparseGibbsReferenceProcess(phase_space)

    blank = product_bernoulli(np.asarray(config["blank_theta"], dtype=float))
    block = product_bernoulli(np.asarray(config["block_theta"], dtype=float))
    h_blank = joint_entropy(blank)
    h_block = joint_entropy(block)
    fan_in = Reachability.kernel_fan_in(reference._kernel)
    envelope = Reachability.entropy_envelope(
        h_blank,
        h_block,
        num_steps,
        fan_in,
        num_tokens,
        2,
        functional="joint",
    )
    cap = float(config["schedule_cap_fraction"]) * float(envelope[num_steps - 1])
    hard_schedule = monotone_schedule(h_blank, cap, num_steps)
    aggressive_schedule = monotone_schedule(
        h_blank,
        float(config["aggressive_cap"]),
        num_steps,
    )
    zeros = np.zeros(num_steps + 1)
    common = {
        "phase_space": phase_space,
        "reference": reference,
        "initial": blank,
        "terminal": block,
        "max_cycles": int(config["ipf_max_cycles"]),
        "tolerance": float(config["ipf_tol"]),
    }

    print("3C: unconstrained Blank -> Block", flush=True)
    run_3c = solve(
        schedule=zeros,
        projection=JointEntropyProjection(),
        on_infeasible="raise",
        **common,
    )
    print("3E: reachable hard joint-entropy schedule", flush=True)
    run_3e = solve(
        schedule=hard_schedule,
        projection=JointEntropyProjection(),
        on_infeasible="raise",
        **common,
    )
    print("3G: hinge penalty at unreachable 6-nat schedule", flush=True)
    run_3g = solve(
        schedule=aggressive_schedule,
        projection=HingePenaltyProjection(float(config["hinge_beta"])),
        on_infeasible="ignore",
        **common,
    )

    checks = {
        "all_runs_converged": all(
            result["converged"] for result in (run_3c, run_3e, run_3g)
        ),
        "all_endpoint_errors_below_1e-6": max(
            max(result["endpoint_maximum_absolute_error"])
            for result in (run_3c, run_3e, run_3g)
        )
        < 1e-6,
        "hard_schedule_satisfied_to_1e-5": max(
            (deficit for _, deficit in run_3e["schedule_deficits"]),
            default=0.0,
        )
        < 1e-5,
        "hinge_records_unreachable_shortfall": bool(run_3g["schedule_deficits"]),
        "all_increment_bounds_satisfied": all(
            result["increment_bound_satisfied"]
            for result in (run_3c, run_3e, run_3g)
        ),
    }
    return {
        "description": "Section 6 binary 4x4 Blank-to-Block experiment.",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "state_space": {
            "alphabet_size": 2,
            "num_tokens": num_tokens,
            "num_states": phase_space.num_states,
            "num_steps": num_steps,
            "reference_nonzeros_per_row": int(fan_in),
        },
        "endpoint_joint_entropy": {"blank": h_blank, "block": h_block},
        "reachable_envelope": [float(x) for x in envelope],
        "hard_cap": cap,
        "hard_schedule": hard_schedule.tolist(),
        "aggressive_schedule": aggressive_schedule.tolist(),
        "hinge_beta": float(config["hinge_beta"]),
        "runs": {"3C": run_3c, "3E": run_3e, "3G": run_3g},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate the figure from --results instead of solving",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=HERE / "expected" / "paper_results.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.figures_only:
        payload = json.loads(args.results.read_text(encoding="utf-8"))
    else:
        payload = run(config)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "results.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    make_figure(
        payload["runs"]["3C"],
        payload["runs"]["3E"],
        np.asarray(payload["hard_schedule"]),
        [int(t) for t in config["display_steps"]],
        int(config["grid_rows"]),
        int(config["grid_cols"]),
        args.output_dir,
    )
    print(f"Outputs: {args.output_dir}")
    if args.figures_only:
        return 0
    print(f"Validation passed: {payload['all_checks_passed']}")
    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
