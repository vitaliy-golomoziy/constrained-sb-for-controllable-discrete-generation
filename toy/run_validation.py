#!/usr/bin/env python3
"""Validate the factorized IPF--Dykstra solver against direct optimization.

The cases are deliberately small enough that the complete path law can be
materialized and optimized by CVXPY.  This gives an enumerated path-space
baseline for the factorized solver used by the scalable formulation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from Formulation.HardConstraintFormulation import HardConstraintFormulation
from Functional.ConditionalEntropyFunctional import ConditionalEntropyFunctional
from Objective.KLDivergence import KLDivergence
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from ReferenceProcess.GibbsReferenceProcess import GibbsReferenceProcess
from SBProblem.SBProblem import SBProblem
from Solver.EntropyProjection import EntropyProjection
from Solver.IPFDykstraSolver import IPFDykstraSolver


@dataclass(frozen=True)
class ValidationCase:
    name: str
    alphabet_size: int
    num_tokens: int
    num_steps: int
    stationary: np.ndarray
    initial: np.ndarray
    terminal: np.ndarray
    schedule: np.ndarray


def validation_cases() -> dict[str, ValidationCase]:
    binary_stationary = np.array([0.40, 0.10, 0.10, 0.40])
    binary = ValidationCase(
        name="binary",
        alphabet_size=2,
        num_tokens=2,
        num_steps=3,
        stationary=binary_stationary,
        initial=binary_stationary.copy(),
        terminal=binary_stationary.copy(),
        schedule=np.array([0.00, 0.60, 0.60, 0.00]),
    )

    ternary_table = 0.05 * np.ones((3, 3)) + 0.25 * np.eye(3)
    ternary_table[0, 0] += 0.10
    ternary_stationary = (ternary_table / ternary_table.sum()).ravel()
    ternary = ValidationCase(
        name="ternary",
        alphabet_size=3,
        num_tokens=2,
        num_steps=3,
        stationary=ternary_stationary,
        initial=ternary_stationary.copy(),
        terminal=ternary_stationary.copy(),
        schedule=np.array([0.00, 0.75, 0.75, 0.00]),
    )
    return {case.name: case for case in (binary, ternary)}


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def entropy_profile(distribution, functional, phase_space) -> list[float]:
    return [
        float(functional(distribution.time_marginal(t), phase_space))
        for t in range(phase_space.num_marginals)
    ]


def solve_direct(problem, initial, terminal, schedule):
    tolerance = (
        1e-9 if problem.phase_space.num_trajectories <= 1_000 else 1e-7
    )
    try:
        return problem.solve(
            initial,
            terminal,
            schedule,
            solver="CLARABEL",
            max_iter=1_000,
            tol_gap_abs=tolerance,
            tol_gap_rel=tolerance,
            tol_feas=tolerance,
        )
    except cp.error.SolverError:
        return problem.solve(
            initial,
            terminal,
            schedule,
            solver="SCS",
            eps=1e-6,
            max_iters=50_000,
        )


def validate_case(case: ValidationCase) -> dict[str, Any]:
    phase_space = DiscretePhaseSpace(
        n=case.num_tokens,
        alphabet_size=case.alphabet_size,
        num_steps=case.num_steps,
    )
    reference = GibbsReferenceProcess(phase_space, case.stationary)
    functional = ConditionalEntropyFunctional()
    direct_problem = SBProblem(
        phase_space,
        reference,
        functional,
        HardConstraintFormulation(),
    )

    zero_schedule = np.zeros(phase_space.num_marginals)
    start = time.perf_counter()
    unconstrained = solve_direct(
        direct_problem, case.initial, case.terminal, zero_schedule
    )
    unconstrained_seconds = time.perf_counter() - start

    start = time.perf_counter()
    direct = solve_direct(
        direct_problem, case.initial, case.terminal, case.schedule
    )
    direct_seconds = time.perf_counter() - start

    factorized_solver = IPFDykstraSolver(
        phase_space,
        reference,
        functional,
        projection=EntropyProjection(
            bisect_tol=1e-8,
            bisect_max_iter=40,
            inner_tol=1e-10,
        ),
        on_infeasible="raise",
    )
    start = time.perf_counter()
    factorized = factorized_solver.solve(
        case.initial,
        case.terminal,
        case.schedule,
        max_cycles=3_000,
        tol=1e-7,
    )
    factorized_seconds = time.perf_counter() - start

    unconstrained_entropy = entropy_profile(
        unconstrained, functional, phase_space
    )
    direct_entropy = entropy_profile(direct, functional, phase_space)
    factorized_entropy = entropy_profile(
        factorized, functional, phase_space
    )
    interior = range(1, phase_space.num_steps)

    direct_objective = KLDivergence.dynamic(direct, reference)
    factorized_objective = KLDivergence.dynamic(factorized, reference)
    marginal_tv = [
        total_variation(
            direct.time_marginal(t), factorized.time_marginal(t)
        )
        for t in range(phase_space.num_marginals)
    ]
    endpoint_tv = [
        total_variation(factorized.time_marginal(0), case.initial),
        total_variation(factorized.time_marginal(phase_space.num_steps), case.terminal),
    ]
    direct_violations = [
        max(0.0, float(case.schedule[t] - direct_entropy[t]))
        for t in interior
    ]
    factorized_violations = [
        max(0.0, float(case.schedule[t] - factorized_entropy[t]))
        for t in interior
    ]
    entropy_lifts = [
        float(case.schedule[t] - unconstrained_entropy[t])
        for t in interior
    ]

    metrics = {
        "case": case.name,
        "alphabet_size": case.alphabet_size,
        "num_tokens": case.num_tokens,
        "num_steps": case.num_steps,
        "states_per_checkpoint": phase_space.num_states,
        "path_variables_direct": phase_space.num_trajectories,
        "potential_entries_factorized": (
            phase_space.num_marginals * phase_space.num_states
        ),
        "schedule": case.schedule.tolist(),
        "unconstrained_entropy": unconstrained_entropy,
        "direct_entropy": direct_entropy,
        "factorized_entropy": factorized_entropy,
        "minimum_entropy_lift_over_unconstrained": min(entropy_lifts),
        "direct_objective": direct_objective,
        "factorized_objective": factorized_objective,
        "absolute_objective_gap": abs(
            direct_objective - factorized_objective
        ),
        "path_total_variation": total_variation(
            direct.full_probabilities(), factorized.full_probabilities()
        ),
        "maximum_marginal_total_variation": max(marginal_tv),
        "maximum_entropy_difference": max(
            abs(a - b) for a, b in zip(direct_entropy, factorized_entropy)
        ),
        "maximum_direct_constraint_violation": max(
            direct_violations, default=0.0
        ),
        "maximum_factorized_constraint_violation": max(
            factorized_violations, default=0.0
        ),
        "maximum_factorized_endpoint_error_tv": max(endpoint_tv),
        "factorized_cycles": factorized_solver.n_cycles,
        "factorized_converged": factorized_solver.converged,
        "factorized_final_residual": factorized_solver.final_residual,
        "seconds": {
            "unconstrained_direct": unconstrained_seconds,
            "constrained_direct": direct_seconds,
            "factorized": factorized_seconds,
        },
    }

    checks = {
        "constraint_is_genuinely_active": (
            metrics["minimum_entropy_lift_over_unconstrained"] > 1e-2
        ),
        "factorized_solver_converged": metrics["factorized_converged"],
        "objective_gap_below_5e-4": (
            metrics["absolute_objective_gap"] < 5e-4
        ),
        "path_tv_below_2e-3": metrics["path_total_variation"] < 2e-3,
        "marginal_tv_below_5e-4": (
            metrics["maximum_marginal_total_variation"] < 5e-4
        ),
        "entropy_difference_below_5e-4": (
            metrics["maximum_entropy_difference"] < 5e-4
        ),
        "constraint_violation_below_2e-5": (
            metrics["maximum_factorized_constraint_violation"] < 2e-5
        ),
        "endpoint_error_below_1e-7": (
            metrics["maximum_factorized_endpoint_error_tv"] < 1e-7
        ),
    }
    metrics["checks"] = checks
    metrics["passed"] = all(checks.values())
    return metrics


def scientific(value: float) -> str:
    return f"{value:.2e}"


def markdown_table(results: list[dict[str, Any]]) -> str:
    headings = [
        "Case",
        "(A,K,N)",
        "Path vars.",
        "Pot. entries",
        "Min. entropy lift",
        "KL direct",
        "Abs. KL gap",
        "Path TV",
        "Max viol.",
    ]
    alignment = [
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
    ]
    rows = [
        [
            result["case"],
            (
                f'({result["alphabet_size"]},'
                f'{result["num_tokens"]},{result["num_steps"]})'
            ),
            f'{result["path_variables_direct"]:,}',
            f'{result["potential_entries_factorized"]:,}',
            f'{result["minimum_entropy_lift_over_unconstrained"]:.3f}',
            f'{result["direct_objective"]:.6f}',
            scientific(result["absolute_objective_gap"]),
            scientific(result["path_total_variation"]),
            scientific(result["maximum_factorized_constraint_violation"]),
        ]
        for result in results
    ]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["all", *validation_cases().keys()],
        default="all",
        help="case to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="directory for results.json and table.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = validation_cases()
    selected = list(cases.values()) if args.case == "all" else [cases[args.case]]

    results = []
    for case in selected:
        print(f"Running {case.name} validation...", flush=True)
        results.append(validate_case(case))

    payload = {
        "description": (
            "Direct path-space CVXPY versus factorized IPF--Dykstra "
            "for active erasure-entropy constraints."
        ),
        "all_passed": all(result["passed"] for result in results),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    table = markdown_table(results)
    (args.output_dir / "table.md").write_text(table, encoding="utf-8")

    print()
    print(table, end="")
    print(f"\nValidation passed: {payload['all_passed']}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
