from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

from four_by_four.run_experiment import erasure_entropy, joint_entropy
from PhaseSpace.DiscretePhaseSpace import DiscretePhaseSpace
from ReferenceProcess.SparseGibbsReferenceProcess import SparseGibbsReferenceProcess
from Solver.Reachability import kernel_fan_in


def test_product_law_entropies() -> None:
    p = np.full(4, 0.25)
    assert np.isclose(joint_entropy(p), np.log(4))
    assert np.isclose(erasure_entropy(p, 2), np.log(2))


def test_binary_gibbs_kernel_has_self_loop_and_one_bit_updates() -> None:
    phase_space = DiscretePhaseSpace(n=3, alphabet_size=2, num_steps=3)
    reference = SparseGibbsReferenceProcess(phase_space)
    kernel = reference._kernel
    assert kernel.shape == (8, 8)
    assert kernel_fan_in(kernel) == 4
    assert np.allclose(np.asarray(kernel.sum(axis=1)).ravel(), 1.0)
