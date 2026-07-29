# Entropy-Constrained Schrödinger Bridges: paper code

This is the publication-facing code for *Entropy-Constrained Schrödinger
Bridges for Controllable Discrete Generation*. It contains only the three
workflows reported in the paper:

1. exact toy validation (Section 5);
2. binary \(4\times4\) Blank-to-Block bridges (Section 6);
3. the amortized MNIST heuristic and its schedule ablations (Section 7).

The exploratory experiments from the research workspace are deliberately
excluded.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For MNIST, install the additional PyTorch dependencies:

```bash
pip install -r requirements-mnist.txt
```

For the smoke tests, install `requirements-dev.txt` and run `pytest`.

## 1. Exact toy validation

This compares the complete path-space CVXPY solution against the factorized
IPF--Dykstra solver on the two cases in Table 1.

```bash
python toy/run_validation.py --output-dir toy/output
```

The command exits unsuccessfully if any numerical validation fails. The
recorded paper outputs are in [`toy/expected`](toy/expected).

## 2. Binary 4x4 experiment

This runner materializes only the three Blank-to-Block cases used in Section 6:

- **3C:** unconstrained bridge;
- **3E:** hard, feasible joint-entropy schedule;
- **3G:** hinge penalty at a deliberately unreachable schedule.

```bash
python four_by_four/run_experiment.py
```

The state space contains \(2^{16}=65{,}536\) states. A typical CPU run takes
about one minute. Results and the Blank-to-Block figure are written to
`four_by_four/output/`. To regenerate only the figure from the recorded result:

```bash
python four_by_four/run_experiment.py --figures-only
```

The script verifies solver convergence, endpoint accuracy, the hard constraint
to numerical tolerance, the expected shortfall of the hinge run, and the
one-step reachable-entropy bound.

## 3. MNIST experiment

The MNIST workflow has three training stages:

```bash
python mnist/train_vqvae.py
python mnist/train_masked_model.py
python mnist/train_noise_mtms.py
```

The first command downloads MNIST, trains the VQ-VAE, and saves the \(7\times7\)
token arrays. The second trains the clean-token masked model. The third creates
the \(T+1=9\) stored models: the clean model at \(t=8\), seven independently
fine-tuned intermediate models, and the \(t=0\) model whose predictions are
ignored in favor of the uniform source.

Run the five reported generation conditions with:

```bash
python mnist/run_ablations.py
```

The paper checkpoints are included according to
[`mnist/checkpoints/README.md`](mnist/checkpoints/README.md). A different set
can be supplied from another directory:

```bash
python mnist/run_ablations.py --checkpoint-dir /path/to/checkpoints
```

The runner records its random seed, numerical settings, and all entropy
profiles in `mnist/output/results.json`. The quantity plotted here is the
model-based predictive-entropy proxy in equation (17), not the exact erasure
entropy of an unknown Schrödinger bridge marginal. The MNIST construction is an
amortized heuristic; its sampling laws \(Q_t\) are not asserted to approximate
the exact optimum \(P_t\). Stochastic generation and device-specific kernels
mean newly trained checkpoints need not reproduce the rounded paper values
exactly. The recorded outputs and figures are in [`mnist/expected`](mnist/expected).

## Repository map

```text
modules/          minimal shared implementation
toy/              Section 5 exact validation
four_by_four/     Section 6 runs 3C, 3E, and 3G
mnist/            Section 7 training and ablations
```

## Before public release

Choose and add a `LICENSE` file. No license has been selected automatically,
because that is an author decision.
