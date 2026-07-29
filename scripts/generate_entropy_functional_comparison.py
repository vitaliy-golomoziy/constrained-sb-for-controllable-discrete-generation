#!/usr/bin/env python3
"""Generate the paper's exact 4x4 joint-versus-erasure-entropy comparison."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import reportlab
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "scripts" / "output" / "entropy_functional_comparison.pdf"
K = 16


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def solve_bernoulli_parameter(target: float) -> float:
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if binary_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def translated_blocks() -> list[list[int]]:
    words: list[list[int]] = []
    for row in range(4):
        for col in range(4):
            grid = [[0 for _ in range(4)] for _ in range(4)]
            for di in (0, 1):
                for dj in (0, 1):
                    grid[(row + di) % 4][(col + dj) % 4] = 1
            words.append([bit for grid_row in grid for bit in grid_row])
    return words


def hamming_distance(left: list[int], right: list[int]) -> int:
    return sum(a != b for a, b in zip(left, right))


def random_codebook(min_distance: int = 4) -> list[list[int]]:
    rng = random.Random(20260729)
    words: list[list[int]] = []
    while len(words) < 16:
        candidate = [rng.getrandbits(1) for _ in range(K)]
        if all(hamming_distance(candidate, word) >= min_distance for word in words):
            words.append(candidate)
    return words


def minimum_distance(words: list[list[int]]) -> int:
    return min(
        hamming_distance(words[i], words[j])
        for i in range(len(words))
        for j in range(i + 1, len(words))
    )


def erasure_entropy_of_uniform_codebook(words: list[list[int]]) -> float:
    total = 0.0
    for k in range(K):
        contexts: dict[tuple[int, ...], list[int]] = {}
        for word in words:
            context = tuple(word[:k] + word[k + 1 :])
            counts = contexts.setdefault(context, [0, 0])
            counts[word[k]] += 1
        conditional = 0.0
        for counts in contexts.values():
            mass = sum(counts) / len(words)
            probabilities = [count / sum(counts) for count in counts if count]
            conditional += mass * (
                -sum(prob * math.log(prob) for prob in probabilities)
            )
        total += conditional
    return total / K


def draw_grid(pdf: canvas.Canvas, word: list[int], x: float, y: float) -> None:
    cell = 11.5
    light = HexColor("#f7f7f7")
    dark = HexColor("#222222")
    grid_color = HexColor("#b0b0b0")
    for row in range(4):
        for col in range(4):
            pdf.setFillColor(dark if word[4 * row + col] else light)
            pdf.setStrokeColor(grid_color)
            pdf.setLineWidth(0.45)
            pdf.rect(
                x + col * cell,
                y + (3 - row) * cell,
                cell,
                cell,
                fill=1,
                stroke=1,
            )
    pdf.setFillColor(light)
    pdf.setStrokeColor(HexColor("#555555"))
    pdf.setLineWidth(0.8)
    pdf.rect(x, y, 4 * cell, 4 * cell, fill=0, stroke=1)


def generate(output: Path) -> None:
    blocks = translated_blocks()
    random_words = random_codebook()
    target_per_token = math.log(16.0) / K
    bernoulli_p = solve_bernoulli_parameter(target_per_token)

    assert minimum_distance(blocks) >= 2
    assert minimum_distance(random_words) >= 2
    assert abs(erasure_entropy_of_uniform_codebook(blocks)) < 1e-12
    assert abs(erasure_entropy_of_uniform_codebook(random_words)) < 1e-12
    assert abs(binary_entropy(bernoulli_p) - target_per_token) < 1e-12

    sample_rng = random.Random(271828)
    bernoulli_samples = [
        [int(sample_rng.random() < bernoulli_p) for _ in range(K)]
        for _ in range(6)
    ]
    rows = [
        (
            "Translated 2 x 2 blocks",
            "structured codebook",
            [blocks[index] for index in (0, 1, 5, 7, 10, 15)],
            0.0,
        ),
        (
            "Random codewords",
            "unstructured codebook",
            random_words[:6],
            0.0,
        ),
        (
            f"Independent pixels (p = {bernoulli_p:.3f})",
            "independent noise",
            bernoulli_samples,
            target_per_token,
        ),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont("FigureSerif", font_dir / "Vera.ttf"))
    pdfmetrics.registerFont(TTFont("FigureSerif-Bold", font_dir / "VeraBd.ttf"))
    width, height = 760.0, 330.0
    pdf = canvas.Canvas(str(output), pagesize=(width, height))
    pdf.setTitle("Joint versus erasure entropy")
    pdf.setAuthor("Vitaliy Golomoziy and Yevhenii Azarov")

    pdf.setFillColor(HexColor("#202020"))
    pdf.setFont("FigureSerif-Bold", 13)
    pdf.drawCentredString(
        width / 2,
        307,
        "Equal joint entropy H = ln 16: different conditional structure",
    )

    row_y = (220.0, 128.0, 36.0)
    sample_x = (225.0, 308.0, 391.0, 474.0, 557.0, 640.0)
    for y, (title, subtitle, samples, erasure) in zip(row_y, rows):
        pdf.setFillColor(HexColor("#202020"))
        pdf.setFont("FigureSerif-Bold", 10.5)
        pdf.drawString(18, y + 35, title)
        pdf.setFillColor(HexColor("#555555"))
        pdf.setFont("FigureSerif", 9.5)
        pdf.drawString(18, y + 20, subtitle)
        pdf.setFillColor(HexColor("#202020"))
        pdf.drawString(
            18,
            y + 2,
            f"H/K = {target_per_token:.3f}, erasure = {erasure:.3f} nats",
        )
        for x, sample in zip(sample_x, samples):
            draw_grid(pdf, sample, x, y)

    pdf.setFillColor(HexColor("#444444"))
    pdf.setFont("FigureSerif", 8.5)
    pdf.drawCentredString(
        width / 2,
        11,
        "Joint entropy is identical in all rows. Erasure entropy detects independent "
        "residual noise, but not visual structure.",
    )
    pdf.showPage()
    pdf.save()

    print(f"Wrote {output}")
    print(
        "metrics:",
        {
            "joint_entropy": math.log(16.0),
            "joint_entropy_per_token": target_per_token,
            "block_min_distance": minimum_distance(blocks),
            "random_min_distance": minimum_distance(random_words),
            "bernoulli_p": bernoulli_p,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
