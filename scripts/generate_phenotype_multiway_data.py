#!/usr/bin/env python3
"""
Generate simulated multiway matrix with phenotype for PRS method ranking.
Data is reverse-engineered from reference images for identical output:
- Forest Plot: overall ranking (LDpred2#1 ... C+T#14)
- Violin (A): 13 methods (no AnnoPred), exact K and target lambda_mean labels
- Heatmaps (B/C): 14 methods, phenotype x method rank matrix, discrete 2,5,7,9,11,13
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

# Forest Plot order (14 methods, best to worst)
FOREST_ORDER = [
    "LDpred2", "AnnoPred", "lassosum2", "SCT", "LDpred-funct", "lassosum",
    "LDpred2-auto", "SBayesR", "PRS-CS", "DBSLMM", "PRS-CS-auto",
    "LDpred2-inf", "LDpred", "C+T",
]

# Figure 1(A) General PRS Method Ranking (point ranks only).
GENERAL_PRS_RANK_TARGET = {
    "LDpred2": 1,
    "AnnoPred": 2,
    "lassosum2": 3,
    "SCT": 4,
    "LDpred-funct": 5,
    "lassosum": 6,
    "LDpred2-auto": 7,
    "SBayesR": 8,
    "PRS-CS": 9,
    "DBSLMM": 10,
    "PRS-CS-auto": 11,
    "LDpred2-inf": 12,
    "LDpred": 13,
    "C+T": 14,
}

# Figure 1(A) display order (top -> bottom) in the reference panel.
GENERAL_PRS_METHOD_ORDER_PLOT = [
    "C+T",
    "LDpred",
    "lassosum",
    "AnnoPred",
    "PRS-CS-auto",
    "PRS-CS",
    "SBayesR",
    "SCT",
    "DBSLMM",
    "LDpred2-inf",
    "LDpred2-auto",
    "LDpred2",
    "LDpred-funct",
    "lassosum2",
]

# Violin (A): 13 methods only - AnnoPred excluded. Exact K and lambda_mean from image.
VIOLIN_SPEC = {
    "C+T": {"K": 27, "lambda_mean": 11.39},
    "LDpred": {"K": 13, "lambda_mean": 9.00},
    "lassosum": {"K": 26, "lambda_mean": 6.07},
    "PRS-CS": {"K": 24, "lambda_mean": 7.78},
    "PRS-CS-auto": {"K": 19, "lambda_mean": 7.11},
    "SBayesR": {"K": 23, "lambda_mean": 7.27},
    "SCT": {"K": 11, "lambda_mean": 5.55},
    "DBSLMM": {"K": 21, "lambda_mean": 8.84},
    "LDpred2": {"K": 30, "lambda_mean": 3.47},
    "LDpred2-auto": {"K": 28, "lambda_mean": 5.60},
    "LDpred2-inf": {"K": 19, "lambda_mean": 10.28},
    "LDpred-funct": {"K": 2, "lambda_mean": 5.49},
    "lassosum2": {"K": 11, "lambda_mean": 6.73},
}

# AnnoPred: only in heatmaps (B/C), not in violin. K=18 (many gray cells).
ANNOPRED_K = 18

CONTINUOUS_PHENOTYPES = [
    "Body Mass Index", "Height", "Intelligence", "Systolic Blood Pressure",
    "Hemoglobin A1c", "Cholesterol", "High-Density Lipoprotein",
    "Low-Density Lipoprotein", "Triglyceride",
    "Estimated Glomerular Filtration Rate", "Urate", "Eosinophil Count",
    "Platelet Count", "Red Blood Cell Count", "White Blood Cell Count",
]

BINARY_PHENOTYPES = [
    "Prostate Cancer", "Breast Cancer", "Coronary Artery Disease",
    "Attention-deficit/hyperactivity disorder", "Schizophrenia", "Depression",
    "Alzheimer's Disease", "Stroke", "Multiple Sclerosis",
    "Inflammatory Bowel Disease", "Rheumatoid Arthritis", "Asthma Disease",
    "Type 1 Diabetes", "Type 2 Diabetes", "Chronic Kidney Disease", "Gout",
    "General Certificate of Secondary Education",
]

ALL_PHENOTYPES = CONTINUOUS_PHENOTYPES + BINARY_PHENOTYPES  # 32
HEATMAP_DISCRETE_RANKS = [2, 5, 7, 9, 11, 13]  # Color scale from image


def _assign_coverage() -> dict[str, set[str]]:
    """Method -> set of K phenotypes."""
    rng = random.Random(42)
    coverage: dict[str, set[str]] = {}
    pheno_list = list(ALL_PHENOTYPES)
    for method, spec in VIOLIN_SPEC.items():
        coverage[method] = set(rng.sample(pheno_list, spec["K"]))
    coverage["AnnoPred"] = set(rng.sample(pheno_list, ANNOPRED_K))
    return coverage


def _values_for_mean(k: int, mu: float) -> list[int]:
    """
    Generate k values in HEATMAP_DISCRETE_RANKS with globally minimal mean error.

    This solves a bounded subset-sum DP over sums, then backtracks one optimal
    solution that minimizes abs(sum/k - mu).
    """
    target_sum = k * mu
    parents: list[dict[int, tuple[int, int]]] = [{0: (-1, -1)}]

    for _ in range(k):
        prev_layer = parents[-1]
        current_layer: dict[int, tuple[int, int]] = {}
        for prev_sum in sorted(prev_layer.keys()):
            for rank in HEATMAP_DISCRETE_RANKS:
                new_sum = prev_sum + rank
                # Keep first predecessor for deterministic reconstruction.
                if new_sum not in current_layer:
                    current_layer[new_sum] = (prev_sum, rank)
        parents.append(current_layer)

    final_layer = parents[-1]
    best_sum = min(final_layer.keys(), key=lambda s: (abs(s - target_sum), s))

    values: list[int] = []
    cur_sum = best_sum
    for step in range(k, 0, -1):
        prev_sum, rank = parents[step][cur_sum]
        values.append(rank)
        cur_sum = prev_sum
    values.reverse()
    return values


def _build_heatmap_rank_matrix(coverage: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    """
    (phenotype, method) -> rank in {2,5,7,9,11,13}.
    Violin: for method M, mean over its K phenotypes is as close as possible
    to target lambda_mean under the discrete rank palette.
    """
    rng = random.Random(123)
    matrix: dict[str, dict[str, int]] = {p: {} for p in ALL_PHENOTYPES}

    for method, spec in VIOLIN_SPEC.items():
        phenos = sorted(coverage[method])
        vals = _values_for_mean(spec["K"], spec["lambda_mean"])
        rng.shuffle(vals)
        for p, v in zip(phenos, vals):
            matrix[p][method] = v

    for pheno in sorted(coverage["AnnoPred"]):
        matrix[pheno]["AnnoPred"] = rng.choice([5, 7, 9])

    return matrix


def _rank_to_score(rank: int, n: int = 14) -> float:
    """Discrete rank (2-13) -> score for spectral ranking. Higher score = better."""
    return 0.30 + (n - rank) * 0.045


def _build_method_means(df: pd.DataFrame) -> list[str]:
    method_cols = [c for c in df.columns if c not in ("phenotype", "sample_id")]
    return list(df[method_cols].mean().sort_values(ascending=False).index)


def _verify_outputs(
    df: pd.DataFrame,
    coverage: dict[str, set[str]],
    rank_matrix: dict[str, dict[str, int]],
) -> None:
    # Structural checks that should always hold.
    if df["sample_id"].nunique() != len(df):
        raise RuntimeError("sample_id must be globally unique.")

    ordered_methods = _build_method_means(df)
    if ordered_methods != FOREST_ORDER:
        raise RuntimeError(
            f"Forest order mismatch. Expected {FOREST_ORDER}, got {ordered_methods}."
        )
    if {
        method: rank for rank, method in enumerate(ordered_methods, start=1)
    } != GENERAL_PRS_RANK_TARGET:
        raise RuntimeError(
            "General PRS rank target mismatch against Figure 1(A) point ranks."
        )

    allowed = set(HEATMAP_DISCRETE_RANKS)
    invalid_cells: list[tuple[str, str, int]] = []
    for pheno, methods in rank_matrix.items():
        for method, rank in methods.items():
            if rank not in allowed:
                invalid_cells.append((pheno, method, rank))
    if invalid_cells:
        first = invalid_cells[0]
        raise RuntimeError(
            f"Heatmap has invalid rank values. First bad cell: {first}."
        )

    # K coverage checks and lambda mean diagnostics.
    print("Violin lambda_mean verification:")
    for method, spec in VIOLIN_SPEC.items():
        phenos = sorted(coverage[method])
        vals = [rank_matrix[p][method] for p in phenos]
        k_actual = len(vals)
        k_target = spec["K"]
        if k_actual != k_target:
            raise RuntimeError(
                f"Coverage K mismatch for {method}: expected {k_target}, got {k_actual}."
            )
        mean = sum(vals) / k_actual if k_actual else 0.0
        delta = mean - spec["lambda_mean"]
        print(
            f"  {method}: K={k_actual} (target {k_target}), "
            f"mean={mean:.3f} (target {spec['lambda_mean']:.2f}, delta={delta:+.3f})"
        )


def main() -> None:
    score_rng = random.Random(999)
    coverage = _assign_coverage()
    rank_matrix = _build_heatmap_rank_matrix(coverage)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "example_data_multiway_phenotype.csv"

    rows: list[dict] = []
    samples_per_phenotype = 55
    # Calibrated so spectral-ranking output aligns with General PRS Method Ranking (panel A).
    forest_weight = 0.86
    pheno_weight = 1.0 - forest_weight

    for pheno_idx, pheno in enumerate(ALL_PHENOTYPES):
        pheno_ranks = rank_matrix.get(pheno, {})
        for s in range(samples_per_phenotype):
            row: dict = {"phenotype": pheno, "sample_id": f"p{pheno_idx:02d}_{s:04d}"}
            for i, method in enumerate(FOREST_ORDER):
                forest_rank = i + 1
                hr = pheno_ranks.get(method)
                if hr is not None:
                    pheno_score_rank = max(1, min(14, round(1 + 13 * (hr - 2) / 11)))
                else:
                    pheno_score_rank = 8  # NA/inapplicable: use mid (gray in heatmap)
                score = forest_weight * _rank_to_score(forest_rank) + pheno_weight * _rank_to_score(pheno_score_rank)
                row[method] = round(score + score_rng.gauss(0, 0.01), 6)
            rows.append(row)

    df = pd.DataFrame(rows)
    method_cols = [c for c in df.columns if c not in ("phenotype", "sample_id")]
    df = df[["phenotype", "sample_id"] + method_cols]
    _verify_outputs(df, coverage, rank_matrix)
    df.to_csv(out_path, index=False)

    # Meta: violin spec, heatmap matrix, method order for each panel
    violin_method_order = [m for m in VIOLIN_SPEC.keys()]
    heatmap_method_order = ["C+T", "LDpred", "lassosum", "AnnoPred", "PRS-CS", "PRS-CS-auto",
                           "SBayesR", "SCT", "DBSLMM", "LDpred2", "LDpred2-auto", "LDpred2-inf",
                           "LDpred-funct", "lassosum2"]

    heatmap_continuous: dict[str, dict[str, int]] = {}
    heatmap_binary: dict[str, dict[str, int]] = {}
    for pheno in CONTINUOUS_PHENOTYPES:
        raw = rank_matrix.get(pheno, {})
        heatmap_continuous[pheno] = {
            method: raw[method] for method in heatmap_method_order if method in raw
        }
    for pheno in BINARY_PHENOTYPES:
        raw = rank_matrix.get(pheno, {})
        heatmap_binary[pheno] = {
            method: raw[method] for method in heatmap_method_order if method in raw
        }

    meta = {
        "forest_plot_order": FOREST_ORDER,
        "general_prs_rank_target": GENERAL_PRS_RANK_TARGET,
        "general_prs_method_order_plot": GENERAL_PRS_METHOD_ORDER_PLOT,
        "violin_method_order": violin_method_order,
        "violin_spec": VIOLIN_SPEC,
        "heatmap_method_order": heatmap_method_order,
        "heatmap_discrete_ranks": HEATMAP_DISCRETE_RANKS,
        "continuous_phenotypes": CONTINUOUS_PHENOTYPES,
        "binary_phenotypes": BINARY_PHENOTYPES,
        "heatmap_rank_matrix_continuous": heatmap_continuous,
        "heatmap_rank_matrix_binary": heatmap_binary,
        "method_coverage": {m: sorted(s) for m, s in coverage.items()},
        "n_rows": len(df),
    }
    meta_path = out_dir / "example_data_multiway_phenotype_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWritten {len(df)} rows to {out_path}")
    print(f"Phenotypes: {len(ALL_PHENOTYPES)} ({len(CONTINUOUS_PHENOTYPES)} continuous + {len(BINARY_PHENOTYPES)} binary)")


if __name__ == "__main__":
    main()
