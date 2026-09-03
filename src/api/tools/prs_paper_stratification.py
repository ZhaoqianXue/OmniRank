"""Phenotype-specific ranking for the PRS benchmarking example.

Reproduces the procedure of Sebastian et al. (2026) (appliedRankings.ipynb):
within each phenotype, methods with fewer than ``MIN_COMPARISONS`` comparisons
are removed iteratively; a method that then wins all of its comparisons is
recorded as the leading method and removed so the rest can be ranked; and each
rank is normalized to ``SCALE * rank / len(retained)``.

This path is used only for the bundled PRS example, whose stratified ranking the
manuscript reports against that study. All other datasets keep the generic
stratified path in ``generate_visualizations``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRS_EXAMPLE_FILENAMES = frozenset({"prs_benchmarking_applied.csv"})
MIN_COMPARISONS = 10
SCALE = 13  # methods available to the phenotype-specific analysis in the source study
NON_PHENOTYPE_LABELS = frozenset({"Unclassified"})


def uses_paper_stratification(csv_path: str | None) -> bool:
    """True when the uploaded file is the bundled PRS benchmarking example."""
    if not csv_path:
        return False
    from pathlib import Path

    return Path(csv_path).name in PRS_EXAMPLE_FILENAMES


def _comparison_matrices(frame: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per ordered pair of methods reported in the same source row.

    Pairs with equal values contribute no comparison, matching ``diffValue = 0``.
    """
    aa: list[dict[str, float]] = []
    ww: list[dict[str, float]] = []
    for _, record in frame.iterrows():
        present = [m for m in methods if pd.notna(record[m])]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                left, right = present[i], present[j]
                diff = float(record[left]) - float(record[right])
                if diff == 0:
                    continue
                pair = dict.fromkeys(methods, 0.0)
                pair[left] = 1.0
                pair[right] = 1.0
                winner = dict.fromkeys(methods, 0.0)
                winner[left if diff > 0 else right] = 1.0
                aa.append(pair)
                ww.append(winner)
    return pd.DataFrame(aa, columns=methods), pd.DataFrame(ww, columns=methods)


def _stationary(aa: pd.DataFrame, ww: pd.DataFrame) -> np.ndarray | None:
    scale = aa.shape[0] + 2
    column_sums = aa.sum()
    if column_sums.empty or float(column_sums.max()) == 0:
        return None
    denominator = float(column_sums.max()) * 2 * scale
    names = list(aa.columns)
    size = len(names)
    transition = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            transition[i][j] = float(
                np.sum(aa[names[i]].values * aa[names[j]].values * ww[names[j]].values)
            ) / denominator
        transition[i][i] = 1 - transition[i].sum()
    values, vectors = np.linalg.eig(transition.T)
    selected = np.isclose(values, 1)
    if not selected.any():
        return None
    vector = vectors[:, selected][:, 0]
    total = vector.sum()
    if total == 0:
        return None
    return (vector / total).real


def _rank_one_phenotype(aa: pd.DataFrame, ww: pd.DataFrame) -> dict[str, float] | None:
    while True:
        sparse = [c for c in aa.columns if aa[c].sum() < MIN_COMPARISONS]
        if not sparse:
            break
        aa = aa.drop(columns=sparse)
        ww = ww.drop(columns=sparse)
    if aa.shape[0] == 0 or aa.shape[1] < 2:
        return None

    ordered: list[str] = []
    stationary: np.ndarray | list[float] = []
    while True:
        if aa.shape[1] == 1:
            ordered.append((aa.columns[0], 0.0))
            stationary = []
            break
        stationary = _stationary(aa, ww)
        if stationary is None:
            return None
        undefeated = next((k for k, v in enumerate(stationary) if np.isclose(v, 1.0)), None)
        if undefeated is None:
            break
        method = aa.columns[undefeated]
        ordered.append((method, float(len(stationary))))
        aa = aa.drop(columns=method)
        ww = ww.drop(columns=method)
    for k in range(len(stationary)):
        ordered.append((aa.columns[k], float(stationary[k])))
    ordered.sort(key=lambda pair: -pair[1])
    return {name: SCALE * (position + 1) / len(ordered) for position, (name, _) in enumerate(ordered)}


def stratified_normalized_ranks(
    frame: pd.DataFrame,
    indicator_col: str,
    methods: list[str],
) -> dict[str, dict[str, float]]:
    """Normalized within-phenotype ranks, keyed by phenotype then method."""
    result: dict[str, dict[str, float]] = {}
    for phenotype, subset in frame.groupby(indicator_col):
        label = str(phenotype)
        if label in NON_PHENOTYPE_LABELS:
            continue
        aa, ww = _comparison_matrices(subset, methods)
        if aa.empty:
            continue
        ranks = _rank_one_phenotype(aa, ww)
        if ranks:
            result[label] = ranks
    return result
