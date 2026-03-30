#!/usr/bin/env python3
"""
Build a symmetric battle-count matrix from a wide pairwise CSV and save an annotated heatmap.

For LM Arena data in the sibling Ranking repo, use `arena_lm_battle_count_heatmap.py` instead.

Each row must represent one pairwise comparison: exactly two model columns are non-null
(0/1 win indicators). The Task column (if present) is ignored for pairing.

Default uses data/examples/example_data_pairwise.csv (demo only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _model_columns(df: pd.DataFrame, task_col: str) -> list[str]:
    return [c for c in df.columns if c != task_col]


def build_pair_counts(df: pd.DataFrame, task_col: str = "Task") -> tuple[pd.DataFrame, dict[str, int], int]:
    """Returns (symmetric count matrix, participation counts per model, skipped rows)."""
    cols = _model_columns(df, task_col)
    models = sorted(cols)
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    mat = np.zeros((n, n), dtype=np.int64)
    participation: dict[str, int] = {m: 0 for m in models}
    skipped = 0

    for _, row in df.iterrows():
        present: list[tuple[str, float]] = []
        for c in cols:
            v = row[c]
            if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            present.append((c, val))

        if len(present) != 2:
            skipped += 1
            continue

        a, b = present[0][0], present[1][0]
        ia, ib = idx[a], idx[b]
        mat[ia, ib] += 1
        mat[ib, ia] += 1
        participation[a] += 1
        participation[b] += 1

    count_df = pd.DataFrame(mat, index=models, columns=models)
    return count_df, participation, skipped


def plot_heatmap(
    count_df: pd.DataFrame,
    participation: dict[str, int],
    out_path: Path,
    title: str,
    figsize: tuple[float, float],
) -> None:
    models = list(count_df.index)
    n = len(models)
    # Annotate: off-diagonal = shared battle count; diagonal = total battles for that model
    annot = np.empty((n, n), dtype=object)
    data = count_df.values.astype(float).copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = str(participation.get(models[i], 0))
                data[i, j] = np.nan
            else:
                c = int(count_df.iloc[i, j])
                annot[i, j] = str(c) if c > 0 else ""

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)

    for i in range(n):
        for j in range(n):
            text = annot[i, j]
            if text == "":
                continue
            color = "white" if not np.isnan(data[i, j]) and data[i, j] > np.nanmax(data) * 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pairwise battle count (off-diag)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise battle count heatmap from wide CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "examples" / "example_data_pairwise.csv",
        help="Wide pairwise CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "pairwise_battle_count_heatmap.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "pairwise_battle_counts_matrix.csv",
        help="Optional symmetric count matrix CSV (set empty to skip)",
    )
    parser.add_argument("--task-col", default="Task", help="Metadata column to exclude from models")
    parser.add_argument(
        "--title",
        default="Pairwise battle counts (off-diagonal); diagonal = total battles per model",
        help="Figure title",
    )
    parser.add_argument("--figwidth", type=float, default=12.0)
    parser.add_argument("--figheight", type=float, default=10.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.task_col not in df.columns:
        args.task_col = df.columns[0]

    count_df, participation, skipped = build_pair_counts(df, task_col=args.task_col)

    total_pairs = int(np.sum(np.triu(count_df.values, k=1)))
    print(f"Input: {args.input}")
    print(f"Models: {len(count_df)}")
    print(f"Parsed pairwise rows: {int(sum(participation.values()) // 2)} (each row = 1 battle)")
    print(f"Skipped rows (not exactly two non-null model cells): {skipped}")
    print(f"Sum of unique pair counts (upper triangle): {total_pairs}")

    plot_heatmap(
        count_df,
        participation,
        args.output,
        title=args.title,
        figsize=(args.figwidth, args.figheight),
    )
    print(f"Saved: {args.output}")

    if args.csv_out:
        count_df.to_csv(args.csv_out)
        print(f"Saved matrix: {args.csv_out}")


if __name__ == "__main__":
    main()
