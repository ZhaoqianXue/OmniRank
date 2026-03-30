#!/usr/bin/env python3
"""
Aggregate LM Arena wide pairwise CSVs from the Ranking repo and plot battle-count heatmaps.

Expected layout (default):
  <Ranking>/data_llm/data_arena/data_processing/all_combinations/arena_spectral_<category>.csv

Each row is one battle: exactly two model columns are non-null (0/1). The first column is metadata
(_category) and is ignored for pairing.

By default we sum counts across the seven single-tag Arena slices only (no combination subsets):
  coding, creative_writing, hard_prompt, instruction_following, longer_query, math, multi_turn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Single-slice Arena categories (do not include multi-tag combination files)
DEFAULT_ARENA_SLICES = (
    "coding",
    "creative_writing",
    "hard_prompt",
    "instruction_following",
    "longer_query",
    "math",
    "multi_turn",
)


def _resolve_ranking_root(script_dir: Path) -> Path:
    """OmniRank repo root -> sibling ../Ranking when present."""
    omnirank_root = script_dir.parent
    sibling = omnirank_root.parent / "Ranking"
    if sibling.is_dir():
        return sibling
    return omnirank_root / "Ranking"


def _model_columns(df: pd.DataFrame, meta_col: str) -> list[str]:
    return [c for c in df.columns if c != meta_col]


def order_models_like_web_leaderboard(ranking_json: Path, present: set[str]) -> list[str]:
    """
    Same ordering as src/web/lib/leaderboard-data.ts loadArenaBaseMethods:
    sort by `rank` ascending from arena ranking_results.json.
    Models in the matrix but missing from JSON are appended alphabetically.
    """
    raw = json.loads(ranking_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {ranking_json}")

    rows = sorted(raw, key=lambda r: float(r.get("rank", 1e9)))
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = row.get("model")
        if not isinstance(name, str):
            continue
        if name in present and name not in seen:
            ordered.append(name)
            seen.add(name)

    extras = sorted(present - seen)
    return ordered + extras


def aggregate_pair_counts(
    csv_paths: list[Path],
    meta_col: str,
) -> tuple[pd.DataFrame, dict[str, int], int, int]:
    """Returns symmetric count_df, participation, skipped_rows, total_battles."""
    models: set[str] = set()
    for p in csv_paths:
        header = pd.read_csv(p, nrows=0)
        for c in header.columns:
            if c != meta_col:
                models.add(c)
    model_list = sorted(models)
    idx = {m: i for i, m in enumerate(model_list)}
    n = len(model_list)
    mat = np.zeros((n, n), dtype=np.int64)
    participation = {m: 0 for m in model_list}
    skipped = 0
    total_battles = 0

    for p in csv_paths:
        df = pd.read_csv(p)
        if meta_col not in df.columns:
            raise ValueError(f"Missing meta column {meta_col!r} in {p}")
        cols = _model_columns(df, meta_col)

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
            total_battles += 1

    count_df = pd.DataFrame(mat, index=model_list, columns=model_list)
    return count_df, participation, skipped, total_battles


def plot_heatmap(
    count_df: pd.DataFrame,
    out_path: Path,
    title: str,
    figsize: tuple[float, float],
    annot_fontsize: float,
    axis_tick_fontsize: float,
) -> None:
    models = list(count_df.index)
    n = len(models)
    annot = np.empty((n, n), dtype=object)
    data = count_df.values.astype(float).copy()
    # Diagonal: no self-pairing; mask for display and leave annotations empty.
    np.fill_diagonal(data, np.nan)
    vmax_raw = np.nanmax(data)
    vmax = float(vmax_raw) if not np.isnan(vmax_raw) else 1.0

    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = ""
            else:
                c = int(count_df.iloc[i, j])
                annot[i, j] = str(c) if c > 0 else ""

    try:
        cmap = plt.colormaps["YlOrRd"].copy()
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#e8e8e8")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=vmax if vmax > 0 else 1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(models, rotation=75, ha="right", fontsize=axis_tick_fontsize)
    ax.set_yticklabels(models, fontsize=axis_tick_fontsize)

    for i in range(n):
        for j in range(n):
            text = annot[i, j]
            if text == "":
                continue
            v = data[i, j]
            color = "white" if not np.isnan(v) and v > vmax * 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=annot_fontsize)

    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Battles between distinct model pair")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_ranking = _resolve_ranking_root(script_dir)
    default_arena_dir = default_ranking / "data_llm" / "data_arena" / "data_processing" / "all_combinations"

    parser = argparse.ArgumentParser(description="LM Arena battle-count heatmap from Ranking CSVs.")
    parser.add_argument(
        "--arena-dir",
        type=Path,
        default=default_arena_dir,
        help="Directory containing arena_spectral_*.csv",
    )
    parser.add_argument(
        "--slices",
        nargs="*",
        default=list(DEFAULT_ARENA_SLICES),
        help="Single-tag slice names (files arena_spectral_<name>.csv)",
    )
    parser.add_argument("--meta-col", default="_category", help="First metadata column to ignore")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir.parent / "output" / "arena_lm_battle_count_heatmap.png",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=script_dir.parent / "output" / "arena_lm_battle_counts_matrix.csv",
    )
    parser.add_argument("--figwidth", type=float, default=22.0)
    parser.add_argument("--figheight", type=float, default=20.0)
    parser.add_argument("--annot-fontsize", type=float, default=4.0)
    parser.add_argument(
        "--axis-tick-fontsize",
        type=float,
        default=10.0,
        help="Font size for model names on x/y axes",
    )
    parser.add_argument(
        "--leaderboard-json",
        type=Path,
        default=script_dir.parent
        / "data"
        / "leaderboard"
        / "arena"
        / "data_ranking"
        / "current"
        / "ranking_results.json",
        help="Arena ranking_results.json (same source as the web leaderboard); rows/columns follow its rank order",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for name in args.slices:
        p = args.arena_dir / f"arena_spectral_{name}.csv"
        if not p.is_file():
            raise FileNotFoundError(f"Missing expected file: {p}")
        paths.append(p)

    count_df, _participation, skipped, total_battles = aggregate_pair_counts(paths, args.meta_col)

    if not args.leaderboard_json.is_file():
        raise FileNotFoundError(
            f"Leaderboard JSON not found: {args.leaderboard_json}. "
            "Pass --leaderboard-json to arena data_ranking/current/ranking_results.json."
        )
    row_order = order_models_like_web_leaderboard(args.leaderboard_json, set(count_df.index))
    count_df = count_df.reindex(index=row_order, columns=row_order)

    title = (
        f"LM Arena pairwise battle counts (sum over {len(paths)} category slices; "
        f"{total_battles} battles). Rows/cols = web leaderboard rank order. Diagonal N/A."
    )
    plot_heatmap(
        count_df,
        args.output,
        title=title,
        figsize=(args.figwidth, args.figheight),
        annot_fontsize=args.annot_fontsize,
        axis_tick_fontsize=args.axis_tick_fontsize,
    )
    count_df.to_csv(args.csv_out)

    print(f"Leaderboard order from: {args.leaderboard_json}")
    print(f"Ranking arena dir: {args.arena_dir}")
    print(f"Files: {len(paths)} slices")
    print(f"Models: {len(count_df)}")
    print(f"Total battles (rows parsed): {total_battles}")
    print(f"Skipped rows (not exactly two model cells): {skipped}")
    print(f"Heatmap: {args.output}")
    print(f"Matrix CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
