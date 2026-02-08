"""Tool: generate_visualizations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.schemas import PlotSpec, RankingResults, VisualizationOutput


def _stable_block_id(prefix: str, payload: str) -> str:
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _ranking_bar(results: RankingResults, artifact_dir: Path) -> PlotSpec:
    order = np.argsort(results.ranks)
    names = [results.items[i] for i in order]
    scores = [results.theta_hat[i] for i in order]
    lower = [results.ci_lower[i] for i in order]
    upper = [results.ci_upper[i] for i in order]

    y = np.arange(len(names))
    err_left = np.maximum(np.array(scores) - np.array(lower), 0)
    err_right = np.maximum(np.array(upper) - np.array(scores), 0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
    ax.barh(y, scores, color="#2f6da3", alpha=0.85)
    ax.errorbar(scores, y, xerr=[err_left, err_right], fmt="none", ecolor="#222222", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("theta_hat")
    ax.set_title("Ranking Scores with Uncertainty")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    payload = "|".join(names) + ":" + "|".join(f"{value:.6f}" for value in scores)
    block_id = _stable_block_id("figure-ranking-bar", payload)
    svg_path = artifact_dir / f"{block_id}.svg"
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    return PlotSpec(
        type="ranking_bar",
        data={"names": names, "scores": scores, "ci_lower": lower, "ci_upper": upper},
        config={"x_label": "theta_hat"},
        svg_path=str(svg_path),
        block_id=block_id,
        caption_plain="Higher bars indicate stronger estimated preference.",
        caption_academic="Bar plot of estimated preference scores with uncertainty whiskers.",
        hint_ids=["hint-theta-hat", "hint-ci"],
    )


def _ci_forest(results: RankingResults, artifact_dir: Path) -> PlotSpec:
    order = np.argsort(results.ranks)
    names = [results.items[i] for i in order]
    scores = [results.theta_hat[i] for i in order]
    lower = [results.ci_lower[i] for i in order]
    upper = [results.ci_upper[i] for i in order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
    ax.hlines(y, lower, upper, color="#555555", linewidth=2)
    ax.scatter(scores, y, color="#0c7da0", s=28, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Rank confidence interval")
    ax.set_title("Forest Plot of Confidence Intervals")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    payload = "|".join(names) + ":" + "|".join(f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(lower, upper))
    block_id = _stable_block_id("figure-ci-forest", payload)
    svg_path = artifact_dir / f"{block_id}.svg"
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    return PlotSpec(
        type="ci_forest",
        data={"names": names, "theta_hat": scores, "ci_lower": lower, "ci_upper": upper},
        config={"x_label": "rank"},
        svg_path=str(svg_path),
        block_id=block_id,
        caption_plain="Lines show uncertainty ranges for each item's inferred rank.",
        caption_academic="Forest plot of confidence intervals and point estimates across ranked items.",
        hint_ids=["hint-ci"],
    )


def generate_visualizations(
    results: RankingResults,
    viz_types: list[str],
    artifact_dir: str,
) -> VisualizationOutput:
    """Create deterministic SVG artifacts from ranking results."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[PlotSpec] = []
    errors: list[str] = []

    for viz_type in viz_types:
        try:
            if viz_type == "ranking_bar":
                plots.append(_ranking_bar(results, output_dir))
            elif viz_type == "ci_forest":
                plots.append(_ci_forest(results, output_dir))
            else:
                errors.append(f"Unsupported viz_type: {viz_type}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{viz_type} failed: {exc}")

    return VisualizationOutput(plots=plots, errors=errors)
