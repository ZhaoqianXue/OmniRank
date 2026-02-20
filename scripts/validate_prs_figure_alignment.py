#!/usr/bin/env python3
"""
Validate alignment targets for PRS reference figures:
1) Figure 1(A): General PRS Method Ranking (point ranks).
2) Figure 2(A/B/C): phenotype-normalized ranking + continuous/binary heatmaps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_meta(root: Path) -> dict:
    meta_path = root / "data" / "examples" / "example_data_multiway_phenotype_meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _validate_phenotype_panels(meta: dict, mean_tol: float) -> list[str]:
    errors: list[str] = []

    violin_order = meta.get("violin_method_order", [])
    if "AnnoPred" in violin_order:
        errors.append("Figure 2(A): violin order must exclude AnnoPred.")
    if len(violin_order) != 13:
        errors.append(f"Figure 2(A): expected 13 methods in violin order, got {len(violin_order)}.")

    heatmap_order = meta.get("heatmap_method_order", [])
    if len(heatmap_order) != 14:
        errors.append(f"Figure 2(B/C): expected 14 methods in heatmap order, got {len(heatmap_order)}.")
    if "AnnoPred" not in heatmap_order:
        errors.append("Figure 2(B/C): heatmap order must include AnnoPred.")

    continuous = meta.get("continuous_phenotypes", [])
    binary = meta.get("binary_phenotypes", [])
    if len(continuous) != 15:
        errors.append(f"Figure 2(B): expected 15 continuous phenotypes, got {len(continuous)}.")
    if len(binary) != 17:
        errors.append(f"Figure 2(C): expected 17 binary phenotypes, got {len(binary)}.")

    rank_matrix = {
        **meta.get("heatmap_rank_matrix_continuous", {}),
        **meta.get("heatmap_rank_matrix_binary", {}),
    }
    allowed_ranks = set(meta.get("heatmap_discrete_ranks", []))
    for pheno, mvals in rank_matrix.items():
        for method, rank in mvals.items():
            if rank not in allowed_ranks:
                errors.append(
                    f"Figure 2(B/C): invalid rank value {rank} at phenotype={pheno}, method={method}."
                )
                break
        if errors:
            break

    coverage = meta.get("method_coverage", {})
    violin_spec = meta.get("violin_spec", {})
    for method, spec in violin_spec.items():
        cov = coverage.get(method, [])
        if len(cov) != int(spec["K"]):
            errors.append(
                f"Figure 2(A): K mismatch for {method}, expected {spec['K']}, got {len(cov)}."
            )
            continue
        vals = [rank_matrix[p][method] for p in cov if method in rank_matrix.get(p, {})]
        if len(vals) != len(cov):
            errors.append(f"Figure 2(A): coverage/rank-matrix mismatch for {method}.")
            continue
        mean_val = sum(vals) / len(vals)
        delta = abs(mean_val - float(spec["lambda_mean"]))
        if delta > mean_tol:
            errors.append(
                f"Figure 2(A): mean mismatch for {method}, target={spec['lambda_mean']}, "
                f"actual={mean_val:.3f}, delta={delta:.3f} > tol={mean_tol:.3f}."
            )

    return errors


def _run_general_ranking(root: Path, csv_path: Path, bootstrap: int, seed: int) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="omnirank_validate_") as tmp:
        out_dir = Path(tmp)
        cmd = [
            "Rscript",
            "src/spectral_ranking/spectral_ranking.R",
            "--csv",
            str(csv_path),
            "--bigbetter",
            "1",
            "--B",
            str(bootstrap),
            "--seed",
            str(seed),
            "--out",
            str(out_dir),
        ]
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R ranking failed.\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return pd.read_csv(out_dir / "ranking_results.csv")


def _validate_general_panel(meta: dict, ranking_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    target = meta.get("general_prs_rank_target", {})
    actual = {
        str(row["method"]): int(row["rank"])
        for _, row in ranking_df.iterrows()
    }
    if actual != target:
        errors.append(
            "Figure 1(A): point-rank mismatch.\n"
            f"target={target}\nactual={actual}"
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PRS figure alignment targets.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations for spectral ranking.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for spectral ranking.")
    parser.add_argument(
        "--mean-tol",
        type=float,
        default=0.02,
        help="Max allowed abs deviation for Figure 2(A) lambda_mean checks.",
    )
    args = parser.parse_args()

    root = _root_dir()
    csv_path = root / "data" / "examples" / "example_data_multiway_phenotype.csv"
    meta = _load_meta(root)

    errors = _validate_phenotype_panels(meta, mean_tol=args.mean_tol)
    ranking_df = _run_general_ranking(root, csv_path, bootstrap=args.bootstrap, seed=args.seed)
    errors.extend(_validate_general_panel(meta, ranking_df))

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    ranking_df = ranking_df.sort_values("rank")
    print("Validation passed.")
    print("Figure 1(A) rank mapping:")
    for _, row in ranking_df.iterrows():
        print(f"  {row['method']}: rank={int(row['rank'])}, ci=[{int(row['ci_two_left'])},{int(row['ci_two_right'])}]")


if __name__ == "__main__":
    main()

