#!/usr/bin/env python3
"""
Validate cluster_items and n_clusters from OmniRank ranking results.

Runs the full pipeline on two example datasets (pairwise, multiway_scores),
then verifies that the CI-overlap clustering is correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "src" / "api"
sys.path.insert(0, str(API_PATH))


def _ci_overlaps(ci_lo_a: float, ci_hi_a: float, ci_lo_b: float, ci_hi_b: float) -> bool:
    """Check if two CIs overlap (intervals intersect)."""
    return not (ci_hi_a < ci_lo_b or ci_hi_b < ci_lo_a)


def _validate_clusters(
    items: list[str],
    ranks: list[int],
    ci_lower: list[float],
    ci_upper: list[float],
    clusters: list[list[str]],
) -> tuple[bool, list[str]]:
    """
    Validate cluster classification:
    1. Within each cluster: consecutive items (by rank) must have overlapping CIs.
    2. Between clusters: last item of cluster i and first of cluster i+1 must NOT overlap.
    """
    errors: list[str] = []
    item_to_idx = {name: i for i, name in enumerate(items)}
    order = sorted(range(len(items)), key=lambda i: ranks[i])
    rank_order_items = [items[i] for i in order]

    flat_cluster_items = []
    for cluster in clusters:
        flat_cluster_items.extend(cluster)

    all_items = set(items)
    cluster_items_set = set(flat_cluster_items)
    if all_items != cluster_items_set:
        errors.append(f"Cluster items mismatch: {all_items} vs {cluster_items_set}")

    for ci, cluster in enumerate(clusters):
        if len(cluster) == 1:
            continue
        for j in range(len(cluster) - 1):
            idx_a = item_to_idx.get(cluster[j])
            idx_b = item_to_idx.get(cluster[j + 1])
            if idx_a is None or idx_b is None:
                errors.append(f"Unknown item in cluster {ci + 1}: {cluster}")
                continue
            if not _ci_overlaps(
                ci_lower[idx_a], ci_upper[idx_a],
                ci_lower[idx_b], ci_upper[idx_b],
            ):
                errors.append(
                    f"Cluster {ci + 1}: {cluster[j]} CI=[{ci_lower[idx_a]:.0f},{ci_upper[idx_a]:.0f}] "
                    f"and {cluster[j+1]} CI=[{ci_lower[idx_b]:.0f},{ci_upper[idx_b]:.0f}] do NOT overlap"
                )

    for ci in range(len(clusters) - 1):
        last_of_curr = clusters[ci][-1]
        first_of_next = clusters[ci + 1][0]
        idx_a = item_to_idx.get(last_of_curr)
        idx_b = item_to_idx.get(first_of_next)
        if idx_a is None or idx_b is None:
            continue
        if _ci_overlaps(
            ci_lower[idx_a], ci_upper[idx_a],
            ci_lower[idx_b], ci_upper[idx_b],
        ):
            errors.append(
                f"Clusters {ci + 1} and {ci + 2} boundary: {last_of_curr} and {first_of_next} "
                f"have OVERLAPPING CIs but are in different clusters"
            )

    return len(errors) == 0, errors


def run_experiment(example_id: str) -> dict | None:
    """Run full pipeline on example and return results + key_findings."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    upload = client.post(f"/api/upload/example/{example_id}")
    if upload.status_code != 200:
        print(f"  ERROR: upload failed {upload.status_code}")
        return None

    session_id = upload.json()["session_id"]
    infer = client.post(f"/api/sessions/{session_id}/infer", json={"user_hints": None})
    if infer.status_code != 200:
        print(f"  ERROR: infer failed {infer.status_code}")
        return None

    infer_body = infer.json()
    if not infer_body.get("success") or not infer_body.get("requires_confirmation"):
        print(f"  ERROR: infer not ready")
        return None

    confirmed_schema = infer_body["schema_result"]["schema"]
    confirm = client.post(
        f"/api/sessions/{session_id}/confirm",
        json={
            "confirmed": True,
            "confirmed_schema": confirmed_schema,
            "user_modifications": [],
            "B": 2000,
            "seed": 42,
        },
    )
    if confirm.status_code != 200:
        print(f"  ERROR: confirm failed {confirm.status_code}")
        return None

    run = client.post(
        f"/api/sessions/{session_id}/run",
        json={"selected_items": None, "selected_indicator_values": None},
    )
    if run.status_code != 200:
        print(f"  ERROR: run failed {run.status_code}: {run.json().get('detail', '')}")
        return None

    run_body = run.json()
    if not run_body.get("success"):
        print(f"  ERROR: run not success")
        return None

    return run_body


def main() -> int:
    print("=" * 70)
    print("Cluster Classification Validation (Example Data)")
    print("=" * 70)

    examples = [
        ("pairwise", "LLM Pairwise Comparison"),
        ("multiway_scores", "Model Performance Matrix"),
    ]

    all_ok = True
    for example_id, title in examples:
        print(f"\n--- {title} ({example_id}) ---")
        run_body = run_experiment(example_id)
        if run_body is None:
            all_ok = False
            continue

        exec_res = run_body.get("execution", {})
        results = exec_res.get("results")
        report = run_body.get("report", {})
        key_findings = report.get("key_findings", {})

        if not results:
            print("  ERROR: No results")
            all_ok = False
            continue

        items = results["items"]
        ranks = results["ranks"]
        ci_lower = results["ci_lower"]
        ci_upper = results["ci_upper"]
        theta_hat = results["theta_hat"]

        order = sorted(range(len(items)), key=lambda i: ranks[i])
        print("\n  Ranking (rank order):")
        for i in order:
            print(f"    {ranks[i]:2d}  {items[i]:20s}  CI=[{ci_lower[i]:.0f}, {ci_upper[i]:.0f}]  theta_hat={theta_hat[i]:.4f}")

        cluster_items = key_findings.get("cluster_items", [])
        n_clusters = key_findings.get("n_clusters", 0)
        print(f"\n  n_clusters: {n_clusters}")
        for gi, group in enumerate(cluster_items):
            print(f"  Group {gi + 1}: {', '.join(group)}")

        ok, errors = _validate_clusters(items, ranks, ci_lower, ci_upper, cluster_items)
        if ok:
            print("\n  [OK] Cluster classification is consistent with CI overlap.")
        else:
            print("\n  [FAIL] Validation errors:")
            for e in errors:
                print(f"    - {e}")
            all_ok = False

    print("\n" + "=" * 70)
    print("DONE" if all_ok else "SOME CHECKS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
