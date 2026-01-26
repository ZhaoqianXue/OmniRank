#!/usr/bin/env python3
"""
Experiment 4.2.3: Two-Step Method Triggering Accuracy

Evaluates the Engine Orchestrator's ability to make correct statistical decisions
about when to apply the two-step refinement method.

The experiment tests three decision conditions:
1. Sparsity Gatekeeper: sparsity_ratio >= 1.0 (coupon collector bound)
2. Heterogeneity Trigger: heterogeneity_index > 0.5
3. Uncertainty Trigger: CI_width / n > 0.2 (20%)

Paper Section: 4.2.3 Two-Step Method Triggering Accuracy
"""

import os
import sys
import json
import time
import logging
import tempfile
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY HERE
# =============================================================================
# Number of experiment runs (for mean +/- std calculation)
# - Set to 1 for quick testing
# - Set to 3 or more for publication-ready results
N_RUNS = 1
# =============================================================================

# Add project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "api"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Paths
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "experiment_config.json")

# Load config
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# R script paths
STEP1_SCRIPT = os.path.join(PROJECT_ROOT, "src", "spectral_ranking", "spectral_ranking_step1.R")

# Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Utilities
# =============================================================================

def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def print_progress(current: int, total: int, message: str):
    """Print progress bar."""
    pct = current / total * 100
    bar_len = 35
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}% | {message:<45}", end="", flush=True)
    if current == total:
        print()


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return float(mean), float(std)


# =============================================================================
# R Script Execution
# =============================================================================

def run_step1_r_script(csv_path: str, output_dir: str, seed: int = 42) -> Dict:
    """
    Run spectral_ranking_step1.R and parse the output.
    
    Returns metadata dict with sparsity_ratio, heterogeneity_index, etc.
    """
    import subprocess
    
    # Prepare command
    cmd = [
        "Rscript",
        STEP1_SCRIPT,
        "--csv", csv_path,
        "--bigbetter", "1",
        "--B", "500",  # Fewer bootstrap iterations for speed
        "--seed", str(seed),
        "--out", output_dir,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr,
                "exit_code": result.returncode,
            }
        
        # Parse output JSON
        json_path = os.path.join(output_dir, "ranking_results.json")
        if not os.path.exists(json_path):
            return {
                "success": False,
                "error": "Output JSON not found",
            }
        
        with open(json_path) as f:
            data = json.load(f)
        
        return {
            "success": True,
            "metadata": data.get("metadata", {}),
            "methods": data.get("methods", []),
            "json_path": json_path,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "R script timed out",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Decision Logic (mirrors r_executor.py)
# =============================================================================

def should_run_step2(metadata: Dict) -> Tuple[bool, str, Dict]:
    """
    Determine whether Step 2 refinement should be triggered.
    
    Decision Logic:
    1. GATEKEEPER: sparsity_ratio >= 1.0 (data sufficiency)
    2. TRIGGERS: heterogeneity > 0.5 OR CI_width/n > 0.2 (20%)
    
    Returns:
        Tuple of (decision: bool, reason: str, details: dict)
    """
    sparsity_ratio = metadata.get("sparsity_ratio", 0.0)
    heterogeneity = metadata.get("heterogeneity_index", 0.0)
    ci_width_top5 = metadata.get("mean_ci_width_top_5", 0.0)
    n_items = metadata.get("k_methods", 1)
    
    details = {
        "sparsity_ratio": sparsity_ratio,
        "heterogeneity_index": heterogeneity,
        "ci_width_top5": ci_width_top5,
        "n_items": n_items,
        "ci_width_ratio": ci_width_top5 / n_items if n_items > 0 else 0.0,
    }
    
    # 1. GATEKEEPER: Check data sufficiency
    if sparsity_ratio < 1.0:
        reason = f"Gatekeeper blocked (sparsity={sparsity_ratio:.2f} < 1.0)"
        details["gatekeeper_passed"] = False
        details["triggers"] = []
        return False, reason, details
    
    details["gatekeeper_passed"] = True
    
    # 2. CHECK TRIGGERS
    triggers_activated = []
    
    # Trigger A: Heterogeneity
    if heterogeneity > 0.5:
        triggers_activated.append("heterogeneity")
    
    # Trigger B: Uncertainty
    ci_ratio = ci_width_top5 / n_items if n_items > 0 else 0.0
    if ci_ratio > 0.2:
        triggers_activated.append("uncertainty")
    
    details["triggers"] = triggers_activated
    
    # 3. FINAL DECISION
    if triggers_activated:
        reason = f"Step 2 triggered ({', '.join(triggers_activated)})"
        return True, reason, details
    
    reason = f"No triggers (het={heterogeneity:.3f}, CI_ratio={ci_ratio:.1%})"
    return False, reason, details


# =============================================================================
# Metrics Calculation
# =============================================================================

def compute_decision_metrics(results: List[Dict]) -> Dict:
    """Compute metrics for decision accuracy."""
    
    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix for Step 2 triggering
    tp = sum(1 for r in results if r["expected"] and r["predicted"])
    fp = sum(1 for r in results if not r["expected"] and r["predicted"])
    tn = sum(1 for r in results if not r["expected"] and not r["predicted"])
    fn = sum(1 for r in results if r["expected"] and not r["predicted"])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Gatekeeper accuracy (how well it identifies sparse data)
    gatekeeper_cases = [r for r in results if r.get("expected_reason") == "gatekeeper_blocked"]
    gatekeeper_correct = sum(1 for r in gatekeeper_cases if not r["predicted"])
    gatekeeper_accuracy = gatekeeper_correct / len(gatekeeper_cases) if gatekeeper_cases else 1.0
    
    return {
        "overall_accuracy": accuracy,
        "n_correct": correct,
        "n_total": total,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gatekeeper_accuracy": gatekeeper_accuracy,
    }


def compute_scenario_metrics(results: List[Dict]) -> Dict:
    """Compute metrics per scenario."""
    
    scenario_results = defaultdict(list)
    for r in results:
        scenario_results[r["scenario"]].append(r)
    
    scenario_metrics = {}
    for scenario, scenario_list in scenario_results.items():
        correct = sum(1 for r in scenario_list if r["correct"])
        total = len(scenario_list)
        scenario_metrics[scenario] = {
            "accuracy": correct / total if total > 0 else 0,
            "n_correct": correct,
            "n_total": total,
        }
    
    return scenario_metrics


# =============================================================================
# Single Run Evaluation
# =============================================================================

def run_single_evaluation(
    datasets: List[Dict],
    run_id: int,
) -> Dict:
    """Run evaluation on all datasets."""
    
    results = []
    errors = []
    total = len(datasets)
    
    for idx, ds_meta in enumerate(datasets):
        dataset_id = ds_meta["dataset_id"]
        filename = ds_meta["filename"]
        scenario = ds_meta["scenario"]
        expected_step2 = ds_meta["expected_step2"]
        expected_reason = ds_meta["expected_reason"]
        
        print_progress(idx + 1, total, f"[Run {run_id}] {dataset_id}")
        
        # Load and run Step 1
        csv_path = os.path.join(DATASETS_DIR, filename)
        
        # Create temp output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            step1_result = run_step1_r_script(csv_path, temp_dir)
            
            if not step1_result.get("success"):
                errors.append({
                    "dataset_id": dataset_id,
                    "error": step1_result.get("error", "Unknown error"),
                })
                # Use default values for failed runs
                predicted_step2 = False
                reason = "R script failed"
                details = {}
            else:
                # Get decision
                metadata = step1_result["metadata"]
                predicted_step2, reason, details = should_run_step2(metadata)
        
        # Record result
        result = {
            "dataset_id": dataset_id,
            "scenario": scenario,
            "expected": expected_step2,
            "expected_reason": expected_reason,
            "predicted": predicted_step2,
            "predicted_reason": reason,
            "correct": expected_step2 == predicted_step2,
            "details": details,
        }
        results.append(result)
    
    return {
        "run_id": run_id,
        "results": results,
        "errors": errors,
    }


# =============================================================================
# Output Generation
# =============================================================================

def generate_latex_table(metrics: Dict, scenario_metrics: Dict, n_runs: int) -> str:
    """Generate LaTeX table for JASA submission."""
    lines = []
    lines.append("% Table: Two-Step Method Triggering Accuracy (Section 4.2.3)")
    lines.append(f"% Results over {n_runs} runs")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Two-Step Method Triggering Decision Accuracy}")
    lines.append("\\label{tab:step2_triggering}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("Metric & Value \\\\")
    lines.append("\\midrule")
    
    acc_mean, acc_std = metrics["overall_accuracy"]
    prec_mean, prec_std = metrics["precision"]
    rec_mean, rec_std = metrics["recall"]
    f1_mean, f1_std = metrics["f1"]
    gate_mean, gate_std = metrics["gatekeeper_accuracy"]
    
    lines.append(f"Overall Accuracy & {acc_mean:.3f}$\\pm${acc_std:.3f} \\\\")
    lines.append(f"Precision (Step 2 Triggered) & {prec_mean:.3f}$\\pm${prec_std:.3f} \\\\")
    lines.append(f"Recall (Step 2 Triggered) & {rec_mean:.3f}$\\pm${rec_std:.3f} \\\\")
    lines.append(f"F1 Score & {f1_mean:.3f}$\\pm${f1_std:.3f} \\\\")
    lines.append(f"Gatekeeper Accuracy & {gate_mean:.3f}$\\pm${gate_std:.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Scenario breakdown
    lines.append("")
    lines.append("% Table: Accuracy by Scenario")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Triggering Accuracy by Test Scenario}")
    lines.append("\\label{tab:step2_by_scenario}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Scenario & Expected & Accuracy \\\\")
    lines.append("\\midrule")
    
    # Map scenario to expected Step 2 decision
    expected_decisions = {
        "sparse_homogeneous": "No",
        "sparse_heterogeneous": "No",
        "sufficient_no_trigger": "No",
        "sufficient_heterogeneity": "Yes",
        "sufficient_uncertainty": "Yes",
        "sufficient_both": "Yes",
        "boundary_sparsity_below": "No",
        "boundary_sparsity_above": "Yes",
        "large_items_sparse": "No",
        "large_items_sufficient": "Yes",
    }
    
    for scenario in sorted(scenario_metrics.keys()):
        acc_mean, acc_std = scenario_metrics[scenario]
        expected = expected_decisions.get(scenario, "?")
        scenario_escaped = scenario.replace("_", "\\_")
        lines.append(f"{scenario_escaped} & {expected} & {acc_mean:.3f}$\\pm${acc_std:.3f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_markdown_table(metrics: Dict, scenario_metrics: Dict, n_runs: int) -> str:
    """Generate Markdown table."""
    lines = []
    lines.append(f"## Two-Step Method Triggering Accuracy")
    lines.append(f"*Results over {n_runs} runs (mean +/- std)*\n")
    
    lines.append("### Overall Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    
    acc_mean, acc_std = metrics["overall_accuracy"]
    prec_mean, prec_std = metrics["precision"]
    rec_mean, rec_std = metrics["recall"]
    f1_mean, f1_std = metrics["f1"]
    gate_mean, gate_std = metrics["gatekeeper_accuracy"]
    
    lines.append(f"| Overall Accuracy | {acc_mean:.3f}+/-{acc_std:.3f} |")
    lines.append(f"| Precision (Step 2 Triggered) | {prec_mean:.3f}+/-{prec_std:.3f} |")
    lines.append(f"| Recall (Step 2 Triggered) | {rec_mean:.3f}+/-{rec_std:.3f} |")
    lines.append(f"| F1 Score | {f1_mean:.3f}+/-{f1_std:.3f} |")
    lines.append(f"| Gatekeeper Accuracy | {gate_mean:.3f}+/-{gate_std:.3f} |")
    
    lines.append("\n### Accuracy by Scenario\n")
    lines.append("| Scenario | Expected Step 2 | Accuracy |")
    lines.append("|----------|-----------------|----------|")
    
    # Map scenario to expected Step 2 decision
    expected_decisions = {
        "sparse_homogeneous": "No",
        "sparse_heterogeneous": "No",
        "sufficient_no_trigger": "No",
        "sufficient_heterogeneity": "Yes",
        "sufficient_uncertainty": "Yes",
        "sufficient_both": "Yes",
        "boundary_sparsity_below": "No",
        "boundary_sparsity_above": "Yes",
        "large_items_sparse": "No",
        "large_items_sufficient": "Yes",
    }
    
    for scenario in sorted(scenario_metrics.keys()):
        acc_mean, acc_std = scenario_metrics[scenario]
        expected = expected_decisions.get(scenario, "?")
        lines.append(f"| {scenario} | {expected} | {acc_mean:.3f}+/-{acc_std:.3f} |")
    
    return "\n".join(lines)


def generate_confusion_matrix_table(all_results: List[Dict]) -> str:
    """Generate confusion matrix as markdown."""
    # Aggregate confusion matrix
    tp = sum(1 for r in all_results if r["expected"] and r["predicted"])
    fp = sum(1 for r in all_results if not r["expected"] and r["predicted"])
    tn = sum(1 for r in all_results if not r["expected"] and not r["predicted"])
    fn = sum(1 for r in all_results if r["expected"] and not r["predicted"])
    
    lines = []
    lines.append("### Confusion Matrix\n")
    lines.append("```")
    lines.append("                 Predicted")
    lines.append("              No Step2   Step2")
    lines.append(f"Actual No    [{tn:4d}      {fp:4d}]")
    lines.append(f"       Yes   [{fn:4d}      {tp:4d}]")
    lines.append("```")
    
    return "\n".join(lines)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the complete experiment."""
    print_section("Experiment 4.2.3: Two-Step Method Triggering Accuracy")
    
    # Create run directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Load manifest
    manifest_path = os.path.join(DATASETS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: Datasets not found. Run generate_datasets.py first.")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    datasets = manifest["datasets"]
    n_datasets = len(datasets)
    n_runs = N_RUNS
    
    print(f"  Run ID: {run_timestamp}")
    print(f"  Datasets: {n_datasets}")
    print(f"  Runs: {n_runs}")
    print(f"  Output: {run_dir}")
    
    # Check R script exists
    if not os.path.exists(STEP1_SCRIPT):
        print(f"ERROR: R script not found: {STEP1_SCRIPT}")
        return
    
    print(f"  R script: {STEP1_SCRIPT}")
    
    # Run evaluations
    print_section("Running Evaluations")
    
    all_run_results = []
    all_run_metrics = {
        "overall_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "gatekeeper_accuracy": [],
    }
    all_scenario_metrics = defaultdict(list)
    
    for run_id in range(1, n_runs + 1):
        print(f"\n--- Run {run_id}/{n_runs} ---")
        
        run_result = run_single_evaluation(datasets, run_id)
        all_run_results.append(run_result)
        
        # Compute metrics for this run
        metrics = compute_decision_metrics(run_result["results"])
        scenario_metrics = compute_scenario_metrics(run_result["results"])
        
        all_run_metrics["overall_accuracy"].append(metrics["overall_accuracy"])
        all_run_metrics["precision"].append(metrics["precision"])
        all_run_metrics["recall"].append(metrics["recall"])
        all_run_metrics["f1"].append(metrics["f1"])
        all_run_metrics["gatekeeper_accuracy"].append(metrics["gatekeeper_accuracy"])
        
        for scenario, sm in scenario_metrics.items():
            all_scenario_metrics[scenario].append(sm["accuracy"])
    
    # Aggregate metrics
    print_section("Aggregating Results")
    
    aggregated_metrics = {}
    for metric, values in all_run_metrics.items():
        aggregated_metrics[metric] = compute_mean_std(values)
    
    aggregated_scenario_metrics = {}
    for scenario, values in all_scenario_metrics.items():
        aggregated_scenario_metrics[scenario] = compute_mean_std(values)
    
    # Print summary
    print_section("Results Summary")
    
    acc_mean, acc_std = aggregated_metrics["overall_accuracy"]
    prec_mean, prec_std = aggregated_metrics["precision"]
    rec_mean, rec_std = aggregated_metrics["recall"]
    f1_mean, f1_std = aggregated_metrics["f1"]
    gate_mean, gate_std = aggregated_metrics["gatekeeper_accuracy"]
    
    print(f"\n  Overall Accuracy:    {acc_mean:.3f} +/- {acc_std:.3f}")
    print(f"  Precision:           {prec_mean:.3f} +/- {prec_std:.3f}")
    print(f"  Recall:              {rec_mean:.3f} +/- {rec_std:.3f}")
    print(f"  F1 Score:            {f1_mean:.3f} +/- {f1_std:.3f}")
    print(f"  Gatekeeper Accuracy: {gate_mean:.3f} +/- {gate_std:.3f}")
    
    print(f"\n  By Scenario:")
    for scenario in sorted(aggregated_scenario_metrics.keys()):
        acc_mean, acc_std = aggregated_scenario_metrics[scenario]
        print(f"    {scenario:30s}: {acc_mean:.3f} +/- {acc_std:.3f}")
    
    # Save results
    print_section("Saving Outputs")
    
    # Flatten all results
    all_results = []
    for run in all_run_results:
        all_results.extend(run["results"])
    
    # Raw results JSON
    raw_results = {
        "experiment_id": CONFIG["experiment_id"],
        "paper_section": CONFIG["paper_section"],
        "run_timestamp": run_timestamp,
        "n_datasets": n_datasets,
        "n_runs": n_runs,
        "decision_rules": CONFIG["decision_rules"],
        "metrics": {
            "overall_accuracy": {"mean": acc_mean, "std": acc_std},
            "precision": {"mean": prec_mean, "std": prec_std},
            "recall": {"mean": rec_mean, "std": rec_std},
            "f1": {"mean": f1_mean, "std": f1_std},
            "gatekeeper_accuracy": {"mean": gate_mean, "std": gate_std},
        },
        "scenario_metrics": {
            s: {"mean": m, "std": st}
            for s, (m, st) in aggregated_scenario_metrics.items()
        },
        "detailed_results": all_results,
    }
    
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"  Results JSON: {json_path}")
    
    # LaTeX table
    latex_content = generate_latex_table(
        aggregated_metrics, aggregated_scenario_metrics, n_runs
    )
    latex_path = os.path.join(run_dir, "table_4_2_3.tex")
    with open(latex_path, "w") as f:
        f.write(latex_content)
    print(f"  LaTeX table: {latex_path}")
    
    # Markdown table
    md_content = generate_markdown_table(
        aggregated_metrics, aggregated_scenario_metrics, n_runs
    )
    md_content += "\n\n" + generate_confusion_matrix_table(all_results)
    md_path = os.path.join(run_dir, "table_4_2_3.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  Markdown table: {md_path}")
    
    # Error analysis
    errors = []
    for run in all_run_results:
        errors.extend(run.get("errors", []))
    
    misclassified = [r for r in all_results if not r["correct"]]
    
    error_analysis = {
        "execution_errors": errors,
        "misclassified_count": len(misclassified),
        "misclassified_datasets": [
            {
                "dataset_id": r["dataset_id"],
                "scenario": r["scenario"],
                "expected": r["expected"],
                "predicted": r["predicted"],
                "expected_reason": r["expected_reason"],
                "predicted_reason": r["predicted_reason"],
                "details": r.get("details", {}),
            }
            for r in misclassified
        ],
    }
    
    error_path = os.path.join(run_dir, "error_analysis.json")
    with open(error_path, "w") as f:
        json.dump(error_analysis, f, indent=2)
    print(f"  Error Analysis: {error_path}")
    
    # Print tables
    print_section("LaTeX Output (Copy to Paper)")
    print(latex_content)
    
    print_section("Markdown Output")
    print(md_content)
    
    print_section("Experiment Complete")
    print(f"  All outputs saved to: {run_dir}")


if __name__ == "__main__":
    run_experiment()
