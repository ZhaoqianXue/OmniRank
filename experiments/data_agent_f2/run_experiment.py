#!/usr/bin/env python3
"""
Experiment 4.2.2: Semantic Schema Inference

Comprehensive evaluation of Data Agent's semantic understanding capabilities:
- BigBetter Direction: Inference accuracy (higher vs lower is better)
- Ranking Items Identification: Precision/Recall/Jaccard for item column detection
- Indicator Column Detection: Exact match accuracy
- Indicator Values Extraction: Set-based F1 score

Features:
- Multiple runs with different seeds for mean +/- std computation
- Publication-ready outputs (LaTeX/Markdown tables)
- Detailed error analysis and breakdown by category

Paper Section: 4.2.2 Semantic Schema Inference
"""

import os
import sys
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple, Set

import numpy as np
import pandas as pd

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY HERE
# =============================================================================
# Number of experiment runs (for mean +/- std calculation)
# - Set to 1 for quick testing
# - Set to 5 or more for publication-ready results
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
    bar = "=" * filled + ">" + " " * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
    print(f"\r  [{bar}] {pct:5.1f}% | {message:<45}", end="", flush=True)
    if current == total:
        print()


def format_to_string(fmt) -> str:
    """Convert DataFormat enum to string."""
    if hasattr(fmt, 'value'):
        return fmt.value.lower()
    return str(fmt).lower()


# =============================================================================
# Metrics Calculation
# =============================================================================

def compute_accuracy(y_true: List, y_pred: List) -> float:
    """Compute accuracy."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def compute_jaccard(set_true: Set, set_pred: Set) -> float:
    """Compute Jaccard index for set comparison."""
    if not set_true and not set_pred:
        return 1.0  # Both empty = perfect match
    if not set_true or not set_pred:
        return 0.0
    intersection = len(set_true & set_pred)
    union = len(set_true | set_pred)
    return intersection / union if union > 0 else 0.0


def compute_set_f1(set_true: Set, set_pred: Set) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for set comparison."""
    if not set_true and not set_pred:
        return 1.0, 1.0, 1.0  # Both empty = perfect
    if not set_pred:
        return 0.0, 0.0, 0.0
    if not set_true:
        return 0.0, 0.0, 0.0
    
    intersection = len(set_true & set_pred)
    precision = intersection / len(set_pred) if set_pred else 0.0
    recall = intersection / len(set_true) if set_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std


# =============================================================================
# Data Agent Wrapper
# =============================================================================

class DataAgentWrapper:
    """Wrapper for Data Agent that handles schema inference."""
    
    def __init__(self):
        try:
            from agents.data_agent import DataAgent
            self.agent = DataAgent()
            self.enabled = self.agent.enabled
        except Exception as e:
            logger.warning(f"Failed to initialize DataAgent: {e}")
            self.agent = None
            self.enabled = False
    
    def infer_schema(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Run schema inference through Data Agent."""
        if not self.agent:
            return {
                "bigbetter": 1,
                "ranking_items": [],
                "indicator_col": None,
                "indicator_values": [],
                "error": "Agent not initialized",
            }
        
        try:
            schema, warnings, explanation = self.agent.process(content, filename)
            
            return {
                "bigbetter": schema.bigbetter,
                "ranking_items": schema.ranking_items or [],
                "indicator_col": schema.indicator_col,
                "indicator_values": schema.indicator_values or [],
                "format": format_to_string(schema.format),
            }
        except Exception as e:
            return {
                "bigbetter": 1,
                "ranking_items": [],
                "indicator_col": None,
                "indicator_values": [],
                "error": str(e),
            }


# =============================================================================
# Single Run Evaluation
# =============================================================================

def run_single_evaluation(
    datasets: List[Dict],
    agent: DataAgentWrapper,
    run_id: int,
) -> Dict:
    """Run evaluation for a single run on all datasets."""
    
    results = {
        "run_id": run_id,
        "bigbetter": {"y_true": [], "y_pred": [], "correct": []},
        "ranking_items": {"jaccards": [], "precisions": [], "recalls": [], "f1s": []},
        "indicator_col": {"y_true": [], "y_pred": [], "correct": []},
        "indicator_values": {"precisions": [], "recalls": [], "f1s": []},
        "detailed_results": [],
        "category_results": defaultdict(list),
        "errors": [],
    }
    
    total = len(datasets)
    
    for idx, ds_meta in enumerate(datasets):
        dataset_id = ds_meta["dataset_id"]
        filename = ds_meta["filename"]
        category = ds_meta["category"]
        
        # Ground truth
        gt_bigbetter = ds_meta.get("expected_bigbetter", 1)
        gt_items = set(ds_meta.get("expected_ranking_items", []))
        gt_indicator = ds_meta.get("expected_indicator_col")
        gt_values = set(ds_meta.get("expected_indicator_values", []))
        
        print_progress(idx + 1, total, f"[Run {run_id}] {dataset_id}")
        
        # Load data and run inference
        csv_path = os.path.join(DATASETS_DIR, filename)
        
        try:
            with open(csv_path, "rb") as f:
                content = f.read()
            
            prediction = agent.infer_schema(content, os.path.basename(filename))
            
            pred_bigbetter = prediction.get("bigbetter", 1)
            pred_items = set(prediction.get("ranking_items", []))
            pred_indicator = prediction.get("indicator_col")
            pred_values = set(prediction.get("indicator_values", []))
            
        except Exception as e:
            pred_bigbetter = 1
            pred_items = set()
            pred_indicator = None
            pred_values = set()
            results["errors"].append({"dataset_id": dataset_id, "error": str(e)})
        
        # === BigBetter Evaluation ===
        bb_correct = gt_bigbetter == pred_bigbetter
        results["bigbetter"]["y_true"].append(gt_bigbetter)
        results["bigbetter"]["y_pred"].append(pred_bigbetter)
        results["bigbetter"]["correct"].append(bb_correct)
        
        # === Ranking Items Evaluation ===
        items_jaccard = compute_jaccard(gt_items, pred_items)
        items_p, items_r, items_f1 = compute_set_f1(gt_items, pred_items)
        results["ranking_items"]["jaccards"].append(items_jaccard)
        results["ranking_items"]["precisions"].append(items_p)
        results["ranking_items"]["recalls"].append(items_r)
        results["ranking_items"]["f1s"].append(items_f1)
        
        # === Indicator Column Evaluation ===
        ind_correct = (gt_indicator == pred_indicator) or (gt_indicator is None and pred_indicator is None)
        results["indicator_col"]["y_true"].append(gt_indicator)
        results["indicator_col"]["y_pred"].append(pred_indicator)
        results["indicator_col"]["correct"].append(ind_correct)
        
        # === Indicator Values Evaluation ===
        val_p, val_r, val_f1 = compute_set_f1(gt_values, pred_values)
        results["indicator_values"]["precisions"].append(val_p)
        results["indicator_values"]["recalls"].append(val_r)
        results["indicator_values"]["f1s"].append(val_f1)
        
        # Detailed result
        detail = {
            "dataset_id": dataset_id,
            "category": category,
            "difficulty": ds_meta.get("difficulty", "unknown"),
            "ground_truth": {
                "bigbetter": gt_bigbetter,
                "ranking_items": list(gt_items),
                "indicator_col": gt_indicator,
                "indicator_values": list(gt_values),
            },
            "prediction": {
                "bigbetter": pred_bigbetter,
                "ranking_items": list(pred_items),
                "indicator_col": pred_indicator,
                "indicator_values": list(pred_values),
            },
            "metrics": {
                "bigbetter_correct": bb_correct,
                "items_jaccard": items_jaccard,
                "items_f1": items_f1,
                "indicator_correct": ind_correct,
                "values_f1": val_f1,
            },
        }
        results["detailed_results"].append(detail)
        results["category_results"][category].append(detail)
    
    return results


# =============================================================================
# Aggregated Metrics
# =============================================================================

def aggregate_run_results(all_runs: List[Dict]) -> Dict:
    """Aggregate results across multiple runs for mean +/- std."""
    
    aggregated = {
        "bigbetter": {"accuracies": []},
        "ranking_items": {"jaccards": [], "f1s": []},
        "indicator_col": {"accuracies": []},
        "indicator_values": {"f1s": []},
        "category_metrics": defaultdict(lambda: {
            "bigbetter_accs": [],
            "items_jaccards": [],
            "indicator_accs": [],
            "values_f1s": [],
        }),
    }
    
    for run in all_runs:
        # BigBetter accuracy
        bb_acc = compute_accuracy(
            run["bigbetter"]["y_true"],
            run["bigbetter"]["y_pred"]
        )
        aggregated["bigbetter"]["accuracies"].append(bb_acc)
        
        # Ranking Items Jaccard and F1
        aggregated["ranking_items"]["jaccards"].append(np.mean(run["ranking_items"]["jaccards"]))
        aggregated["ranking_items"]["f1s"].append(np.mean(run["ranking_items"]["f1s"]))
        
        # Indicator Column accuracy
        ind_acc = sum(run["indicator_col"]["correct"]) / len(run["indicator_col"]["correct"])
        aggregated["indicator_col"]["accuracies"].append(ind_acc)
        
        # Indicator Values F1
        aggregated["indicator_values"]["f1s"].append(np.mean(run["indicator_values"]["f1s"]))
        
        # Per-category metrics
        for category, details in run["category_results"].items():
            if not details:
                continue
            
            cat_bb_acc = sum(1 for d in details if d["metrics"]["bigbetter_correct"]) / len(details)
            cat_items_jaccard = np.mean([d["metrics"]["items_jaccard"] for d in details])
            cat_ind_acc = sum(1 for d in details if d["metrics"]["indicator_correct"]) / len(details)
            cat_val_f1 = np.mean([d["metrics"]["values_f1"] for d in details])
            
            aggregated["category_metrics"][category]["bigbetter_accs"].append(cat_bb_acc)
            aggregated["category_metrics"][category]["items_jaccards"].append(cat_items_jaccard)
            aggregated["category_metrics"][category]["indicator_accs"].append(cat_ind_acc)
            aggregated["category_metrics"][category]["values_f1s"].append(cat_val_f1)
    
    return aggregated


# =============================================================================
# Output Generation (LaTeX & Markdown)
# =============================================================================

def generate_latex_table(aggregated: Dict, n_runs: int) -> str:
    """Generate LaTeX table with mean +/- std for JASA submission."""
    lines = []
    lines.append("% Table: Semantic Schema Inference Performance (Section 4.2.2)")
    lines.append(f"% Results over {n_runs} run(s)")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Semantic Schema Inference Performance (Mean $\\pm$ Std over {n_runs} runs)}}")
    lines.append("\\label{tab:schema_inference}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Category & BigBetter Acc & Items Jaccard & Indicator Acc & Values F1 \\\\")
    lines.append("\\midrule")
    
    # Per-category results
    for category in ["bigbetter_high", "bigbetter_low", "bigbetter_ambiguous", 
                     "indicator_clear", "indicator_none", "items_complex"]:
        metrics = aggregated["category_metrics"].get(category, {})
        
        bb_m, bb_s = compute_mean_std(metrics.get("bigbetter_accs", []))
        items_m, items_s = compute_mean_std(metrics.get("items_jaccards", []))
        ind_m, ind_s = compute_mean_std(metrics.get("indicator_accs", []))
        val_m, val_s = compute_mean_std(metrics.get("values_f1s", []))
        
        cat_display = category.replace("_", " ").title()
        lines.append(f"{cat_display} & {bb_m:.3f}$\\pm${bb_s:.3f} & {items_m:.3f}$\\pm${items_s:.3f} & {ind_m:.3f}$\\pm${ind_s:.3f} & {val_m:.3f}$\\pm${val_s:.3f} \\\\")
    
    lines.append("\\midrule")
    
    # Overall results
    bb_mean, bb_std = compute_mean_std(aggregated["bigbetter"]["accuracies"])
    items_mean, items_std = compute_mean_std(aggregated["ranking_items"]["jaccards"])
    ind_mean, ind_std = compute_mean_std(aggregated["indicator_col"]["accuracies"])
    val_mean, val_std = compute_mean_std(aggregated["indicator_values"]["f1s"])
    
    lines.append(f"\\textbf{{Overall}} & \\textbf{{{bb_mean:.3f}$\\pm${bb_std:.3f}}} & \\textbf{{{items_mean:.3f}$\\pm${items_std:.3f}}} & \\textbf{{{ind_mean:.3f}$\\pm${ind_std:.3f}}} & \\textbf{{{val_mean:.3f}$\\pm${val_std:.3f}}} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_markdown_table(aggregated: Dict, n_runs: int) -> str:
    """Generate Markdown table for writing.md Section 4.2.2."""
    lines = []
    lines.append(f"**Results Table** (Mean over {n_runs} run(s)):\n")
    lines.append("| Category | BigBetter Acc | Items Jaccard | Indicator Acc | Values F1 |")
    lines.append("|----------|---------------|---------------|---------------|-----------|")
    
    # Per-category results
    for category in ["bigbetter_high", "bigbetter_low", "bigbetter_ambiguous", 
                     "indicator_clear", "indicator_none", "items_complex"]:
        metrics = aggregated["category_metrics"].get(category, {})
        
        bb_m, bb_s = compute_mean_std(metrics.get("bigbetter_accs", []))
        items_m, items_s = compute_mean_std(metrics.get("items_jaccards", []))
        ind_m, ind_s = compute_mean_std(metrics.get("indicator_accs", []))
        val_m, val_s = compute_mean_std(metrics.get("values_f1s", []))
        
        cat_display = category.replace("_", " ").title()
        
        # Format with +/- if multiple runs
        if n_runs > 1:
            lines.append(f"| {cat_display} | {bb_m:.3f}+/-{bb_s:.3f} | {items_m:.3f}+/-{items_s:.3f} | {ind_m:.3f}+/-{ind_s:.3f} | {val_m:.3f}+/-{val_s:.3f} |")
        else:
            lines.append(f"| {cat_display} | {bb_m:.3f} | {items_m:.3f} | {ind_m:.3f} | {val_m:.3f} |")
    
    # Overall results
    bb_mean, bb_std = compute_mean_std(aggregated["bigbetter"]["accuracies"])
    items_mean, items_std = compute_mean_std(aggregated["ranking_items"]["jaccards"])
    ind_mean, ind_std = compute_mean_std(aggregated["indicator_col"]["accuracies"])
    val_mean, val_std = compute_mean_std(aggregated["indicator_values"]["f1s"])
    
    if n_runs > 1:
        lines.append(f"| **Overall** | **{bb_mean:.3f}+/-{bb_std:.3f}** | **{items_mean:.3f}+/-{items_std:.3f}** | **{ind_mean:.3f}+/-{ind_std:.3f}** | **{val_mean:.3f}+/-{val_std:.3f}** |")
    else:
        lines.append(f"| **Overall** | **{bb_mean:.3f}** | **{items_mean:.3f}** | **{ind_mean:.3f}** | **{val_mean:.3f}** |")
    
    return "\n".join(lines)


def generate_error_analysis(all_runs: List[Dict]) -> Dict:
    """Generate detailed error analysis."""
    
    error_analysis = {
        "bigbetter_errors": [],
        "indicator_errors": [],
        "items_low_jaccard": [],
        "by_category": defaultdict(lambda: {"total": 0, "errors": 0}),
        "by_difficulty": defaultdict(lambda: {"total": 0, "errors": 0}),
    }
    
    # Use first run for detailed analysis
    run = all_runs[0]
    
    for detail in run["detailed_results"]:
        category = detail["category"]
        difficulty = detail["difficulty"]
        
        error_analysis["by_category"][category]["total"] += 1
        error_analysis["by_difficulty"][difficulty]["total"] += 1
        
        # Track BigBetter errors
        if not detail["metrics"]["bigbetter_correct"]:
            error_analysis["bigbetter_errors"].append({
                "dataset_id": detail["dataset_id"],
                "expected": detail["ground_truth"]["bigbetter"],
                "predicted": detail["prediction"]["bigbetter"],
                "category": category,
            })
            error_analysis["by_category"][category]["errors"] += 1
            error_analysis["by_difficulty"][difficulty]["errors"] += 1
        
        # Track Indicator errors
        if not detail["metrics"]["indicator_correct"]:
            error_analysis["indicator_errors"].append({
                "dataset_id": detail["dataset_id"],
                "expected": detail["ground_truth"]["indicator_col"],
                "predicted": detail["prediction"]["indicator_col"],
                "category": category,
            })
        
        # Track low Jaccard (< 0.8)
        if detail["metrics"]["items_jaccard"] < 0.8:
            error_analysis["items_low_jaccard"].append({
                "dataset_id": detail["dataset_id"],
                "jaccard": detail["metrics"]["items_jaccard"],
                "expected_count": len(detail["ground_truth"]["ranking_items"]),
                "predicted_count": len(detail["prediction"]["ranking_items"]),
                "category": category,
            })
    
    return error_analysis


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the complete semantic schema inference experiment."""
    print_section("Experiment 4.2.2: Semantic Schema Inference")
    
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
    print(f"  Runs: {n_runs} (modify N_RUNS at top of file to change)")
    print(f"  Output: {run_dir}")
    
    # Initialize agent
    agent = DataAgentWrapper()
    print(f"\n  Data Agent LLM Enabled: {agent.enabled}")
    
    # Run multiple evaluations
    print_section("Running Evaluations")
    
    all_runs = []
    
    for run_id in range(1, n_runs + 1):
        print(f"\n--- Run {run_id}/{n_runs} ---")
        result = run_single_evaluation(datasets, agent, run_id)
        all_runs.append(result)
    
    # Aggregate results
    print_section("Aggregating Results")
    aggregated = aggregate_run_results(all_runs)
    
    # Print summary
    print_section("Results Summary")
    
    bb_mean, bb_std = compute_mean_std(aggregated["bigbetter"]["accuracies"])
    items_mean, items_std = compute_mean_std(aggregated["ranking_items"]["jaccards"])
    ind_mean, ind_std = compute_mean_std(aggregated["indicator_col"]["accuracies"])
    val_mean, val_std = compute_mean_std(aggregated["indicator_values"]["f1s"])
    
    print(f"\n  Overall Metrics:")
    print(f"    BigBetter Accuracy:      {bb_mean:.3f} +/- {bb_std:.3f}")
    print(f"    Items Jaccard:           {items_mean:.3f} +/- {items_std:.3f}")
    print(f"    Indicator Col Accuracy:  {ind_mean:.3f} +/- {ind_std:.3f}")
    print(f"    Indicator Values F1:     {val_mean:.3f} +/- {val_std:.3f}")
    
    print(f"\n  By Category (BigBetter Accuracy):")
    for category in ["bigbetter_high", "bigbetter_low", "bigbetter_ambiguous", 
                     "indicator_clear", "indicator_none", "items_complex"]:
        accs = aggregated["category_metrics"].get(category, {}).get("bigbetter_accs", [])
        if accs:
            mean, std = compute_mean_std(accs)
            print(f"    {category:22s}: {mean:.3f} +/- {std:.3f}")
    
    # Save results
    print_section("Saving Outputs")
    
    # Raw results JSON
    raw_results = {
        "experiment_id": CONFIG["experiment_id"],
        "paper_section": CONFIG["paper_section"],
        "run_timestamp": run_timestamp,
        "n_datasets": n_datasets,
        "n_runs": n_runs,
        "agent_enabled": agent.enabled,
        "overall_metrics": {
            "bigbetter_accuracy": {"mean": bb_mean, "std": bb_std},
            "ranking_items_jaccard": {"mean": items_mean, "std": items_std},
            "indicator_col_accuracy": {"mean": ind_mean, "std": ind_std},
            "indicator_values_f1": {"mean": val_mean, "std": val_std},
        },
        "category_metrics": {
            cat: {
                "bigbetter_accuracy": {"mean": m1, "std": s1},
                "items_jaccard": {"mean": m2, "std": s2},
                "indicator_accuracy": {"mean": m3, "std": s3},
                "values_f1": {"mean": m4, "std": s4},
            }
            for cat in aggregated["category_metrics"]
            for m1, s1 in [compute_mean_std(aggregated["category_metrics"][cat]["bigbetter_accs"])]
            for m2, s2 in [compute_mean_std(aggregated["category_metrics"][cat]["items_jaccards"])]
            for m3, s3 in [compute_mean_std(aggregated["category_metrics"][cat]["indicator_accs"])]
            for m4, s4 in [compute_mean_std(aggregated["category_metrics"][cat]["values_f1s"])]
        },
    }
    
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"  Results JSON: {json_path}")
    
    # Error analysis
    error_analysis = generate_error_analysis(all_runs)
    error_path = os.path.join(run_dir, "error_analysis.json")
    with open(error_path, "w") as f:
        json.dump(error_analysis, f, indent=2, default=list)
    print(f"  Error Analysis: {error_path}")
    
    # LaTeX table
    latex_content = generate_latex_table(aggregated, n_runs)
    latex_path = os.path.join(run_dir, "table_4_2_2.tex")
    with open(latex_path, "w") as f:
        f.write(latex_content)
    print(f"  LaTeX table: {latex_path}")
    
    # Markdown table
    md_content = generate_markdown_table(aggregated, n_runs)
    md_path = os.path.join(run_dir, "table_4_2_2.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  Markdown table: {md_path}")
    
    # Print tables
    print_section("LaTeX Output (Copy to Paper)")
    print(latex_content)
    
    print_section("Markdown Output (Copy to writing.md Section 4.2.2)")
    print(md_content)
    
    print_section("Experiment Complete")
    print(f"  All outputs saved to: {run_dir}")


if __name__ == "__main__":
    run_experiment()
