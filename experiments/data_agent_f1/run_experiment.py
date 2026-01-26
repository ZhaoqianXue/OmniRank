#!/usr/bin/env python3
"""
Experiment 4.2.1: Format Recognition & Standardization

Comprehensive evaluation of Data Agent's format detection and standardization capabilities.

Features:
- Multiple runs with different seeds for mean ± std computation
- Three evaluation targets: format detection, engine compatibility, standardization decision
- Baseline rule-based comparator for comparison
- Error analysis and detailed breakdown
- Publication-ready outputs (LaTeX/Markdown tables, confusion matrices, figures)

Paper Section: 4.2.1 Format Recognition & Standardization
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# EXPERIMENT CONFIGURATION - MODIFY HERE
# =============================================================================
# Number of experiment runs (for mean ± std calculation)
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

# Labels
FORMAT_LABELS = ["pointwise", "pairwise", "multiway", "invalid"]
COMPAT_LABELS = ["compatible", "incompatible"]
ACTION_LABELS = ["none", "standardize", "reject"]


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
    bar = "█" * filled + "░" * (bar_len - filled)
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

def compute_confusion_matrix(y_true: List, y_pred: List, labels: List) -> np.ndarray:
    """Compute confusion matrix."""
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        true_str = str(true).lower()
        pred_str = str(pred).lower()
        if true_str in label_to_idx and pred_str in label_to_idx:
            cm[label_to_idx[true_str], label_to_idx[pred_str]] += 1
    
    return cm


def compute_metrics_from_cm(cm: np.ndarray, labels: List) -> Dict:
    """Compute precision, recall, F1 per class and overall accuracy."""
    metrics = {}
    n_total = cm.sum()
    n_correct = np.trace(cm)
    
    metrics["overall_accuracy"] = n_correct / n_total if n_total > 0 else 0
    metrics["per_class"] = {}
    
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["per_class"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }
    
    return metrics


# =============================================================================
# Baseline Rule-Based Detector
# =============================================================================

class RuleBasedDetector:
    """
    Simple rule-based format detector as baseline.
    Uses value patterns and structural heuristics.
    """
    
    def detect_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect format using simple rules."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_numeric = len(numeric_cols)
        
        # Rule 1: Insufficient numeric columns -> invalid
        if n_numeric < 2:
            return {
                "format": "invalid",
                "engine_compatible": False,
                "standardization_action": "reject",
                "reason": "insufficient_numeric_columns",
            }
        
        # Get numeric data
        numeric_df = df[numeric_cols]
        
        # Rule 2: Check for pairwise pattern (sparse with mostly NaN, values are 0/1)
        nan_ratio = numeric_df.isna().sum().sum() / numeric_df.size
        if nan_ratio > 0.5:
            unique_values = set()
            for col in numeric_cols:
                unique_values.update(numeric_df[col].dropna().unique())
            
            if unique_values.issubset({0, 1, 0.0, 1.0, 0.5}):
                return {
                    "format": "pairwise",
                    "engine_compatible": True,
                    "standardization_action": "none",
                    "reason": "sparse_binary_pattern",
                }
        
        # Rule 3: Check for multiway pattern (small consecutive integers)
        all_values = numeric_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        
        if len(all_values) > 0:
            unique_values = sorted(set(all_values))
            if len(unique_values) <= 20:
                # Check if values are consecutive integers starting from 1
                int_values = [int(v) for v in unique_values if v == int(v)]
                if int_values == list(range(1, len(int_values) + 1)):
                    return {
                        "format": "multiway",
                        "engine_compatible": True,
                        "standardization_action": "none",
                        "reason": "consecutive_rank_integers",
                    }
        
        # Rule 4: Check column names for standardization need
        needs_std = False
        for col in df.columns:
            if any(c in str(col) for c in [' ', '(', ')', '-', '.']):
                needs_std = True
                break
        
        # Default: pointwise
        return {
            "format": "pointwise",
            "engine_compatible": not needs_std,
            "standardization_action": "standardize" if needs_std else "none",
            "reason": "default_dense_numeric",
        }


# =============================================================================
# Data Agent Wrapper
# =============================================================================

class DataAgentWrapper:
    """Wrapper for Data Agent that handles format detection."""
    
    def __init__(self):
        try:
            from agents.data_agent import DataAgent
            self.agent = DataAgent()
            self.enabled = self.agent.enabled
        except Exception as e:
            logger.warning(f"Failed to initialize DataAgent: {e}")
            self.agent = None
            self.enabled = False
    
    def detect_format(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Run format detection through Data Agent."""
        if not self.agent:
            return {
                "format": "error",
                "engine_compatible": False,
                "standardization_action": "reject",
                "error": "Agent not initialized",
            }
        
        try:
            schema, warnings, explanation = self.agent.process(content, filename)
            
            # Extract predictions
            fmt = format_to_string(schema.format)
            engine_compatible = getattr(schema, 'engine_compatible', True)
            standardization_needed = getattr(schema, 'standardization_needed', False)
            
            if fmt == "invalid" or not engine_compatible:
                action = "reject" if standardization_needed == False else "standardize"
            elif standardization_needed:
                action = "standardize"
            else:
                action = "none"
            
            return {
                "format": fmt,
                "engine_compatible": engine_compatible,
                "standardization_action": action,
                "schema": schema,
                "warnings": warnings,
                "explanation": explanation,
            }
        except Exception as e:
            return {
                "format": "error",
                "engine_compatible": False,
                "standardization_action": "reject",
                "error": str(e),
            }


# =============================================================================
# Single Run Evaluation
# =============================================================================

def run_single_evaluation(
    datasets: List[Dict],
    detector: Any,
    detector_name: str,
    run_id: int,
) -> Dict:
    """Run evaluation for a single detector on all datasets."""
    
    results = {
        "run_id": run_id,
        "detector": detector_name,
        "format_predictions": {"y_true": [], "y_pred": []},
        "compat_predictions": {"y_true": [], "y_pred": []},
        "action_predictions": {"y_true": [], "y_pred": []},
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
        gt_format = ds_meta["expected_format"]
        gt_compat = "compatible" if ds_meta.get("expected_engine_compatible", True) else "incompatible"
        gt_action = ds_meta.get("expected_standardization_action", "none")
        
        print_progress(idx + 1, total, f"[Run {run_id}] {detector_name}: {dataset_id}")
        
        # Load data
        csv_path = os.path.join(DATASETS_DIR, filename)
        
        try:
            if isinstance(detector, DataAgentWrapper):
                with open(csv_path, "rb") as f:
                    content = f.read()
                prediction = detector.detect_format(content, os.path.basename(filename))
            else:
                df = pd.read_csv(csv_path)
                prediction = detector.detect_format(df)
            
            pred_format = prediction.get("format", "error")
            pred_compat = "compatible" if prediction.get("engine_compatible", True) else "incompatible"
            pred_action = prediction.get("standardization_action", "none")
            
        except Exception as e:
            pred_format = "error"
            pred_compat = "incompatible"
            pred_action = "reject"
            results["errors"].append({"dataset_id": dataset_id, "error": str(e)})
        
        # Store predictions
        results["format_predictions"]["y_true"].append(gt_format)
        results["format_predictions"]["y_pred"].append(pred_format)
        
        results["compat_predictions"]["y_true"].append(gt_compat)
        results["compat_predictions"]["y_pred"].append(pred_compat)
        
        results["action_predictions"]["y_true"].append(gt_action)
        results["action_predictions"]["y_pred"].append(pred_action)
        
        # Detailed result
        detail = {
            "dataset_id": dataset_id,
            "category": category,
            "difficulty": ds_meta.get("difficulty", "unknown"),
            "ground_truth": {
                "format": gt_format,
                "engine_compatible": gt_compat,
                "standardization_action": gt_action,
            },
            "prediction": {
                "format": pred_format,
                "engine_compatible": pred_compat,
                "standardization_action": pred_action,
            },
            "correct": {
                "format": gt_format == pred_format,
                "engine_compatible": gt_compat == pred_compat,
                "standardization_action": gt_action == pred_action,
            },
        }
        results["detailed_results"].append(detail)
        results["category_results"][category].append(detail)
    
    return results


# =============================================================================
# Aggregated Metrics
# =============================================================================

def aggregate_run_results(all_runs: List[Dict], labels_map: Dict) -> Dict:
    """Aggregate results across multiple runs for mean ± std."""
    
    aggregated = {
        "format": {"accuracies": [], "per_class_metrics": defaultdict(lambda: defaultdict(list))},
        "compat": {"accuracies": [], "per_class_metrics": defaultdict(lambda: defaultdict(list))},
        "action": {"accuracies": [], "per_class_metrics": defaultdict(lambda: defaultdict(list))},
        "category_accuracies": defaultdict(list),
    }
    
    for run in all_runs:
        # Format detection metrics
        cm_format = compute_confusion_matrix(
            run["format_predictions"]["y_true"],
            run["format_predictions"]["y_pred"],
            labels_map["format"]
        )
        metrics_format = compute_metrics_from_cm(cm_format, labels_map["format"])
        aggregated["format"]["accuracies"].append(metrics_format["overall_accuracy"])
        for label, m in metrics_format["per_class"].items():
            for metric_name, value in m.items():
                aggregated["format"]["per_class_metrics"][label][metric_name].append(value)
        
        # Compatibility metrics
        cm_compat = compute_confusion_matrix(
            run["compat_predictions"]["y_true"],
            run["compat_predictions"]["y_pred"],
            labels_map["compat"]
        )
        metrics_compat = compute_metrics_from_cm(cm_compat, labels_map["compat"])
        aggregated["compat"]["accuracies"].append(metrics_compat["overall_accuracy"])
        for label, m in metrics_compat["per_class"].items():
            for metric_name, value in m.items():
                aggregated["compat"]["per_class_metrics"][label][metric_name].append(value)
        
        # Standardization action metrics
        cm_action = compute_confusion_matrix(
            run["action_predictions"]["y_true"],
            run["action_predictions"]["y_pred"],
            labels_map["action"]
        )
        metrics_action = compute_metrics_from_cm(cm_action, labels_map["action"])
        aggregated["action"]["accuracies"].append(metrics_action["overall_accuracy"])
        for label, m in metrics_action["per_class"].items():
            for metric_name, value in m.items():
                aggregated["action"]["per_class_metrics"][label][metric_name].append(value)
        
        # Category-level accuracy
        for category, details in run["category_results"].items():
            if details:
                cat_acc = sum(1 for d in details if d["correct"]["format"]) / len(details)
                aggregated["category_accuracies"][category].append(cat_acc)
    
    return aggregated


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std


# =============================================================================
# Output Generation (LaTeX & Markdown)
# =============================================================================

def generate_latex_table(aggregated: Dict, detector_name: str, n_runs: int) -> str:
    """Generate LaTeX table with mean ± std for JASA submission."""
    lines = []
    lines.append("% Table: Format Detection Performance (Section 4.2.1)")
    lines.append(f"% Results over {n_runs} runs")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Format Detection Performance of {detector_name} (Mean $\\pm$ Std over {n_runs} runs)}}")
    lines.append("\\label{tab:format_detection}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Format & Precision & Recall & F1 & Support \\\\")
    lines.append("\\midrule")
    
    for label in FORMAT_LABELS:
        metrics = aggregated["format"]["per_class_metrics"].get(label, {})
        p_mean, p_std = compute_mean_std(metrics.get("precision", []))
        r_mean, r_std = compute_mean_std(metrics.get("recall", []))
        f1_mean, f1_std = compute_mean_std(metrics.get("f1", []))
        support = int(np.mean(metrics.get("support", [0])))
        
        lines.append(f"{label.capitalize()} & {p_mean:.3f}$\\pm${p_std:.3f} & {r_mean:.3f}$\\pm${r_std:.3f} & {f1_mean:.3f}$\\pm${f1_std:.3f} & {support} \\\\")
    
    acc_mean, acc_std = compute_mean_std(aggregated["format"]["accuracies"])
    lines.append("\\midrule")
    lines.append(f"\\textbf{{Overall Accuracy}} & \\multicolumn{{4}}{{c}}{{{acc_mean:.3f}$\\pm${acc_std:.3f}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Category breakdown table
    lines.append("")
    lines.append("% Table: Performance by Dataset Category")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Format Detection Accuracy by Dataset Category}")
    lines.append("\\label{tab:format_by_category}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("Category & Accuracy \\\\")
    lines.append("\\midrule")
    
    for category in ["standard", "ambiguous", "transposed", "invalid", "realworld"]:
        accs = aggregated["category_accuracies"].get(category, [])
        if accs:
            mean, std = compute_mean_std(accs)
            lines.append(f"{category.capitalize()} & {mean:.3f}$\\pm${std:.3f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_markdown_table(aggregated: Dict, detector_name: str, n_runs: int) -> str:
    """Generate Markdown table."""
    lines = []
    lines.append(f"## Format Detection Performance ({detector_name})")
    lines.append(f"*Results over {n_runs} runs (mean ± std)*\n")
    lines.append("| Format | Precision | Recall | F1 | Support |")
    lines.append("|--------|-----------|--------|-----|---------|")
    
    for label in FORMAT_LABELS:
        metrics = aggregated["format"]["per_class_metrics"].get(label, {})
        p_mean, p_std = compute_mean_std(metrics.get("precision", []))
        r_mean, r_std = compute_mean_std(metrics.get("recall", []))
        f1_mean, f1_std = compute_mean_std(metrics.get("f1", []))
        support = int(np.mean(metrics.get("support", [0])))
        
        lines.append(f"| {label.capitalize()} | {p_mean:.3f}±{p_std:.3f} | {r_mean:.3f}±{r_std:.3f} | {f1_mean:.3f}±{f1_std:.3f} | {support} |")
    
    acc_mean, acc_std = compute_mean_std(aggregated["format"]["accuracies"])
    lines.append(f"| **Overall Accuracy** | | | **{acc_mean:.3f}±{acc_std:.3f}** | |")
    
    lines.append("\n### Performance by Category\n")
    lines.append("| Category | Accuracy |")
    lines.append("|----------|----------|")
    
    for category in ["standard", "ambiguous", "transposed", "invalid", "realworld"]:
        accs = aggregated["category_accuracies"].get(category, [])
        if accs:
            mean, std = compute_mean_std(accs)
            lines.append(f"| {category.capitalize()} | {mean:.3f}±{std:.3f} |")
    
    return "\n".join(lines)


def save_confusion_matrix_plot(cm: np.ndarray, labels: List, output_path: str, title: str):
    """Save confusion matrix as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize for display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[l.capitalize() for l in labels],
            yticklabels=[l.capitalize() for l in labels],
            ax=ax, vmin=0, vmax=1
        )
        
        # Add raw counts as text
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j + 0.5, i + 0.75, f'(n={cm[i, j]})',
                       ha='center', va='center', fontsize=8, color='gray')
        
        ax.set_xlabel('Predicted Format')
        ax.set_ylabel('True Format')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    except ImportError as e:
        print(f"  Warning: Cannot generate plot ({e})")


def generate_error_analysis(all_runs: List[Dict]) -> Dict:
    """Generate detailed error analysis."""
    
    error_analysis = {
        "misclassified_datasets": defaultdict(list),
        "error_patterns": defaultdict(int),
        "category_error_rates": defaultdict(lambda: {"total": 0, "errors": 0}),
        "difficulty_error_rates": defaultdict(lambda: {"total": 0, "errors": 0}),
    }
    
    # Use first run for detailed analysis (patterns should be consistent)
    run = all_runs[0]
    
    for detail in run["detailed_results"]:
        category = detail["category"]
        difficulty = detail["difficulty"]
        
        error_analysis["category_error_rates"][category]["total"] += 1
        error_analysis["difficulty_error_rates"][difficulty]["total"] += 1
        
        if not detail["correct"]["format"]:
            error_analysis["category_error_rates"][category]["errors"] += 1
            error_analysis["difficulty_error_rates"][difficulty]["errors"] += 1
            
            gt = detail["ground_truth"]["format"]
            pred = detail["prediction"]["format"]
            pattern = f"{gt} -> {pred}"
            
            error_analysis["error_patterns"][pattern] += 1
            error_analysis["misclassified_datasets"][pattern].append(detail["dataset_id"])
    
    # Compute error rates
    for key in error_analysis["category_error_rates"]:
        stats = error_analysis["category_error_rates"][key]
        stats["error_rate"] = stats["errors"] / stats["total"] if stats["total"] > 0 else 0
    
    for key in error_analysis["difficulty_error_rates"]:
        stats = error_analysis["difficulty_error_rates"][key]
        stats["error_rate"] = stats["errors"] / stats["total"] if stats["total"] > 0 else 0
    
    return error_analysis


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the complete format detection experiment."""
    print_section("Experiment 4.2.1: Format Recognition & Standardization")
    
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
    n_runs = N_RUNS  # Use explicit N_RUNS variable (modify at top of file)
    
    print(f"  Run ID: {run_timestamp}")
    print(f"  Datasets: {n_datasets}")
    print(f"  Runs: {n_runs} (modify N_RUNS at top of file to change)")
    print(f"  Output: {run_dir}")
    
    # Initialize detectors
    agent_wrapper = DataAgentWrapper()
    baseline = RuleBasedDetector()
    
    print(f"\n  Data Agent LLM Enabled: {agent_wrapper.enabled}")
    
    labels_map = {
        "format": FORMAT_LABELS,
        "compat": COMPAT_LABELS,
        "action": ACTION_LABELS,
    }
    
    # Run multiple evaluations
    print_section("Running Evaluations")
    
    agent_runs = []
    baseline_runs = []
    
    for run_id in range(1, n_runs + 1):
        print(f"\n--- Run {run_id}/{n_runs} ---")
        
        # Data Agent evaluation
        agent_result = run_single_evaluation(datasets, agent_wrapper, "DataAgent", run_id)
        agent_runs.append(agent_result)
        
        # Baseline evaluation
        baseline_result = run_single_evaluation(datasets, baseline, "Baseline", run_id)
        baseline_runs.append(baseline_result)
    
    # Aggregate results
    print_section("Aggregating Results")
    
    agent_aggregated = aggregate_run_results(agent_runs, labels_map)
    baseline_aggregated = aggregate_run_results(baseline_runs, labels_map)
    
    # Print summary
    print_section("Results Summary")
    
    agent_acc_mean, agent_acc_std = compute_mean_std(agent_aggregated["format"]["accuracies"])
    baseline_acc_mean, baseline_acc_std = compute_mean_std(baseline_aggregated["format"]["accuracies"])
    
    print(f"\n  Format Detection Accuracy:")
    print(f"    DataAgent:  {agent_acc_mean:.3f} ± {agent_acc_std:.3f}")
    print(f"    Baseline:   {baseline_acc_mean:.3f} ± {baseline_acc_std:.3f}")
    
    print(f"\n  By Category (DataAgent):")
    for category in ["standard", "ambiguous", "transposed", "invalid", "realworld"]:
        accs = agent_aggregated["category_accuracies"].get(category, [])
        if accs:
            mean, std = compute_mean_std(accs)
            print(f"    {category:12s}: {mean:.3f} ± {std:.3f}")
    
    # Save results
    print_section("Saving Outputs")
    
    # Compute final confusion matrix (aggregated from all runs)
    all_y_true_format = []
    all_y_pred_format = []
    for run in agent_runs:
        all_y_true_format.extend(run["format_predictions"]["y_true"])
        all_y_pred_format.extend(run["format_predictions"]["y_pred"])
    
    cm_format = compute_confusion_matrix(all_y_true_format, all_y_pred_format, FORMAT_LABELS)
    
    # Raw results JSON
    raw_results = {
        "experiment_id": CONFIG["experiment_id"],
        "paper_section": CONFIG["paper_section"],
        "run_timestamp": run_timestamp,
        "n_datasets": n_datasets,
        "n_runs": n_runs,
        "agent_enabled": agent_wrapper.enabled,
        "results": {
            "data_agent": {
                "format_accuracy": {"mean": agent_acc_mean, "std": agent_acc_std},
                "category_accuracies": {
                    cat: {"mean": m, "std": s} 
                    for cat in agent_aggregated["category_accuracies"]
                    for m, s in [compute_mean_std(agent_aggregated["category_accuracies"][cat])]
                },
                "per_class_format_metrics": {
                    label: {
                        metric: {"mean": m, "std": s}
                        for metric in agent_aggregated["format"]["per_class_metrics"][label]
                        for m, s in [compute_mean_std(agent_aggregated["format"]["per_class_metrics"][label][metric])]
                    }
                    for label in FORMAT_LABELS
                },
            },
            "baseline": {
                "format_accuracy": {"mean": baseline_acc_mean, "std": baseline_acc_std},
                "category_accuracies": {
                    cat: {"mean": m, "std": s} 
                    for cat in baseline_aggregated["category_accuracies"]
                    for m, s in [compute_mean_std(baseline_aggregated["category_accuracies"][cat])]
                },
            },
        },
        "confusion_matrix_format": cm_format.tolist(),
        "labels": FORMAT_LABELS,
    }
    
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"  Results JSON: {json_path}")
    
    # Error analysis
    error_analysis = generate_error_analysis(agent_runs)
    error_path = os.path.join(run_dir, "error_analysis.json")
    with open(error_path, "w") as f:
        json.dump(error_analysis, f, indent=2, default=list)
    print(f"  Error Analysis: {error_path}")
    
    # LaTeX table
    latex_content = generate_latex_table(agent_aggregated, "Data Agent", n_runs)
    latex_path = os.path.join(run_dir, "table_4_2_1.tex")
    with open(latex_path, "w") as f:
        f.write(latex_content)
    print(f"  LaTeX table: {latex_path}")
    
    # Markdown table
    md_content = generate_markdown_table(agent_aggregated, "Data Agent", n_runs)
    md_path = os.path.join(run_dir, "table_4_2_1.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  Markdown table: {md_path}")
    
    # Confusion matrix plot
    plot_path = os.path.join(run_dir, "confusion_matrix.png")
    save_confusion_matrix_plot(
        cm_format, FORMAT_LABELS, plot_path,
        f"Format Detection Confusion Matrix (n={n_runs} runs)"
    )
    
    # Print tables
    print_section("LaTeX Output (Copy to Paper)")
    print(latex_content)
    
    print_section("Markdown Output")
    print(md_content)
    
    print_section("Experiment Complete")
    print(f"  All outputs saved to: {run_dir}")


if __name__ == "__main__":
    run_experiment()
