#!/usr/bin/env python3
"""
Dataset Generator for Experiment 4.2.3: Two-Step Method Triggering Accuracy

Generates synthetic datasets with controlled properties to test the Engine Orchestrator's
decision-making for when to trigger Step 2 refinement.

Properties controlled:
- Sparsity ratio (M / n*log(n)): Data sufficiency for spectral ranking
- Heterogeneity: Coefficient of variation in comparison counts
- Score distribution: Affects CI width through variance in estimates

Paper Section: 4.2.3 Two-Step Method Triggering Accuracy
"""

import os
import json
import math
import random
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "experiment_config.json")

# Load config
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

RANDOM_SEED = CONFIG.get("random_seed", 42)
DATASETS_PER_SCENARIO = CONFIG.get("datasets_per_scenario", 5)


# =============================================================================
# Utility Functions
# =============================================================================

def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_progress(current: int, total: int, message: str):
    """Print progress indicator."""
    pct = current / total * 100
    print(f"  [{current:3d}/{total:3d}] {pct:5.1f}% | {message}")


def calculate_sparsity_ratio(n_rows: int, n_items: int) -> float:
    """
    Calculate sparsity ratio as defined by spectral_ranking_step1.R.
    
    Formula: sparsity_ratio = nrow(df) / (n * log(n))
    where n = number of items (columns) and nrow = number of observations.
    
    This matches the R script's computation for consistency.
    """
    if n_items < 2:
        return 0.0
    n_log_n = n_items * math.log(n_items)
    return n_rows / n_log_n if n_log_n > 0 else 0.0


def rows_needed_for_sparsity(target_sparsity: float, n_items: int) -> int:
    """
    Calculate number of rows needed to achieve target sparsity ratio.
    
    Formula: n_rows = target_sparsity * n * log(n)
    """
    if n_items < 2:
        return 1
    n_log_n = n_items * math.log(n_items)
    rows = max(1, int(math.ceil(target_sparsity * n_log_n)))
    return rows


# =============================================================================
# Dataset Generation Functions
# =============================================================================

def generate_homogeneous_pointwise(
    n_rows: int,
    n_items: int,
    seed: int,
    score_spread: float = 2.0,
) -> pd.DataFrame:
    """
    Generate pointwise data with homogeneous comparison structure.
    All items have similar comparison counts (low heterogeneity).
    No missing values - all items participate in all comparisons.
    
    Args:
        n_rows: Number of observations/samples
        n_items: Number of items to rank
        seed: Random seed
        score_spread: Controls spread of true scores (affects CI width)
    
    Returns:
        DataFrame with items as columns, rows as observations
    """
    np.random.seed(seed)
    
    # Generate true scores for each item (well-separated to keep CI narrow)
    true_scores = np.linspace(0, score_spread, n_items)
    
    # Generate observations with moderate noise
    items = [f"Item_{i+1}" for i in range(n_items)]
    data = {}
    
    for i, item in enumerate(items):
        # Moderate noise for reasonable CI widths
        noise = np.random.normal(0, 0.5, n_rows)
        data[item] = true_scores[i] + noise
    
    return pd.DataFrame(data)


def generate_heterogeneous_pointwise(
    n_rows: int,
    n_items: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate pointwise data with HIGHLY heterogeneous comparison structure.
    
    To achieve CV > 0.5 in comparison counts:
    - Use a bimodal pattern: half items have full data, half have 70-80% missing
    - This creates a stark difference in comparison counts
    
    CV = sd/mean for counts [n, n, ..., n*0.2, n*0.2, ...]
    Example: n=10, half full, half at 20% -> counts = [100, 100, 100, 100, 100, 20, 20, 20, 20, 20]
    mean = 60, sd = 40, CV = 0.67 > 0.5
    
    Args:
        n_rows: Number of observations/samples
        n_items: Number of items to rank
        seed: Random seed
    
    Returns:
        DataFrame with items as columns, rows as observations
    """
    np.random.seed(seed)
    
    # Generate true scores
    true_scores = np.linspace(0, 3.0, n_items)
    
    # Generate base data
    items = [f"Item_{i+1}" for i in range(n_items)]
    data = {}
    
    for i, item in enumerate(items):
        noise = np.random.normal(0, 0.5, n_rows)
        data[item] = true_scores[i] + noise
    
    df = pd.DataFrame(data)
    
    # Create bimodal heterogeneous pattern
    # First half: no missing (100% data)
    # Second half: 75% missing (25% data)
    # This gives CV ~ 0.6-0.7 which exceeds 0.5 threshold
    half_point = n_items // 2
    
    for i, item in enumerate(items):
        if i >= half_point:
            # Second half has 75% missing
            missing_ratio = 0.75
            mask = np.random.random(n_rows) < missing_ratio
            df.loc[mask, item] = np.nan
    
    return df


def generate_high_uncertainty_pointwise(
    n_rows: int,
    n_items: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate pointwise data with HIGH NOISE to produce wide CIs.
    
    To achieve CI_width/n > 0.2:
    - Use closely spaced true scores
    - Add very high noise
    - With few items (n=5), CI_width > 1 means ratio > 20%
    
    Args:
        n_rows: Number of observations/samples
        n_items: Number of items to rank
        seed: Random seed
    
    Returns:
        DataFrame with items as columns, rows as observations
    """
    np.random.seed(seed)
    
    # Generate closely spaced true scores (hard to distinguish)
    true_scores = np.linspace(0, 0.3, n_items)  # Very small spread
    
    items = [f"Item_{i+1}" for i in range(n_items)]
    data = {}
    
    for i, item in enumerate(items):
        # Very high noise - many times larger than score differences
        noise = np.random.normal(0, 2.0, n_rows)
        data[item] = true_scores[i] + noise
    
    return pd.DataFrame(data)


def generate_heterogeneous_high_uncertainty_pointwise(
    n_rows: int,
    n_items: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate data with BOTH high heterogeneity AND high uncertainty.
    
    Combines:
    - Bimodal missing pattern (for CV > 0.5)
    - High noise with close true scores (for wide CIs)
    
    Args:
        n_rows: Number of observations/samples
        n_items: Number of items to rank
        seed: Random seed
    
    Returns:
        DataFrame with items as columns, rows as observations
    """
    np.random.seed(seed)
    
    # Closely spaced true scores (hard to rank)
    true_scores = np.linspace(0, 0.3, n_items)
    
    items = [f"Item_{i+1}" for i in range(n_items)]
    data = {}
    
    for i, item in enumerate(items):
        # High noise relative to score differences
        noise = np.random.normal(0, 2.0, n_rows)
        data[item] = true_scores[i] + noise
    
    df = pd.DataFrame(data)
    
    # Bimodal heterogeneous pattern (same as heterogeneous generator)
    half_point = n_items // 2
    
    for i, item in enumerate(items):
        if i >= half_point:
            missing_ratio = 0.70
            mask = np.random.random(n_rows) < missing_ratio
            df.loc[mask, item] = np.nan
    
    return df


def generate_low_uncertainty_pointwise(
    n_rows: int,
    n_items: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate pointwise data with low noise (leads to narrow CIs).
    
    Args:
        n_rows: Number of observations/samples
        n_items: Number of items to rank
        seed: Random seed
    
    Returns:
        DataFrame with items as columns, rows as observations
    """
    np.random.seed(seed)
    
    # Generate well-separated true scores
    true_scores = np.linspace(0, 5.0, n_items)  # Large spread
    
    items = [f"Item_{i+1}" for i in range(n_items)]
    data = {}
    
    for i, item in enumerate(items):
        # Low noise
        noise = np.random.normal(0, 0.1, n_rows)
        data[item] = true_scores[i] + noise
    
    return pd.DataFrame(data)


# =============================================================================
# Scenario Definitions
# =============================================================================

def create_scenario_datasets(
    scenario_id: str,
    n_datasets: int,
    base_seed: int,
) -> List[Dict]:
    """
    Create datasets for a specific test scenario.
    
    Returns list of dataset metadata dictionaries.
    """
    datasets = []
    
    # Scenario parameters
    # Note: sparsity_ratio = n_rows / (n * log(n))
    # For n=10: n*log(n) ~ 23, so need ~23 rows for sparsity=1.0
    # For n=8: n*log(n) ~ 17, so need ~17 rows for sparsity=1.0
    scenarios = {
        # =================================================================
        # GATEKEEPER TESTS (sparsity < 1.0 -> Step 2 blocked)
        # =================================================================
        "sparse_homogeneous": {
            "description": "Sparse data, low heterogeneity - Gatekeeper should block",
            "n_items": 10,
            "target_sparsity": 0.5,  # ~12 rows
            "generator": "homogeneous",
            "expected_step2": False,
            "expected_reason": "gatekeeper_blocked",
        },
        "sparse_heterogeneous": {
            "description": "Sparse data, high heterogeneity - Gatekeeper still blocks",
            "n_items": 10,
            "target_sparsity": 0.7,  # ~16 rows
            "generator": "heterogeneous",
            "expected_step2": False,
            "expected_reason": "gatekeeper_blocked",
        },
        
        # =================================================================
        # SUFFICIENT DATA + NO TRIGGERS (sparsity >= 1.0 but no triggers)
        # =================================================================
        "sufficient_no_trigger": {
            "description": "Sufficient data, low heterogeneity, narrow CI",
            "n_items": 8,
            "target_sparsity": 2.5,  # ~43 rows - plenty of data
            "generator": "homogeneous",  # No NaN -> heterogeneity = 0
            "expected_step2": False,
            "expected_reason": "no_triggers",
        },
        
        # =================================================================
        # SUFFICIENT DATA + HETEROGENEITY TRIGGER (CV > 0.5)
        # =================================================================
        "sufficient_heterogeneity": {
            "description": "Sufficient data, high heterogeneity - Trigger A should activate",
            "n_items": 10,
            "target_sparsity": 2.5,  # ~58 rows
            "generator": "heterogeneous",  # Uneven NaN -> high CV
            "expected_step2": True,
            "expected_reason": "heterogeneity_trigger",
        },
        
        # =================================================================
        # SUFFICIENT DATA + UNCERTAINTY TRIGGER (CI/n > 0.2)
        # =================================================================
        "sufficient_uncertainty": {
            "description": "Sufficient data, wide CI - Trigger B should activate",
            "n_items": 5,  # Few items so CI/n ratio can be > 20%
            "target_sparsity": 1.5,  # ~12 rows - just enough
            "generator": "high_uncertainty",  # High noise
            "expected_step2": True,
            "expected_reason": "uncertainty_trigger",
        },
        
        # =================================================================
        # SUFFICIENT DATA + BOTH TRIGGERS
        # =================================================================
        "sufficient_both": {
            "description": "Sufficient data, both triggers active",
            "n_items": 6,
            "target_sparsity": 2.0,  # ~22 rows
            "generator": "heterogeneous_high_uncertainty",
            "expected_step2": True,
            "expected_reason": "both_triggers",
        },
        
        # =================================================================
        # BOUNDARY TESTS (sparsity near 1.0)
        # =================================================================
        "boundary_sparsity_below": {
            "description": "Sparsity just below 1.0 - Gatekeeper blocks",
            "n_items": 10,
            "target_sparsity": 0.9,  # ~21 rows (threshold is ~23)
            "generator": "heterogeneous",
            "expected_step2": False,
            "expected_reason": "gatekeeper_blocked",
        },
        "boundary_sparsity_above": {
            "description": "Sparsity just above 1.0 with heterogeneity trigger",
            "n_items": 10,
            "target_sparsity": 1.2,  # ~28 rows
            "generator": "heterogeneous",
            "expected_step2": True,
            "expected_reason": "heterogeneity_trigger",
        },
        
        # =================================================================
        # SCALE TESTS (larger n)
        # =================================================================
        "large_items_sparse": {
            "description": "Many items (n=15), sparse data",
            "n_items": 15,
            "target_sparsity": 0.5,  # ~20 rows (threshold is ~41)
            "generator": "homogeneous",
            "expected_step2": False,
            "expected_reason": "gatekeeper_blocked",
        },
        "large_items_sufficient": {
            "description": "Many items (n=15), sufficient data with heterogeneity",
            "n_items": 15,
            "target_sparsity": 2.0,  # ~81 rows
            "generator": "heterogeneous",
            "expected_step2": True,
            "expected_reason": "heterogeneity_trigger",
        },
    }
    
    if scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    
    params = scenarios[scenario_id]
    
    for i in range(n_datasets):
        seed = base_seed + i
        n_items = params["n_items"]
        n_rows = rows_needed_for_sparsity(params["target_sparsity"], n_items)
        
        # Generate data based on generator type
        gen_type = params["generator"]
        
        if gen_type == "homogeneous":
            df = generate_homogeneous_pointwise(n_rows, n_items, seed)
        elif gen_type == "heterogeneous":
            df = generate_heterogeneous_pointwise(n_rows, n_items, seed)
        elif gen_type == "high_uncertainty":
            df = generate_high_uncertainty_pointwise(n_rows, n_items, seed)
        elif gen_type == "heterogeneous_high_uncertainty":
            df = generate_heterogeneous_high_uncertainty_pointwise(n_rows, n_items, seed)
        else:
            df = generate_homogeneous_pointwise(n_rows, n_items, seed)
        
        # Calculate actual sparsity
        actual_sparsity = calculate_sparsity_ratio(len(df), n_items)
        
        # Save dataset
        dataset_id = f"{scenario_id}_{i+1:02d}"
        filename = f"{scenario_id}/{dataset_id}.csv"
        
        # Create scenario directory
        scenario_dir = os.path.join(DATASETS_DIR, scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)
        
        csv_path = os.path.join(DATASETS_DIR, filename)
        df.to_csv(csv_path, index=False)
        
        # Metadata
        metadata = {
            "dataset_id": dataset_id,
            "filename": filename,
            "scenario": scenario_id,
            "description": params["description"],
            "n_rows": len(df),
            "n_items": n_items,
            "target_sparsity": params["target_sparsity"],
            "actual_sparsity": actual_sparsity,
            "generator": gen_type,
            "seed": seed,
            "expected_step2": params["expected_step2"],
            "expected_reason": params["expected_reason"],
        }
        datasets.append(metadata)
    
    return datasets


# =============================================================================
# Main Generation
# =============================================================================

def generate_all_datasets():
    """Generate all datasets for the experiment."""
    print_section("Generating Datasets for Experiment 4.2.3")
    print(f"  Output directory: {DATASETS_DIR}")
    print(f"  Datasets per scenario: {DATASETS_PER_SCENARIO}")
    print(f"  Random seed: {RANDOM_SEED}")
    
    # Create output directory
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Define scenarios to generate
    scenarios = [
        "sparse_homogeneous",
        "sparse_heterogeneous",
        "sufficient_no_trigger",
        "sufficient_heterogeneity",
        "sufficient_uncertainty",
        "sufficient_both",
        "boundary_sparsity_below",
        "boundary_sparsity_above",
        "large_items_sparse",
        "large_items_sufficient",
    ]
    
    all_datasets = []
    total_scenarios = len(scenarios)
    
    print_section("Generating Scenario Datasets")
    
    for idx, scenario in enumerate(scenarios):
        print_progress(idx + 1, total_scenarios, f"Scenario: {scenario}")
        
        base_seed = RANDOM_SEED + idx * 100
        datasets = create_scenario_datasets(
            scenario,
            DATASETS_PER_SCENARIO,
            base_seed,
        )
        all_datasets.extend(datasets)
    
    # Create manifest
    manifest = {
        "experiment_id": CONFIG["experiment_id"],
        "paper_section": CONFIG["paper_section"],
        "generated_at": datetime.now().isoformat(),
        "total_datasets": len(all_datasets),
        "scenarios": scenarios,
        "datasets_per_scenario": DATASETS_PER_SCENARIO,
        "random_seed": RANDOM_SEED,
        "datasets": all_datasets,
    }
    
    manifest_path = os.path.join(DATASETS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Summary
    print_section("Generation Summary")
    print(f"  Total datasets: {len(all_datasets)}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Manifest: {manifest_path}")
    
    # Count by expected decision
    expected_true = sum(1 for d in all_datasets if d["expected_step2"])
    expected_false = sum(1 for d in all_datasets if not d["expected_step2"])
    print(f"\n  Expected Step 2 triggered: {expected_true}")
    print(f"  Expected Step 2 skipped: {expected_false}")
    
    print("\n  Scenario breakdown:")
    for scenario in scenarios:
        count = sum(1 for d in all_datasets if d["scenario"] == scenario)
        sample = next((d for d in all_datasets if d["scenario"] == scenario), None)
        expected = "Yes" if sample and sample["expected_step2"] else "No"
        print(f"    {scenario:30s}: {count} datasets (Step2: {expected})")
    
    print_section("Generation Complete")


if __name__ == "__main__":
    generate_all_datasets()
