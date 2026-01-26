#!/usr/bin/env python3
"""
Dataset Generation for Experiment 4.2.2: Semantic Schema Inference

Generates test datasets to evaluate Data Agent's semantic understanding:
- BigBetter Direction: Higher vs Lower is better
- Ranking Items Identification: Which columns are entities to rank
- Indicator Column Detection: Categorical stratification dimension
- Indicator Values Extraction: Unique groups within indicator

Dataset Categories:
- bigbetter_high: Columns with names suggesting higher is better (accuracy, score, f1)
- bigbetter_low: Columns with names suggesting lower is better (error, latency, rank)
- bigbetter_ambiguous: Neutral column names (model_A, item_1)
- indicator_clear: Clear categorical indicator column present
- indicator_none: No indicator column
- items_complex: Mixed metadata and ranking columns

Paper Section: 4.2.2 Semantic Schema Inference
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "experiment_config.json")

# Load config
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

RANDOM_SEED = CONFIG.get("random_seed", 42)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def print_progress(current: int, total: int, category: str, message: str):
    """Print progress indicator."""
    pct = current / total * 100
    print(f"  [{current:2d}/{total}] {pct:5.1f}% | [{category:20s}] {message}")


# =============================================================================
# Category A: BigBetter = 1 (Higher is Better)
# =============================================================================

def generate_bigbetter_high_accuracy(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with accuracy-related terms.
    Expected: bigbetter = 1
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    # Accuracy-related column names (higher is better)
    name_patterns = [
        [f"accuracy_{chr(65+i)}" for i in range(n_items)],
        [f"model_{chr(65+i)}_accuracy" for i in range(n_items)],
        [f"acc_{i+1}" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    # Generate scores in [0, 1] range
    data = {}
    true_scores = np.linspace(0.85, 0.45, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.08, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_high_accuracy_{dataset_id:02d}",
        "category": "bigbetter_high",
        "expected_bigbetter": 1,
        "bigbetter_hint": "column_names_contain_accuracy",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Accuracy scores - column names contain 'accuracy'",
    }


def generate_bigbetter_high_score(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with score-related terms.
    Expected: bigbetter = 1
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    name_patterns = [
        [f"score_{chr(65+i)}" for i in range(n_items)],
        [f"f1_score_{i+1}" for i in range(n_items)],
        [f"model_{chr(65+i)}_score" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    data = {}
    true_scores = np.linspace(0.9, 0.5, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_high_score_{dataset_id:02d}",
        "category": "bigbetter_high",
        "expected_bigbetter": 1,
        "bigbetter_hint": "column_names_contain_score",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Score values - column names contain 'score' or 'f1'",
    }


def generate_bigbetter_high_win_rate(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with win/reward-related terms.
    Expected: bigbetter = 1
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(40, 120)
    
    name_patterns = [
        [f"win_rate_{chr(65+i)}" for i in range(n_items)],
        [f"player_{i+1}_wins" for i in range(n_items)],
        [f"reward_{chr(65+i)}" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    data = {}
    true_rates = np.linspace(0.75, 0.35, n_items)
    for idx, item in enumerate(item_names):
        rates = true_rates[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(rates, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_high_win_{dataset_id:02d}",
        "category": "bigbetter_high",
        "expected_bigbetter": 1,
        "bigbetter_hint": "column_names_contain_win_or_reward",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Win rates or rewards - column names contain 'win' or 'reward'",
    }


def generate_bigbetter_high_precision(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with precision/recall/auc terms.
    Expected: bigbetter = 1
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(30, 80)
    
    metrics = ["precision", "recall", "auc", "mAP"]
    chosen_metric = random.choice(metrics)
    item_names = [f"{chosen_metric}_{chr(65+i)}" for i in range(n_items)]
    
    data = {}
    true_scores = np.linspace(0.88, 0.52, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.08, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_high_metric_{dataset_id:02d}",
        "category": "bigbetter_high",
        "expected_bigbetter": 1,
        "bigbetter_hint": f"column_names_contain_{chosen_metric}",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": f"ML metrics ({chosen_metric}) - higher is better",
    }


# =============================================================================
# Category B: BigBetter = 0 (Lower is Better)
# =============================================================================

def generate_bigbetter_low_error(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with error-related terms.
    Expected: bigbetter = 0
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    name_patterns = [
        [f"error_{chr(65+i)}" for i in range(n_items)],
        [f"model_{chr(65+i)}_error" for i in range(n_items)],
        [f"mse_{i+1}" for i in range(n_items)],
        [f"rmse_{chr(65+i)}" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    data = {}
    true_errors = np.linspace(0.05, 0.45, n_items)  # Lower errors for better items
    for idx, item in enumerate(item_names):
        errors = true_errors[idx] + np.random.normal(0, 0.05, n_samples)
        data[item] = np.clip(errors, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_low_error_{dataset_id:02d}",
        "category": "bigbetter_low",
        "expected_bigbetter": 0,
        "bigbetter_hint": "column_names_contain_error_or_mse",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Error metrics - column names contain 'error', 'mse', 'rmse'",
    }


def generate_bigbetter_low_loss(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with loss-related terms.
    Expected: bigbetter = 0
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    name_patterns = [
        [f"loss_{chr(65+i)}" for i in range(n_items)],
        [f"model_{chr(65+i)}_loss" for i in range(n_items)],
        [f"cross_entropy_{i+1}" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    data = {}
    true_losses = np.linspace(0.1, 0.8, n_items)
    for idx, item in enumerate(item_names):
        losses = true_losses[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(losses, 0, 2)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_low_loss_{dataset_id:02d}",
        "category": "bigbetter_low",
        "expected_bigbetter": 0,
        "bigbetter_hint": "column_names_contain_loss",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Loss values - column names contain 'loss' or 'cross_entropy'",
    }


def generate_bigbetter_low_latency(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with latency/time-related terms.
    Expected: bigbetter = 0
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(40, 120)
    
    name_patterns = [
        [f"latency_{chr(65+i)}" for i in range(n_items)],
        [f"response_time_{i+1}" for i in range(n_items)],
        [f"time_{chr(65+i)}_ms" for i in range(n_items)],
        [f"inference_time_{chr(65+i)}" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    data = {}
    true_times = np.linspace(10, 200, n_items)  # Lower time for better items
    for idx, item in enumerate(item_names):
        times = true_times[idx] + np.random.normal(0, 15, n_samples)
        data[item] = np.clip(times, 1, 500)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_low_latency_{dataset_id:02d}",
        "category": "bigbetter_low",
        "expected_bigbetter": 0,
        "bigbetter_hint": "column_names_contain_latency_or_time",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Latency/time metrics - column names contain 'latency', 'time'",
    }


def generate_bigbetter_low_rank(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with rank-related terms (lower rank = better).
    Expected: bigbetter = 0
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 80)
    
    name_patterns = [
        [f"rank_{chr(65+i)}" for i in range(n_items)],
        [f"position_{i+1}" for i in range(n_items)],
        [f"ranking_{chr(65+i)}" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    data = {}
    true_ranks = np.linspace(1, n_items, n_items)
    for idx, item in enumerate(item_names):
        ranks = true_ranks[idx] + np.random.normal(0, 1, n_samples)
        data[item] = np.clip(np.round(ranks), 1, n_items * 2).astype(int)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_low_rank_{dataset_id:02d}",
        "category": "bigbetter_low",
        "expected_bigbetter": 0,
        "bigbetter_hint": "column_names_contain_rank_or_position",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Rank positions - column names contain 'rank' or 'position'",
    }


def generate_bigbetter_low_cost(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Columns named with cost/distance-related terms.
    Expected: bigbetter = 0
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(30, 80)
    
    name_patterns = [
        [f"cost_{chr(65+i)}" for i in range(n_items)],
        [f"distance_{i+1}" for i in range(n_items)],
        [f"penalty_{chr(65+i)}" for i in range(n_items)],
    ]
    item_names = random.choice(name_patterns)
    
    data = {}
    true_costs = np.linspace(5, 50, n_items)
    for idx, item in enumerate(item_names):
        costs = true_costs[idx] + np.random.normal(0, 5, n_samples)
        data[item] = np.clip(costs, 0, 100)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_low_cost_{dataset_id:02d}",
        "category": "bigbetter_low",
        "expected_bigbetter": 0,
        "bigbetter_hint": "column_names_contain_cost_or_distance",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Cost/distance metrics - column names contain 'cost', 'distance', 'penalty'",
    }


# =============================================================================
# Category C: BigBetter Ambiguous (Neutral Column Names)
# =============================================================================

def generate_bigbetter_ambiguous_model(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Neutral column names like model_A, model_B.
    Values in [0, 1] suggest probability/accuracy -> bigbetter=1 by convention.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    item_names = [f"model_{chr(65+i)}" for i in range(n_items)]
    
    data = {}
    true_scores = np.linspace(0.85, 0.45, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)  # [0,1] range suggests higher=better
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_amb_model_{dataset_id:02d}",
        "category": "bigbetter_ambiguous",
        "expected_bigbetter": 1,  # Convention: [0,1] bounded -> higher is better
        "bigbetter_hint": "neutral_names_but_bounded_0_1_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "medium",
        "description": "Neutral names (model_X) with [0,1] values - ambiguous but likely accuracy",
    }


def generate_bigbetter_ambiguous_item(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Neutral column names like item_1, item_2 with unbounded positive values.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    item_names = [f"item_{i+1}" for i in range(n_items)]
    
    data = {}
    true_values = np.linspace(100, 20, n_items)
    for idx, item in enumerate(item_names):
        values = true_values[idx] + np.random.normal(0, 15, n_samples)
        data[item] = np.clip(values, 0, 200)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_amb_item_{dataset_id:02d}",
        "category": "bigbetter_ambiguous",
        "expected_bigbetter": 1,  # Default convention
        "bigbetter_hint": "neutral_names_unbounded_positive_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "hard",
        "description": "Neutral names (item_X) with unbounded values - truly ambiguous",
    }


def generate_bigbetter_ambiguous_algorithm(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Column names like algorithm_1, system_A.
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(30, 80)
    
    patterns = [
        [f"algorithm_{i+1}" for i in range(n_items)],
        [f"system_{chr(65+i)}" for i in range(n_items)],
        [f"method_{i+1}" for i in range(n_items)],
    ]
    item_names = random.choice(patterns)
    
    data = {}
    true_scores = np.linspace(0.8, 0.4, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"bb_amb_algo_{dataset_id:02d}",
        "category": "bigbetter_ambiguous",
        "expected_bigbetter": 1,  # Convention for [0,1]
        "bigbetter_hint": "neutral_names_bounded_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "medium",
        "description": "Neutral names (algorithm/system/method) with [0,1] values",
    }


# =============================================================================
# Category D: Indicator Column Present (Clear)
# =============================================================================

def generate_indicator_clear_task(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Clear 'Task' indicator column with semantic categories.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(40, 120)
    
    item_names = [f"model_{chr(65+i)}" for i in range(n_items)]
    indicator_values = ["code", "math", "reasoning", "writing"]
    
    data = {"Task": random.choices(indicator_values, k=n_samples)}
    
    true_scores = np.linspace(0.85, 0.45, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"ind_clear_task_{dataset_id:02d}",
        "category": "indicator_clear",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": "Task",
        "expected_indicator_values": indicator_values,
        "difficulty": "easy",
        "description": "Clear 'Task' indicator with semantic categories",
    }


def generate_indicator_clear_category(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Clear 'Category' indicator column.
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(40, 100)
    
    item_names = [f"product_{i+1}" for i in range(n_items)]
    indicator_values = ["electronics", "clothing", "food", "home"]
    
    data = {"Category": random.choices(indicator_values, k=n_samples)}
    
    true_scores = np.linspace(4.5, 2.5, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.5, n_samples)
        data[item] = np.clip(scores, 1, 5)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"ind_clear_category_{dataset_id:02d}",
        "category": "indicator_clear",
        "expected_bigbetter": 1,
        "bigbetter_hint": "rating_scale_1_to_5",
        "expected_ranking_items": item_names,
        "expected_indicator_col": "Category",
        "expected_indicator_values": indicator_values,
        "difficulty": "easy",
        "description": "Clear 'Category' indicator for product ratings",
    }


def generate_indicator_clear_domain(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    'Domain' indicator column with scientific domains.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(50, 150)
    
    item_names = [f"algorithm_{chr(65+i)}" for i in range(n_items)]
    indicator_values = ["biology", "physics", "chemistry", "medicine"]
    
    data = {"domain": random.choices(indicator_values, k=n_samples)}
    
    true_scores = np.linspace(0.9, 0.5, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"ind_clear_domain_{dataset_id:02d}",
        "category": "indicator_clear",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": "domain",
        "expected_indicator_values": indicator_values,
        "difficulty": "easy",
        "description": "Clear 'domain' indicator for scientific domains",
    }


def generate_indicator_clear_group(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    'Group' or 'Type' indicator column.
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(40, 100)
    
    item_names = [f"method_{i+1}" for i in range(n_items)]
    indicator_name = random.choice(["group", "type", "class"])
    indicator_values = ["A", "B", "C"]
    
    data = {indicator_name: random.choices(indicator_values, k=n_samples)}
    
    true_scores = np.linspace(0.85, 0.55, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"ind_clear_group_{dataset_id:02d}",
        "category": "indicator_clear",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": indicator_name,
        "expected_indicator_values": indicator_values,
        "difficulty": "easy",
        "description": f"Clear '{indicator_name}' indicator with simple categories",
    }


# =============================================================================
# Category E: No Indicator Column
# =============================================================================

def generate_indicator_none_pure_numeric(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Pure numeric matrix with no categorical columns.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    item_names = [f"model_{chr(65+i)}" for i in range(n_items)]
    
    data = {}
    true_scores = np.linspace(0.85, 0.45, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"ind_none_numeric_{dataset_id:02d}",
        "category": "indicator_none",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "easy",
        "description": "Pure numeric matrix - no indicator column",
    }


def generate_indicator_none_with_id(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Numeric matrix with ID column (should NOT be indicator).
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    item_names = [f"algorithm_{i+1}" for i in range(n_items)]
    
    data = {"sample_id": [f"s{i+1:04d}" for i in range(n_samples)]}
    
    true_scores = np.linspace(0.85, 0.45, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"ind_none_with_id_{dataset_id:02d}",
        "category": "indicator_none",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,  # sample_id is NOT an indicator
        "expected_indicator_values": [],
        "difficulty": "medium",
        "description": "Numeric with ID column - ID should NOT be indicator",
    }


def generate_indicator_none_high_cardinality(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Categorical column with too high cardinality (not suitable as indicator).
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(50, 150)
    
    item_names = [f"system_{chr(65+i)}" for i in range(n_items)]
    
    # High cardinality categorical (unique per row)
    data = {"experiment_id": [f"exp_{i+1:05d}" for i in range(n_samples)]}
    
    true_scores = np.linspace(0.85, 0.55, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"ind_none_high_card_{dataset_id:02d}",
        "category": "indicator_none",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,
        "expected_indicator_col": None,  # Too high cardinality
        "expected_indicator_values": [],
        "difficulty": "medium",
        "description": "High cardinality categorical - not suitable as indicator",
    }


# =============================================================================
# Category F: Items with Complex Metadata
# =============================================================================

def generate_items_complex_with_description(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Ranking items mixed with description/metadata columns.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    item_names = [f"model_{chr(65+i)}" for i in range(n_items)]
    
    data = {
        "case_num": list(range(1, n_samples + 1)),
        "description": [f"Test case {i+1}" for i in range(n_samples)],
    }
    
    true_scores = np.linspace(0.85, 0.45, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    # Add indicator
    data["task_type"] = random.choices(["qa", "code", "math"], k=n_samples)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"items_complex_desc_{dataset_id:02d}",
        "category": "items_complex",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,  # Excludes case_num, description
        "expected_indicator_col": "task_type",
        "expected_indicator_values": ["qa", "code", "math"],
        "difficulty": "medium",
        "description": "Ranking items with metadata columns to exclude",
    }


def generate_items_complex_multi_metadata(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Multiple metadata columns that should be excluded.
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(40, 100)
    
    item_names = [f"algorithm_{chr(65+i)}" for i in range(n_items)]
    
    data = {
        "row_id": [f"r{i+1:04d}" for i in range(n_samples)],
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h").astype(str).tolist(),
        "notes": ["" for _ in range(n_samples)],
    }
    
    true_scores = np.linspace(0.9, 0.5, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"items_complex_multi_{dataset_id:02d}",
        "category": "items_complex",
        "expected_bigbetter": 1,
        "bigbetter_hint": "bounded_0_1_values",
        "expected_ranking_items": item_names,  # Excludes row_id, timestamp, notes
        "expected_indicator_col": None,
        "expected_indicator_values": [],
        "difficulty": "medium",
        "description": "Multiple metadata columns (id, timestamp, notes) to exclude",
    }


def generate_items_complex_mixed_columns(dataset_id: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Mixed scenario: ranking items + metadata + indicator.
    """
    n_items = random.randint(4, 6)
    n_samples = random.randint(40, 100)
    
    item_names = [f"score_{chr(65+i)}" for i in range(n_items)]
    indicator_values = ["benchmark_1", "benchmark_2", "benchmark_3"]
    
    data = {
        "experiment_name": [f"exp_{i+1}" for i in range(n_samples)],
        "dataset": random.choices(indicator_values, k=n_samples),
        "run_date": ["2024-01-01" for _ in range(n_samples)],
    }
    
    true_scores = np.linspace(0.9, 0.5, n_items)
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"items_complex_mixed_{dataset_id:02d}",
        "category": "items_complex",
        "expected_bigbetter": 1,
        "bigbetter_hint": "column_names_contain_score",
        "expected_ranking_items": item_names,
        "expected_indicator_col": "dataset",
        "expected_indicator_values": indicator_values,
        "difficulty": "hard",
        "description": "Complex mix: ranking items + metadata + indicator column",
    }


# =============================================================================
# Dataset Generation Dispatcher
# =============================================================================

GENERATORS = {
    "bigbetter_high": [
        (generate_bigbetter_high_accuracy, 2),
        (generate_bigbetter_high_score, 2),
        (generate_bigbetter_high_win_rate, 2),
        (generate_bigbetter_high_precision, 2),
    ],
    "bigbetter_low": [
        (generate_bigbetter_low_error, 2),
        (generate_bigbetter_low_loss, 2),
        (generate_bigbetter_low_latency, 2),
        (generate_bigbetter_low_rank, 2),
        (generate_bigbetter_low_cost, 2),
    ],
    "bigbetter_ambiguous": [
        (generate_bigbetter_ambiguous_model, 2),
        (generate_bigbetter_ambiguous_item, 2),
        (generate_bigbetter_ambiguous_algorithm, 2),
    ],
    "indicator_clear": [
        (generate_indicator_clear_task, 2),
        (generate_indicator_clear_category, 2),
        (generate_indicator_clear_domain, 2),
        (generate_indicator_clear_group, 2),
    ],
    "indicator_none": [
        (generate_indicator_none_pure_numeric, 2),
        (generate_indicator_none_with_id, 2),
        (generate_indicator_none_high_cardinality, 2),
    ],
    "items_complex": [
        (generate_items_complex_with_description, 2),
        (generate_items_complex_multi_metadata, 2),
        (generate_items_complex_mixed_columns, 2),
    ],
}


def main():
    print("=" * 70)
    print("Experiment 4.2.2: Semantic Schema Inference")
    print("Dataset Generation - Function 2 Test Cases")
    print("=" * 70)
    print(f"Output directory: {DATASETS_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print("-" * 70)
    
    # Create output directories
    os.makedirs(DATASETS_DIR, exist_ok=True)
    for category in GENERATORS.keys():
        os.makedirs(os.path.join(DATASETS_DIR, category), exist_ok=True)
    
    # Calculate total
    total_datasets = sum(count for gens in GENERATORS.values() for _, count in gens)
    
    manifest = {
        "experiment_id": CONFIG.get("experiment_id", "data_agent_f2"),
        "paper_section": CONFIG.get("paper_section", "4.2.2"),
        "generated_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_datasets": total_datasets,
        "categories": list(GENERATORS.keys()),
        "evaluation_targets": [
            "bigbetter",
            "ranking_items",
            "indicator_col",
            "indicator_values",
        ],
        "datasets": [],
    }
    
    current = 0
    
    for category, generator_list in GENERATORS.items():
        print(f"\n[{category.upper()}] Generating datasets...")
        
        for generator_func, count in generator_list:
            for i in range(count):
                current += 1
                
                df, metadata = generator_func(i + 1)
                
                # Save CSV to category folder
                filename = f"{metadata['dataset_id']}.csv"
                filepath = os.path.join(DATASETS_DIR, category, filename)
                df.to_csv(filepath, index=False)
                
                # Update metadata
                metadata["filename"] = f"{category}/{filename}"
                metadata["n_rows"] = len(df)
                metadata["n_columns"] = len(df.columns)
                manifest["datasets"].append(metadata)
                
                print_progress(current, total_datasets, category, f"{metadata['dataset_id']}")
    
    # Save manifest
    manifest_path = os.path.join(DATASETS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    category_counts = {}
    bigbetter_counts = {0: 0, 1: 0}
    indicator_counts = {"present": 0, "absent": 0}
    
    for ds in manifest["datasets"]:
        cat = ds["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        bb = ds.get("expected_bigbetter", 1)
        bigbetter_counts[bb] = bigbetter_counts.get(bb, 0) + 1
        
        if ds.get("expected_indicator_col"):
            indicator_counts["present"] += 1
        else:
            indicator_counts["absent"] += 1
    
    print(f"\nBy Category:")
    for cat, count in category_counts.items():
        print(f"  {cat:20s}: {count:3d} datasets")
    
    print(f"\nBy Expected BigBetter:")
    print(f"  bigbetter=1 (higher is better): {bigbetter_counts[1]:3d}")
    print(f"  bigbetter=0 (lower is better):  {bigbetter_counts[0]:3d}")
    
    print(f"\nBy Indicator Column:")
    print(f"  Indicator present: {indicator_counts['present']:3d}")
    print(f"  Indicator absent:  {indicator_counts['absent']:3d}")
    
    print(f"\nTotal: {total_datasets} datasets")
    print(f"Manifest: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
