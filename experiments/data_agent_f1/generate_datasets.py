#!/usr/bin/env python3
"""
Dataset Generation for Experiment 4.2.1: Format Recognition & Standardization

Generates realistic test datasets across five difficulty categories:
- Standard: Clean, unambiguous format examples
- Ambiguous: Edge cases where format could be misinterpreted
- Transposed: Data with rows/columns swapped
- Invalid: Data that should trigger rejection/errors
- Real-world: Formats based on actual internet data sources

Each dataset has ground truth labels for:
- expected_format: pointwise|pairwise|multiway|invalid
- expected_engine_compatible: whether spectral engine can process directly
- expected_standardization_action: none|standardize|reject
- difficulty: easy|medium|hard

Paper Section: 4.2.1 Format Recognition & Standardization
"""

import os
import io
import json
import random
import string
import numpy as np
import pandas as pd
import urllib.request
from datetime import datetime
from typing import Optional

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
    print(f"  [{current:2d}/{total}] {pct:5.1f}% | [{category:10s}] {message}")


# =============================================================================
# Category A: Standard Datasets (Clean, Unambiguous)
# =============================================================================

def generate_standard_pointwise(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Standard pointwise: Dense numeric matrix, clear column names.
    Difficulty: EASY - should be correctly identified.
    """
    n_items = random.randint(4, 12)
    n_samples = random.randint(30, 200)
    
    # Use realistic item names
    item_name_patterns = [
        [f"Model_{chr(65+i)}" for i in range(n_items)],
        [f"product_{i+1}" for i in range(n_items)],
        [f"algorithm_{i+1}" for i in range(n_items)],
        [f"system_{chr(65+i)}" for i in range(n_items)],
    ]
    item_names = random.choice(item_name_patterns)
    
    # Generate scores with realistic distribution
    true_scores = np.linspace(0.8, 0.3, n_items) + np.random.normal(0, 0.05, n_items)
    
    data = {}
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.1, n_samples)
        data[item] = np.clip(scores, 0, 1)
    
    # Optionally add sample ID column (non-numeric metadata)
    if random.random() > 0.3:
        data["sample_id"] = [f"s{i+1:04d}" for i in range(n_samples)]
    
    df = pd.DataFrame(data)
    
    # Shuffle column order randomly
    cols = list(df.columns)
    random.shuffle(cols)
    df = df[cols]
    
    return df, {
        "dataset_id": f"std_pointwise_{dataset_id:02d}",
        "category": "standard",
        "expected_format": "pointwise",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "easy",
        "n_items": n_items,
        "n_samples": n_samples,
        "description": "Clean dense numeric matrix with item scores",
    }


def generate_standard_pairwise(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Standard pairwise: Sparse 0/1/NaN matrix, clear pairwise structure.
    Difficulty: EASY - should be correctly identified.
    """
    n_items = random.randint(4, 10)
    n_comparisons = random.randint(50, 300)
    
    item_name_patterns = [
        [f"model_{chr(65+i)}" for i in range(n_items)],
        [f"player_{i+1}" for i in range(n_items)],
        [f"team_{chr(65+i)}" for i in range(n_items)],
    ]
    item_names = random.choice(item_name_patterns)
    
    # True skill levels for Bradley-Terry simulation
    true_skills = np.linspace(2.0, 0.5, n_items)
    
    rows = []
    for _ in range(n_comparisons):
        i, j = random.sample(range(n_items), 2)
        prob_i_wins = 1 / (1 + np.exp(true_skills[j] - true_skills[i]))
        winner_idx = i if random.random() < prob_i_wins else j
        loser_idx = j if winner_idx == i else i
        
        row = {item: np.nan for item in item_names}
        row[item_names[winner_idx]] = 1
        row[item_names[loser_idx]] = 0
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Optionally add indicator column
    if random.random() > 0.4:
        indicators = ["task_A", "task_B", "task_C", "task_D"][:random.randint(2, 4)]
        df.insert(0, "task_type", [random.choice(indicators) for _ in range(n_comparisons)])
    
    return df, {
        "dataset_id": f"std_pairwise_{dataset_id:02d}",
        "category": "standard",
        "expected_format": "pairwise",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "easy",
        "n_items": n_items,
        "n_comparisons": n_comparisons,
        "description": "Clean sparse 0/1/NaN pairwise comparison matrix",
    }


def generate_standard_multiway(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Standard multiway: Rank positions (1, 2, 3...) per row.
    Difficulty: EASY - should be correctly identified.
    """
    n_items = random.randint(4, 8)
    n_races = random.randint(20, 100)
    
    item_name_patterns = [
        [f"horse_{i+1}" for i in range(n_items)],
        [f"racer_{chr(65+i)}" for i in range(n_items)],
        [f"contestant_{i+1}" for i in range(n_items)],
    ]
    item_names = random.choice(item_name_patterns)
    
    true_skills = np.linspace(2.0, 0.5, n_items)
    
    rows = []
    for race_id in range(n_races):
        race_skills = true_skills + np.random.normal(0, 0.5, n_items)
        ranks = np.argsort(-race_skills) + 1  # 1-indexed ranks
        
        row = {}
        if random.random() > 0.3:
            row["race_id"] = f"race_{race_id+1:03d}"
        for idx, item in enumerate(item_names):
            row[item] = int(ranks[idx])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df, {
        "dataset_id": f"std_multiway_{dataset_id:02d}",
        "category": "standard",
        "expected_format": "multiway",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "easy",
        "n_items": n_items,
        "n_races": n_races,
        "description": "Clean rank position matrix (1st, 2nd, etc.)",
    }


# =============================================================================
# Category B: Ambiguous Datasets (Edge Cases)
# =============================================================================

def generate_ambiguous_pointwise_sparse(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Pointwise with some missing values - could look like pairwise.
    Difficulty: MEDIUM - has some NaN values but is still pointwise.
    """
    n_items = random.randint(5, 10)
    n_samples = random.randint(40, 150)
    
    item_names = [f"system_{chr(65+i)}" for i in range(n_items)]
    true_scores = np.linspace(0.85, 0.35, n_items)
    
    data = {}
    for idx, item in enumerate(item_names):
        scores = true_scores[idx] + np.random.normal(0, 0.12, n_samples)
        scores = np.clip(scores, 0, 1)
        # Introduce some missing values (15-30% sparsity)
        mask = np.random.random(n_samples) < random.uniform(0.15, 0.30)
        scores[mask] = np.nan
        data[item] = scores
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"amb_pointwise_sparse_{dataset_id:02d}",
        "category": "ambiguous",
        "expected_format": "pointwise",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "medium",
        "sparsity_percent": random.uniform(15, 30),
        "description": "Pointwise with missing values - could be confused with pairwise",
    }


def generate_ambiguous_small_multiway(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Multiway with only 2 items per race - looks like pairwise but has rank values.
    Difficulty: HARD - only 2 items with values 1,2 per row.
    """
    n_races = random.randint(50, 200)
    
    item_names = [f"player_{chr(65+i)}" for i in range(2)]
    true_skills = [1.5, 0.8]
    
    rows = []
    for race_id in range(n_races):
        race_skills = [s + np.random.normal(0, 0.3) for s in true_skills]
        if race_skills[0] > race_skills[1]:
            ranks = [1, 2]
        else:
            ranks = [2, 1]
        
        row = {"match_id": f"m{race_id+1:04d}"}
        for idx, item in enumerate(item_names):
            row[item] = ranks[idx]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df, {
        "dataset_id": f"amb_2way_{dataset_id:02d}",
        "category": "ambiguous",
        "expected_format": "multiway",  # Technically multiway with ranks 1,2
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "hard",
        "note": "2-item multiway with rank values 1,2 - could be confused with pairwise",
        "description": "Two-item races with rank positions (looks like pairwise)",
    }


def generate_ambiguous_mixed_scale(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Pointwise with different value scales per column (percentages vs decimals).
    Difficulty: MEDIUM - values have different scales.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(30, 100)
    
    item_names = [f"model_{chr(65+i)}" for i in range(n_items)]
    
    data = {}
    for idx, item in enumerate(item_names):
        base_score = 0.7 - idx * 0.08
        # Alternate between 0-1 scale and 0-100 scale
        if idx % 2 == 0:
            scores = base_score + np.random.normal(0, 0.1, n_samples)
            scores = np.clip(scores, 0, 1)
        else:
            scores = (base_score * 100) + np.random.normal(0, 10, n_samples)
            scores = np.clip(scores, 0, 100)
        data[item] = scores
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"amb_mixed_scale_{dataset_id:02d}",
        "category": "ambiguous",
        "expected_format": "pointwise",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "medium",
        "description": "Pointwise with mixed scales (0-1 and 0-100)",
    }


def generate_ambiguous_binary_pointwise(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Pointwise with binary values (success/failure counts) - looks like pairwise.
    Difficulty: HARD - all values are 0/1 but it's actually pointwise.
    """
    n_items = random.randint(5, 10)
    n_samples = random.randint(60, 200)
    
    item_names = [f"method_{chr(65+i)}" for i in range(n_items)]
    true_success_rates = np.linspace(0.9, 0.4, n_items)
    
    data = {}
    for idx, item in enumerate(item_names):
        successes = np.random.random(n_samples) < true_success_rates[idx]
        data[item] = successes.astype(int)
    
    df = pd.DataFrame(data)
    df.insert(0, "trial_id", [f"t{i+1:04d}" for i in range(n_samples)])
    
    return df, {
        "dataset_id": f"amb_binary_{dataset_id:02d}",
        "category": "ambiguous",
        "expected_format": "pointwise",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "hard",
        "description": "Binary success/failure pointwise - could be confused with pairwise",
    }


# =============================================================================
# Category C: Transposed Datasets
# =============================================================================

def generate_transposed_pointwise(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Pointwise with items as rows and samples as columns.
    This is a transposed structure - items should be columns.
    Difficulty: HARD - requires recognizing transposition.
    """
    n_items = random.randint(4, 8)
    n_samples = random.randint(10, 30)  # Fewer samples for transposed
    
    item_names = [f"model_{chr(65+i)}" for i in range(n_items)]
    sample_names = [f"sample_{i+1:02d}" for i in range(n_samples)]
    
    true_scores = np.linspace(0.8, 0.4, n_items)
    
    data = {}
    data["item"] = item_names
    for sample_idx, sample in enumerate(sample_names):
        scores = true_scores + np.random.normal(0, 0.1, n_items)
        data[sample] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"trans_pointwise_{dataset_id:02d}",
        "category": "transposed",
        "expected_format": "pointwise",  # After transposition
        "expected_engine_compatible": True,  # Can process as-is
        "expected_standardization_action": "none",  # Engine may handle it
        "difficulty": "hard",
        "actual_structure": "items_as_rows",
        "description": "Transposed pointwise - items as rows, samples as columns",
    }


def generate_transposed_multiway(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Multiway with races as columns and items as rows.
    Difficulty: HARD - requires recognizing transposition.
    """
    n_items = random.randint(4, 6)
    n_races = random.randint(15, 40)
    
    item_names = [f"horse_{i+1}" for i in range(n_items)]
    race_names = [f"race_{i+1:02d}" for i in range(n_races)]
    
    true_skills = np.linspace(2.0, 0.5, n_items)
    
    data = {"horse_name": item_names}
    for race_idx, race in enumerate(race_names):
        race_skills = true_skills + np.random.normal(0, 0.5, n_items)
        ranks = np.argsort(-race_skills) + 1
        data[race] = [int(r) for r in ranks]
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"trans_multiway_{dataset_id:02d}",
        "category": "transposed",
        "expected_format": "multiway",  # After transposition
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "hard",
        "actual_structure": "items_as_rows_races_as_columns",
        "description": "Transposed multiway - items as rows, races as columns",
    }


# =============================================================================
# Category D: Invalid Datasets (Should Reject)
# =============================================================================

def generate_invalid_single_column(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Only one numeric column - cannot perform ranking comparison.
    Difficulty: EASY to detect as invalid.
    """
    n_samples = random.randint(50, 200)
    
    df = pd.DataFrame({
        "id": [f"sample_{i+1}" for i in range(n_samples)],
        "score": np.random.normal(0.7, 0.15, n_samples),
    })
    
    return df, {
        "dataset_id": f"inv_single_col_{dataset_id:02d}",
        "category": "invalid",
        "expected_format": "invalid",
        "expected_engine_compatible": False,
        "expected_standardization_action": "reject",
        "difficulty": "easy",
        "failure_reason": "insufficient_columns",
        "description": "Only one numeric column - cannot compare items",
    }


def generate_invalid_all_text(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    All columns are text/categorical - no numeric data.
    Difficulty: EASY to detect as invalid.
    """
    n_rows = random.randint(30, 100)
    categories = ["excellent", "good", "fair", "poor"]
    
    df = pd.DataFrame({
        "item_A": [random.choice(categories) for _ in range(n_rows)],
        "item_B": [random.choice(categories) for _ in range(n_rows)],
        "item_C": [random.choice(categories) for _ in range(n_rows)],
        "comment": ["Some text comment" for _ in range(n_rows)],
    })
    
    return df, {
        "dataset_id": f"inv_all_text_{dataset_id:02d}",
        "category": "invalid",
        "expected_format": "invalid",
        "expected_engine_compatible": False,
        "expected_standardization_action": "reject",
        "difficulty": "easy",
        "failure_reason": "no_numeric_data",
        "description": "All columns are categorical text - no numeric scores",
    }


def generate_invalid_empty_or_tiny(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Too few rows or mostly empty data.
    Difficulty: MEDIUM - has structure but insufficient data.
    """
    n_items = random.randint(3, 6)
    n_rows = random.randint(1, 3)  # Very few rows
    
    item_names = [f"item_{chr(65+i)}" for i in range(n_items)]
    
    data = {}
    for item in item_names:
        data[item] = np.random.random(n_rows)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"inv_tiny_{dataset_id:02d}",
        "category": "invalid",
        "expected_format": "pointwise",  # Structure is recognizable
        "expected_engine_compatible": False,
        "expected_standardization_action": "reject",
        "difficulty": "medium",
        "failure_reason": "insufficient_data",
        "description": "Too few rows for meaningful ranking",
    }


def generate_invalid_disconnected(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Pairwise comparisons forming disconnected components.
    Difficulty: MEDIUM - valid structure but disconnected graph.
    """
    # Create two isolated groups of items
    group_a = ["player_1", "player_2", "player_3"]
    group_b = ["player_4", "player_5", "player_6"]
    n_comparisons = random.randint(30, 60)
    
    rows = []
    for _ in range(n_comparisons // 2):
        # Group A comparisons only
        i, j = random.sample(group_a, 2)
        row = {item: np.nan for item in group_a + group_b}
        row[i] = 1
        row[j] = 0
        rows.append(row)
    
    for _ in range(n_comparisons // 2):
        # Group B comparisons only - no cross-group
        i, j = random.sample(group_b, 2)
        row = {item: np.nan for item in group_a + group_b}
        row[i] = 1
        row[j] = 0
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df, {
        "dataset_id": f"inv_disconnected_{dataset_id:02d}",
        "category": "invalid",
        "expected_format": "pairwise",
        "expected_engine_compatible": False,  # Disconnected graph issue
        "expected_standardization_action": "reject",
        "difficulty": "medium",
        "failure_reason": "disconnected_graph",
        "description": "Pairwise with disconnected comparison graph",
    }


# =============================================================================
# Category E: Real-World Datasets (Based on Internet Formats)
# =============================================================================

def generate_realworld_tennis_matches(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Tennis match format based on JeffSackmann ATP data structure.
    Real-world pairwise comparison format with winner/loser columns.
    """
    n_matches = random.randint(100, 400)
    
    players = [
        "Djokovic", "Nadal", "Federer", "Murray", "Thiem",
        "Zverev", "Medvedev", "Tsitsipas", "Rublev", "Berrettini"
    ][:random.randint(6, 10)]
    
    # Skill levels for simulation
    skills = {p: 2.0 - 0.2 * i for i, p in enumerate(players)}
    
    rows = []
    for match_id in range(n_matches):
        p1, p2 = random.sample(players, 2)
        prob_p1_wins = 1 / (1 + np.exp(skills[p2] - skills[p1]))
        
        if random.random() < prob_p1_wins:
            winner, loser = p1, p2
        else:
            winner, loser = p2, p1
        
        rows.append({
            "match_num": match_id + 1,
            "tourney_name": random.choice(["US Open", "Wimbledon", "French Open", "Australian Open"]),
            "surface": random.choice(["Hard", "Clay", "Grass"]),
            "winner_name": winner,
            "loser_name": loser,
            "score": f"{random.randint(6,7)}-{random.randint(3,4)} {random.randint(6,7)}-{random.randint(4,5)}",
        })
    
    df = pd.DataFrame(rows)
    
    return df, {
        "dataset_id": f"real_tennis_{dataset_id:02d}",
        "category": "realworld",
        "expected_format": "pairwise",
        "expected_engine_compatible": False,  # winner/loser format needs transformation
        "expected_standardization_action": "standardize",
        "difficulty": "hard",
        "source_format": "ATP_tennis_matches",
        "standardization_reason": "winner_loser_column_format_requires_transformation",
        "description": "Tennis match results - winner/loser format (based on ATP data)",
    }


def generate_realworld_product_ratings(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    E-commerce product ratings format (like Amazon, Kaggle datasets).
    Real-world pointwise format with messy column names and mixed types.
    """
    n_products = random.randint(5, 15)
    n_reviews = random.randint(50, 200)
    
    # Messy real-world column names
    products = [f"Product {i+1} (SKU_{random.randint(1000, 9999)})" for i in range(n_products)]
    
    data = {}
    for idx, product in enumerate(products):
        base_rating = 4.5 - idx * 0.3
        ratings = base_rating + np.random.normal(0, 0.8, n_reviews)
        data[product] = np.clip(ratings, 1, 5)
    
    df = pd.DataFrame(data)
    df.insert(0, "reviewer_id", [f"user_{random.randint(10000, 99999)}" for _ in range(n_reviews)])
    df.insert(1, "review_date", pd.date_range("2023-01-01", periods=n_reviews, freq="D").astype(str).tolist())
    
    return df, {
        "dataset_id": f"real_product_{dataset_id:02d}",
        "category": "realworld",
        "expected_format": "pointwise",
        "expected_engine_compatible": False,  # Column names have spaces/special chars
        "expected_standardization_action": "standardize",
        "difficulty": "medium",
        "source_format": "ecommerce_ratings",
        "standardization_reason": "column_names_contain_special_characters",
        "description": "Product ratings with messy column names (needs standardization)",
    }


def generate_realworld_sports_league(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Sports league standings format (like FIFA, EPL).
    Real-world multiway format with additional statistics.
    """
    n_teams = random.randint(8, 20)
    n_matchdays = random.randint(10, 38)
    
    teams = [
        "Manchester United", "Liverpool", "Chelsea", "Arsenal",
        "Manchester City", "Tottenham", "Leicester", "West Ham",
        "Everton", "Leeds", "Aston Villa", "Newcastle",
        "Wolves", "Crystal Palace", "Southampton", "Brighton"
    ][:n_teams]
    
    true_skills = {t: 2.0 - 0.1 * i for i, t in enumerate(teams)}
    
    # Generate cumulative points (multiway ranking per matchday)
    points = {t: 0 for t in teams}
    
    rows = []
    for matchday in range(1, n_matchdays + 1):
        # Simulate matchday results
        for t in teams:
            points[t] += random.choices([3, 1, 0], weights=[0.4, 0.3, 0.3])[0]
        
        # Create ranking for this matchday
        sorted_teams = sorted(teams, key=lambda x: (points[x], random.random()), reverse=True)
        
        row = {"matchday": matchday}
        for idx, team in enumerate(teams):
            rank = sorted_teams.index(team) + 1
            row[team] = rank
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df, {
        "dataset_id": f"real_league_{dataset_id:02d}",
        "category": "realworld",
        "expected_format": "multiway",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "easy",
        "source_format": "sports_league_standings",
        "description": "Sports league rankings per matchday",
    }


def generate_realworld_benchmark_messy(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    ML benchmark format with messy headers (spaces, units in column names).
    Requires standardization for R engine compatibility.
    """
    n_models = random.randint(5, 12)
    n_datasets = random.randint(20, 80)
    
    # Messy column names with spaces and special characters
    model_names = [
        "GPT-4 (v1.0)", "Claude 3.5", "LLaMA 2-70B", "Gemini Pro",
        "Mixtral 8x7B", "Yi-34B-Chat", "Qwen 72B", "DeepSeek V2",
        "Palm 2 (L)", "Falcon 40B", "BLOOM 176B", "Vicuna-13B"
    ][:n_models]
    
    data = {}
    for idx, model in enumerate(model_names):
        base_score = 0.85 - idx * 0.05
        scores = base_score + np.random.normal(0, 0.1, n_datasets)
        data[model] = np.clip(scores, 0, 1)
    
    df = pd.DataFrame(data)
    df.insert(0, "Dataset Name", [f"benchmark_{i+1}" for i in range(n_datasets)])
    df.insert(1, "Task Category", random.choices(["QA", "Reasoning", "Math", "Code"], k=n_datasets))
    
    return df, {
        "dataset_id": f"real_benchmark_{dataset_id:02d}",
        "category": "realworld",
        "expected_format": "pointwise",
        "expected_engine_compatible": False,  # Column names have special chars
        "expected_standardization_action": "standardize",
        "difficulty": "medium",
        "source_format": "ml_benchmark_results",
        "standardization_reason": "column_names_contain_special_characters",
        "description": "ML benchmark scores with messy headers (needs standardization)",
    }


def generate_realworld_survey_likert(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Survey data with Likert scale responses (1-5 or 1-7).
    Real-world pointwise format from survey platforms.
    """
    n_items = random.randint(4, 10)
    n_respondents = random.randint(50, 300)
    
    scale_max = random.choice([5, 7, 10])
    item_names = [f"Q{i+1}_satisfaction" for i in range(n_items)]
    
    true_means = np.linspace(scale_max * 0.8, scale_max * 0.4, n_items)
    
    data = {}
    data["respondent_id"] = [f"R{i+1:05d}" for i in range(n_respondents)]
    data["age_group"] = random.choices(["18-25", "26-35", "36-45", "46-55", "55+"], k=n_respondents)
    
    for idx, item in enumerate(item_names):
        scores = true_means[idx] + np.random.normal(0, scale_max * 0.15, n_respondents)
        data[item] = np.clip(np.round(scores), 1, scale_max).astype(int)
    
    df = pd.DataFrame(data)
    
    return df, {
        "dataset_id": f"real_survey_{dataset_id:02d}",
        "category": "realworld",
        "expected_format": "pointwise",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "easy",
        "source_format": "survey_likert_scale",
        "scale_range": f"1-{scale_max}",
        "description": f"Survey responses on Likert scale (1-{scale_max})",
    }


def generate_realworld_chess_elo(dataset_id: int) -> tuple[pd.DataFrame, dict]:
    """
    Chess match data with ELO-style format (winner/loser/draw).
    Real-world pairwise format with additional game metadata.
    """
    n_games = random.randint(100, 500)
    
    players = [
        "Magnus Carlsen", "Fabiano Caruana", "Ding Liren", "Ian Nepomniachtchi",
        "Alireza Firouzja", "Wesley So", "Anish Giri", "Levon Aronian",
        "Maxime Vachier-Lagrave", "Hikaru Nakamura"
    ][:random.randint(6, 10)]
    
    skills = {p: 2800 - i * 30 for i, p in enumerate(players)}
    
    rows = []
    for game_id in range(n_games):
        white, black = random.sample(players, 2)
        
        # Calculate expected scores
        exp_white = 1 / (1 + 10 ** ((skills[black] - skills[white]) / 400))
        
        result = random.random()
        if result < exp_white * 0.85:  # White wins
            white_score, black_score = 1.0, 0.0
        elif result < exp_white * 0.85 + 0.3:  # Draw
            white_score, black_score = 0.5, 0.5
        else:  # Black wins
            white_score, black_score = 0.0, 1.0
        
        row = {p: np.nan for p in players}
        row["game_id"] = game_id + 1
        row["event"] = random.choice(["World Championship", "Candidates", "Norway Chess", "Tata Steel"])
        row["white"] = white
        row["black"] = black
        row[white] = white_score
        row[black] = black_score
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["game_id", "event", "white", "black"] + players
    df = df[cols]
    
    return df, {
        "dataset_id": f"real_chess_{dataset_id:02d}",
        "category": "realworld",
        "expected_format": "pairwise",
        "expected_engine_compatible": True,
        "expected_standardization_action": "none",
        "difficulty": "medium",
        "source_format": "chess_game_results",
        "description": "Chess match results with ELO-style scoring (includes draws)",
    }


# =============================================================================
# Dataset Generation Dispatcher
# =============================================================================

GENERATORS = {
    "standard": [
        (generate_standard_pointwise, 3),
        (generate_standard_pairwise, 3),
        (generate_standard_multiway, 3),
    ],
    "ambiguous": [
        (generate_ambiguous_pointwise_sparse, 2),
        (generate_ambiguous_small_multiway, 2),
        (generate_ambiguous_mixed_scale, 2),
        (generate_ambiguous_binary_pointwise, 2),
    ],
    "transposed": [
        (generate_transposed_pointwise, 2),
        (generate_transposed_multiway, 2),
    ],
    "invalid": [
        (generate_invalid_single_column, 2),
        (generate_invalid_all_text, 2),
        (generate_invalid_empty_or_tiny, 2),
        (generate_invalid_disconnected, 2),
    ],
    "realworld": [
        (generate_realworld_tennis_matches, 2),
        (generate_realworld_product_ratings, 2),
        (generate_realworld_sports_league, 2),
        (generate_realworld_benchmark_messy, 2),
        (generate_realworld_survey_likert, 2),
        (generate_realworld_chess_elo, 2),
    ],
}


def main():
    print("=" * 70)
    print("Experiment 4.2.1: Format Recognition & Standardization")
    print("Dataset Generation - Realistic & Challenging Test Cases")
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
        "experiment_id": CONFIG.get("experiment_id", "data_agent_f1"),
        "paper_section": CONFIG.get("paper_section", "4.2.1"),
        "generated_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_datasets": total_datasets,
        "categories": list(GENERATORS.keys()),
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
    format_counts = {}
    difficulty_counts = {}
    
    for ds in manifest["datasets"]:
        cat = ds["category"]
        fmt = ds["expected_format"]
        diff = ds["difficulty"]
        
        category_counts[cat] = category_counts.get(cat, 0) + 1
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print(f"\nBy Category:")
    for cat, count in category_counts.items():
        print(f"  {cat:12s}: {count:3d} datasets")
    
    print(f"\nBy Expected Format:")
    for fmt, count in format_counts.items():
        print(f"  {fmt:12s}: {count:3d} datasets")
    
    print(f"\nBy Difficulty:")
    for diff, count in difficulty_counts.items():
        print(f"  {diff:8s}: {count:3d} datasets")
    
    print(f"\nTotal: {total_datasets} datasets")
    print(f"Manifest: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
