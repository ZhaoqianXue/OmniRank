#!/usr/bin/env python3
"""
Generate high-quality test datasets for Function 1: Format Recognition & Standardization.

This script creates diverse test cases to validate:
1. Format Recognition (Pointwise, Pairwise, Multiway)
2. Engine Compatibility Assessment
3. Conditional Standardization Triggers

Dataset Categories:
- Standard cases: Clear examples of each format
- Edge cases: Boundary conditions and ambiguous data
- Standardization triggers: Cases requiring preprocessing
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).parent


def generate_pointwise_datasets():
    """Generate Pointwise format test datasets."""
    output_dir = BASE_DIR / "pointwise"
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # Dataset 1: Standard LLM Benchmark Scores
    # - Clear pointwise format
    # - Column names indicate models
    # - Values are accuracy scores [0, 1]
    # =========================================================================
    n_samples = 100
    models = ["GPT-4", "Claude-3", "Gemini-Pro", "Llama-70B", "Mistral-7B"]
    
    # Generate correlated scores (some models consistently better)
    base_scores = np.random.uniform(0.5, 0.9, n_samples)
    data = {
        "sample_id": [f"benchmark_{i:03d}" for i in range(n_samples)],
        "GPT-4": np.clip(base_scores + np.random.normal(0.05, 0.05, n_samples), 0, 1),
        "Claude-3": np.clip(base_scores + np.random.normal(0.03, 0.06, n_samples), 0, 1),
        "Gemini-Pro": np.clip(base_scores + np.random.normal(0.0, 0.07, n_samples), 0, 1),
        "Llama-70B": np.clip(base_scores + np.random.normal(-0.05, 0.08, n_samples), 0, 1),
        "Mistral-7B": np.clip(base_scores + np.random.normal(-0.10, 0.09, n_samples), 0, 1),
        "task_category": np.random.choice(["reasoning", "coding", "math", "writing"], n_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "standard_llm_benchmark.csv", index=False)
    print(f"Created: {output_dir / 'standard_llm_benchmark.csv'}")
    
    # =========================================================================
    # Dataset 2: Product Ratings (Different Scale)
    # - Values are ratings [1, 5]
    # - Clear higher-is-better semantics
    # =========================================================================
    n_samples = 80
    products = ["Product_A", "Product_B", "Product_C", "Product_D"]
    
    data = {
        "reviewer_id": [f"user_{i:04d}" for i in range(n_samples)],
        "Product_A": np.random.uniform(3.5, 5.0, n_samples).round(1),
        "Product_B": np.random.uniform(2.5, 4.5, n_samples).round(1),
        "Product_C": np.random.uniform(3.0, 4.8, n_samples).round(1),
        "Product_D": np.random.uniform(2.0, 4.0, n_samples).round(1),
        "category": np.random.choice(["electronics", "home", "outdoor"], n_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "product_ratings.csv", index=False)
    print(f"Created: {output_dir / 'product_ratings.csv'}")
    
    # =========================================================================
    # Dataset 3: Response Time Data (Lower is Better)
    # - Column names suggest latency/time
    # - Clear lower-is-better semantics
    # =========================================================================
    n_samples = 60
    
    data = {
        "test_id": [f"latency_test_{i:03d}" for i in range(n_samples)],
        "Server_A_latency_ms": np.random.exponential(50, n_samples) + 10,
        "Server_B_latency_ms": np.random.exponential(40, n_samples) + 15,
        "Server_C_latency_ms": np.random.exponential(60, n_samples) + 8,
        "Server_D_latency_ms": np.random.exponential(45, n_samples) + 12,
        "region": np.random.choice(["US-East", "US-West", "EU", "Asia"], n_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "latency_measurements.csv", index=False)
    print(f"Created: {output_dir / 'latency_measurements.csv'}")
    
    return 3


def generate_pairwise_datasets():
    """Generate Pairwise format test datasets."""
    output_dir = BASE_DIR / "pairwise"
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # Dataset 1: Standard LLM Comparison (with Indicator)
    # - Sparse 0/1 matrix
    # - Exactly 2 non-null values per row
    # - Task indicator column
    # =========================================================================
    n_comparisons = 500
    models = ["GPT-4", "Claude-3", "Gemini-Pro", "Llama-70B", "Mistral-7B", "Qwen-72B"]
    tasks = ["code", "math", "writing", "reasoning"]
    
    # Generate random pairwise comparisons
    rows = []
    for i in range(n_comparisons):
        # Pick two random models
        m1, m2 = np.random.choice(len(models), 2, replace=False)
        # Assign winner/loser based on model "quality" with some noise
        model_quality = [0.9, 0.85, 0.8, 0.7, 0.65, 0.75]
        p_m1_wins = model_quality[m1] / (model_quality[m1] + model_quality[m2])
        winner = m1 if np.random.random() < p_m1_wins else m2
        loser = m2 if winner == m1 else m1
        
        row = {model: np.nan for model in models}
        row[models[winner]] = 1
        row[models[loser]] = 0
        row["Task"] = np.random.choice(tasks)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Reorder columns: Task first, then models
    cols = ["Task"] + models
    df = df[cols]
    df.to_csv(output_dir / "standard_llm_comparison.csv", index=False)
    print(f"Created: {output_dir / 'standard_llm_comparison.csv'}")
    
    # =========================================================================
    # Dataset 2: A/B Test Results (No Indicator)
    # - Pure pairwise comparison
    # - No categorical indicator
    # =========================================================================
    n_comparisons = 300
    variants = ["Control", "Variant_A", "Variant_B", "Variant_C"]
    
    rows = []
    for i in range(n_comparisons):
        m1, m2 = np.random.choice(len(variants), 2, replace=False)
        variant_quality = [0.5, 0.55, 0.6, 0.52]
        p_m1_wins = variant_quality[m1] / (variant_quality[m1] + variant_quality[m2])
        winner = m1 if np.random.random() < p_m1_wins else m2
        loser = m2 if winner == m1 else m1
        
        row = {variant: np.nan for variant in variants}
        row[variants[winner]] = 1
        row[variants[loser]] = 0
        row["comparison_id"] = f"ab_test_{i:04d}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    cols = ["comparison_id"] + variants
    df = df[cols]
    df.to_csv(output_dir / "ab_test_results.csv", index=False)
    print(f"Created: {output_dir / 'ab_test_results.csv'}")
    
    # =========================================================================
    # Dataset 3: Tournament Brackets (Many Items)
    # - 10 teams
    # - Many comparisons
    # =========================================================================
    n_comparisons = 800
    teams = [f"Team_{chr(65+i)}" for i in range(10)]  # Team_A to Team_J
    
    rows = []
    for i in range(n_comparisons):
        t1, t2 = np.random.choice(len(teams), 2, replace=False)
        team_strength = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
        p_t1_wins = team_strength[t1] / (team_strength[t1] + team_strength[t2])
        winner = t1 if np.random.random() < p_t1_wins else t2
        loser = t2 if winner == t1 else t1
        
        row = {team: np.nan for team in teams}
        row[teams[winner]] = 1
        row[teams[loser]] = 0
        row["round"] = np.random.choice(["preliminary", "quarterfinal", "semifinal", "final"])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    cols = ["round"] + teams
    df = df[cols]
    df.to_csv(output_dir / "tournament_brackets.csv", index=False)
    print(f"Created: {output_dir / 'tournament_brackets.csv'}")
    
    return 3


def generate_multiway_datasets():
    """Generate Multiway format test datasets."""
    output_dir = BASE_DIR / "multiway"
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # Dataset 1: Horse Race Results
    # - Classic multiway ranking
    # - Each row is a race with rank positions
    # =========================================================================
    n_races = 100
    horses = ["Thunderbolt", "Lightning", "Storm", "Blaze", "Shadow", "Spirit"]
    
    rows = []
    for i in range(n_races):
        # Generate rankings based on horse "quality" with noise
        horse_quality = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
        scores = [q + np.random.normal(0, 0.2) for q in horse_quality]
        ranks = np.argsort(np.argsort(-np.array(scores))) + 1  # 1-indexed ranks
        
        row = {horse: int(rank) for horse, rank in zip(horses, ranks)}
        row["race_id"] = f"race_{i:03d}"
        row["track_type"] = np.random.choice(["grass", "dirt", "synthetic"])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    cols = ["race_id", "track_type"] + horses
    df = df[cols]
    df.to_csv(output_dir / "horse_race_results.csv", index=False)
    print(f"Created: {output_dir / 'horse_race_results.csv'}")
    
    # =========================================================================
    # Dataset 2: Multi-player Game Rankings
    # - 8 players per game
    # - Position 1 = winner
    # =========================================================================
    n_games = 120
    players = [f"Player_{i}" for i in range(1, 9)]
    
    rows = []
    for i in range(n_games):
        player_skill = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        scores = [s + np.random.normal(0, 0.25) for s in player_skill]
        ranks = np.argsort(np.argsort(-np.array(scores))) + 1
        
        row = {player: int(rank) for player, rank in zip(players, ranks)}
        row["game_id"] = f"game_{i:04d}"
        row["game_mode"] = np.random.choice(["solo", "team", "battle_royale"])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    cols = ["game_id", "game_mode"] + players
    df = df[cols]
    df.to_csv(output_dir / "multiplayer_game_rankings.csv", index=False)
    print(f"Created: {output_dir / 'multiplayer_game_rankings.csv'}")
    
    # =========================================================================
    # Dataset 3: Academic Paper Rankings (Top-k)
    # - Only top 5 positions recorded (partial ranking)
    # - Some positions may have ties (same rank value)
    # =========================================================================
    n_sessions = 50
    papers = [f"Paper_{chr(65+i)}" for i in range(10)]  # Paper_A to Paper_J
    
    rows = []
    for i in range(n_sessions):
        paper_quality = np.random.uniform(0.3, 0.9, len(papers))
        scores = paper_quality + np.random.normal(0, 0.15, len(papers))
        
        # Only record top 5
        top_indices = np.argsort(-scores)[:5]
        
        row = {paper: np.nan for paper in papers}
        for rank, idx in enumerate(top_indices, 1):
            row[papers[idx]] = rank
        
        row["session_id"] = f"session_{i:03d}"
        row["conference"] = np.random.choice(["NeurIPS", "ICML", "ICLR", "ACL"])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    cols = ["session_id", "conference"] + papers
    df = df[cols]
    df.to_csv(output_dir / "paper_rankings_topk.csv", index=False)
    print(f"Created: {output_dir / 'paper_rankings_topk.csv'}")
    
    return 3


def generate_edge_case_datasets():
    """Generate edge case datasets for standardization testing."""
    output_dir = BASE_DIR / "edge_cases"
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # Edge Case 1: Column Names with Spaces (Standardization Trigger)
    # - Should trigger standardization due to R incompatibility
    # =========================================================================
    n_samples = 50
    data = {
        "Sample ID": [f"s_{i:03d}" for i in range(n_samples)],
        "Model Alpha Score": np.random.uniform(0.5, 0.9, n_samples),
        "Model Beta Score": np.random.uniform(0.4, 0.85, n_samples),
        "Model Gamma Score": np.random.uniform(0.45, 0.88, n_samples),
        "Test Category": np.random.choice(["type_a", "type_b"], n_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "columns_with_spaces.csv", index=False)
    print(f"Created: {output_dir / 'columns_with_spaces.csv'}")
    
    # =========================================================================
    # Edge Case 2: Column Names with Special Characters
    # - Contains hyphens, parentheses
    # =========================================================================
    n_samples = 50
    data = {
        "sample-id": [f"test_{i:03d}" for i in range(n_samples)],
        "model(v1)": np.random.uniform(0.5, 0.9, n_samples),
        "model(v2)": np.random.uniform(0.4, 0.85, n_samples),
        "model-v3": np.random.uniform(0.45, 0.88, n_samples),
        "task_type": np.random.choice(["eval_a", "eval_b"], n_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "columns_with_special_chars.csv", index=False)
    print(f"Created: {output_dir / 'columns_with_special_chars.csv'}")
    
    # =========================================================================
    # Edge Case 3: Minimal Data (Boundary Condition)
    # - Just enough for engine to work (2 columns, few rows)
    # =========================================================================
    data = {
        "item_a": [0.8, 0.7, 0.9, 0.6, 0.75],
        "item_b": [0.7, 0.8, 0.85, 0.65, 0.7],
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "minimal_data.csv", index=False)
    print(f"Created: {output_dir / 'minimal_data.csv'}")
    
    # =========================================================================
    # Edge Case 4: Ambiguous Format (Could be Pointwise or Multiway)
    # - Dense matrix with integers 1-5
    # - Could be ratings (pointwise) or partial ranks (multiway)
    # =========================================================================
    n_samples = 40
    data = {
        "case_id": [f"case_{i:03d}" for i in range(n_samples)],
        "Option_1": np.random.randint(1, 6, n_samples),
        "Option_2": np.random.randint(1, 6, n_samples),
        "Option_3": np.random.randint(1, 6, n_samples),
        "Option_4": np.random.randint(1, 6, n_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "ambiguous_format.csv", index=False)
    print(f"Created: {output_dir / 'ambiguous_format.csv'}")
    
    # =========================================================================
    # Edge Case 5: Mixed Numeric Types (Int + Float)
    # =========================================================================
    n_samples = 60
    data = {
        "id": range(n_samples),
        "score_int": np.random.randint(50, 100, n_samples),
        "score_float": np.random.uniform(0.5, 1.0, n_samples),
        "model_a": np.random.uniform(0.6, 0.95, n_samples),
        "model_b": np.random.uniform(0.55, 0.9, n_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "mixed_numeric_types.csv", index=False)
    print(f"Created: {output_dir / 'mixed_numeric_types.csv'}")
    
    # =========================================================================
    # Edge Case 6: Many Non-Numeric Columns
    # - Tests engine's ability to auto-filter
    # =========================================================================
    n_samples = 50
    data = {
        "id": [f"sample_{i:03d}" for i in range(n_samples)],
        "name": [f"Experiment {i}" for i in range(n_samples)],
        "description": ["This is a test description"] * n_samples,
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "subcategory": np.random.choice(["X", "Y", "Z"], n_samples),
        "model_1": np.random.uniform(0.5, 0.9, n_samples),
        "model_2": np.random.uniform(0.45, 0.85, n_samples),
        "model_3": np.random.uniform(0.55, 0.88, n_samples),
        "notes": ["Additional notes here"] * n_samples,
    }
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "many_non_numeric_columns.csv", index=False)
    print(f"Created: {output_dir / 'many_non_numeric_columns.csv'}")
    
    return 6


def main():
    """Generate all test datasets."""
    print("=" * 60)
    print("Generating Test Datasets for Function 1: Format Recognition")
    print("=" * 60)
    
    total = 0
    
    print("\n--- Pointwise Datasets ---")
    total += generate_pointwise_datasets()
    
    print("\n--- Pairwise Datasets ---")
    total += generate_pairwise_datasets()
    
    print("\n--- Multiway Datasets ---")
    total += generate_multiway_datasets()
    
    print("\n--- Edge Case Datasets ---")
    total += generate_edge_case_datasets()
    
    print("\n" + "=" * 60)
    print(f"Total datasets generated: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
