# Experiment: Engine Orchestrator - Two-Step Method Triggering Accuracy

This experiment evaluates the Engine Orchestrator's ability to make correct statistical decisions about when to apply the two-step refinement method (Section 4.2.3 of the paper).

---

## 1. Methodology Reference

This experiment corresponds to **Section 3.3 Engine Orchestrator** of the OmniRank paper, specifically **Function 2: Robust Engine Execution with Dynamic Orchestration**.

### 3.3 Engine Orchestrator (writing.md Reference)

> The Engine Orchestrator is a **deterministic system component** that manages the transition from data schema to statistical computation. It ensures execution reliability through interactive configuration and robust resource management.

### Function 2: Robust Engine Execution with Dynamic Orchestration

**Definition from writing.md:**

> It encapsulates the spectral ranking logic, executing it within isolated processes. Crucially, the orchestrator implements the **Two-Step Spectral Method** logic from Fan et al. (2023).

**Decision Logic:**

| Condition | Threshold | Role |
|-----------|-----------|------|
| **Gatekeeper** | `sparsity_ratio >= 1.0` | MUST pass, otherwise Step 2 is blocked |
| **Trigger A** | `heterogeneity_index > 0.5` | At least ONE trigger needed |
| **Trigger B** | `CI_width / n > 0.2` (20%) | At least ONE trigger needed |

**Final Logic:**
```
IF sparsity_ratio < 1.0:
    DECISION = NO (Data too sparse, Step 2 unstable)
ELSE IF heterogeneity_index > 0.5 OR (CI_width / n) > 0.2:
    DECISION = YES (Refinement beneficial)
ELSE:
    DECISION = NO (Step 1 sufficient)
```

**Theoretical Background (Fan et al., 2023):**

- **Sparsity Gatekeeper**: Based on coupon collector bound (M >= n*log(n)) for consistent estimation
- **Heterogeneity Trigger**: Vanilla spectral method with uniform weights is suboptimal under heterogeneous comparison counts
- **Uncertainty Trigger**: Wide CIs indicate high variance; Step 2's optimal weighting can reduce variance and narrow CIs (Theorem 2, Remark 6)

---

## 2. Experiments Reference

This experiment corresponds to **Section 4.2.3 Two-Step Method Triggering Accuracy** of the OmniRank paper.

### 4.2.3 Two-Step Method Triggering Accuracy (writing.md Reference)

> **Purpose**: Evaluate the Engine Orchestrator's ability to make correct statistical decisions about when to apply the two-step refinement method.
>
> **What to include**:
> - **Test Scenarios**: Datasets with varying levels of heterogeneity and sparsity
> - **Trigger Conditions Evaluated**:
>   - **Sparsity Gatekeeper**: Does the orchestrator correctly skip refinement when sparsity_ratio < 1.0?
>   - **Heterogeneity Trigger**: Does it correctly activate when CV > 0.5?
>   - **Uncertainty Trigger**: Does it correctly activate when CI_width / n > 20%?
> - **Metrics**: Trigger decision accuracy (TPR/TNR), ranking improvement when correctly triggered
> - **Results Table**: Decision accuracy across different data conditions
> - **Discussion**: Demonstrate that dynamic orchestration improves ranking quality

---

## 3. Experimental Design

### 3.1 Test Scenarios

The experiment uses 50 synthetic datasets across 10 test scenarios:

| Scenario | Count | Description | Expected Step 2 |
|----------|-------|-------------|-----------------|
| `sparse_homogeneous` | 5 | Sparse data, low heterogeneity | No (Gatekeeper blocked) |
| `sparse_heterogeneous` | 5 | Sparse data, high heterogeneity | No (Gatekeeper blocked) |
| `sufficient_no_trigger` | 5 | Sufficient data, no triggers | No (Step 1 sufficient) |
| `sufficient_heterogeneity` | 5 | Sufficient data, high heterogeneity | Yes (Trigger A) |
| `sufficient_uncertainty` | 5 | Sufficient data, wide CIs | Yes (Trigger B) |
| `sufficient_both` | 5 | Sufficient data, both triggers | Yes (Both triggers) |
| `boundary_sparsity_below` | 5 | Sparsity just below 1.0 | No (Gatekeeper blocked) |
| `boundary_sparsity_above` | 5 | Sparsity just above 1.0 with trigger | Yes (Trigger A) |
| `large_items_sparse` | 5 | Many items (n=20), sparse | No (Gatekeeper blocked) |
| `large_items_sufficient` | 5 | Many items (n=20), sufficient | Yes (Trigger A) |

### 3.2 Dataset Properties

Each dataset is generated with controlled properties:

- **Sparsity ratio**: M / (n * log(n)), where M = total pairwise comparisons
- **Heterogeneity**: Coefficient of variation in comparison counts per item
- **Score distribution**: Affects CI width through variance in estimates

### 3.3 Evaluation Process

For each dataset:

1. Run `spectral_ranking_step1.R` to compute Step 1 results
2. Extract metadata: `sparsity_ratio`, `heterogeneity_index`, `mean_ci_width_top_5`, `k_methods`
3. Apply decision logic (`should_run_step2` function)
4. Compare predicted decision with expected decision

### 3.4 Metrics

| Metric | Description |
|--------|-------------|
| **Overall Accuracy** | Proportion of correct decisions across all datasets |
| **Precision** | TP / (TP + FP) for Step 2 triggering |
| **Recall** | TP / (TP + FN) for Step 2 triggering |
| **F1 Score** | Harmonic mean of precision and recall |
| **Gatekeeper Accuracy** | Accuracy of blocking sparse data |

---

## 4. Directory Structure

```
experiments/orchestrator_step2/
|-- README.md                    # This file
|-- configs/
|   +-- experiment_config.json   # Experiment configuration
|-- datasets/
|   |-- manifest.json            # Ground truth labels for all datasets
|   |-- sparse_homogeneous/      # Sparse data scenarios
|   |-- sparse_heterogeneous/
|   |-- sufficient_no_trigger/   # Sufficient data scenarios
|   |-- sufficient_heterogeneity/
|   |-- sufficient_uncertainty/
|   |-- sufficient_both/
|   |-- boundary_sparsity_below/ # Boundary cases
|   |-- boundary_sparsity_above/
|   |-- large_items_sparse/      # Large scale tests
|   +-- large_items_sufficient/
|-- generate_datasets.py         # Dataset generation script
|-- run_experiment.py            # Main experiment runner
+-- runs/
    +-- YYYYMMDD_HHMMSS/         # Timestamped run outputs
        |-- results.json         # Full metrics in JSON
        |-- error_analysis.json  # Detailed error breakdown
        |-- table_4_2_3.tex      # LaTeX table for paper
        +-- table_4_2_3.md       # Markdown table
```

---

## 5. Running the Experiment

### 5.1 Generate Datasets

```bash
cd experiments/orchestrator_step2
python generate_datasets.py
```

This creates 50 synthetic datasets across 10 test scenarios.

### 5.2 Run Experiment

```bash
python run_experiment.py
```

**Configuration (in run_experiment.py):**
- `N_RUNS = 1` for quick testing, set to `3+` for publication results

**Requirements:**
- R with `readr`, `dplyr`, `jsonlite` packages installed
- Python with `numpy`, `pandas`, `python-dotenv`

### 5.3 Output Files

Results are saved to `runs/YYYYMMDD_HHMMSS/`:
- `results.json`: Machine-readable full results
- `table_4_2_3.tex`: Copy directly to paper
- `table_4_2_3.md`: Markdown version
- `error_analysis.json`: Misclassified datasets details

---

## 6. Configuration Reference

### experiment_config.json

```json
{
  "experiment_id": "orchestrator_step2",
  "paper_section": "4.2.3 Two-Step Method Triggering Accuracy",
  
  "decision_rules": {
    "gatekeeper": {
      "condition": "sparsity_ratio >= 1.0",
      "threshold": 1.0
    },
    "trigger_heterogeneity": {
      "condition": "heterogeneity_index > 0.5",
      "threshold": 0.5
    },
    "trigger_uncertainty": {
      "condition": "ci_width_ratio > 0.2",
      "threshold": 0.2
    }
  },
  
  "decision_logic": "IF (gatekeeper_pass) AND (heterogeneity_trigger OR uncertainty_trigger) THEN step2_triggered"
}
```

---

## 7. Related Code

| File | Description |
|------|-------------|
| `src/api/core/r_executor.py` | `should_run_step2` function implementation |
| `src/api/agents/orchestrator.py` | Engine Orchestrator implementation |
| `src/spectral_ranking/spectral_ranking_step1.R` | Step 1 R script |

---

## 8. Experimental Results (Run: 20260126_030215)

### 8.1 Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 1.000 +/- 0.000 |
| **Precision** | 1.000 +/- 0.000 |
| **Recall** | 1.000 +/- 0.000 |
| **F1 Score** | 1.000 +/- 0.000 |
| **Gatekeeper Accuracy** | 1.000 +/- 0.000 |

*Results over 3 runs with 50 datasets.*

### 8.2 Performance by Scenario

| Scenario | Expected Step 2 | Accuracy |
|----------|-----------------|----------|
| sparse_homogeneous | No | 100% |
| sparse_heterogeneous | No | 100% |
| sufficient_no_trigger | No | 100% |
| sufficient_heterogeneity | Yes | 100% |
| sufficient_uncertainty | Yes | 100% |
| sufficient_both | Yes | 100% |
| boundary_sparsity_below | No | 100% |
| boundary_sparsity_above | Yes | 100% |
| large_items_sparse | No | 100% |
| large_items_sufficient | Yes | 100% |

### 8.3 Confusion Matrix

```
                 Predicted
              No Step2   Step2
Actual No    [  75         0]
       Yes   [   0        75]
```

### 8.4 Key Findings

1. **Perfect Gatekeeper Performance**: The sparsity check correctly blocked Step 2 for all sparse datasets.
2. **Accurate Trigger Detection**: Both heterogeneity (CV > 0.5) and uncertainty (CI/n > 20%) triggers were correctly identified.
3. **Zero False Positives**: No unnecessary Step 2 executions when data was sufficient but triggers not activated.
4. **Deterministic Behavior**: As expected for rule-based logic, results are perfectly reproducible.

---

## 9. Citation

If you use this experiment in your research, please cite:

```bibtex
@article{omnirank2026,
  title={OmniRank: A Large-Language-Model Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons},
  author={...},
  journal={Journal of the American Statistical Association},
  year={2026}
}
```
