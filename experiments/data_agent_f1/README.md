# Experiment: Data Agent Function 1 - Format Recognition & Standardization

This experiment evaluates the Data Agent's first core function: **Format Recognition & Standardization** (Section 4.2.1 of the paper).

---

## 1. Methodology Reference

This experiment corresponds to **Section 3.2 Data Agent** of the OmniRank paper, specifically **Function 1: Format Recognition & Standardization**.

### 3.2 Data Agent (writing.md Reference)

> The Data Agent acts as the intelligent interface between user data and the spectral engine, performing three critical functions to ensure data readiness and semantic understanding.



### Function 1: Format Recognition & Standardization

**Definition from writing.md:**
> The agent automatically identifies the structure of uploaded data (e.g., Pointwise, Pairwise, Multiway) and performs lightweight standardization to ensure compatibility with the spectral engine (`spectral_ranking_step1.R`). Instead of enforcing a rigid conversion to a single format, it adapts to the input structure, preserving the original data fidelity while ensuring it meets the engine's interface requirements.

**Implementation Details:**

| Task | Description |
|------|-------------|
| **Format Detection** | Classify input data into one of three formats: Pointwise (dense numeric matrix), Pairwise (sparse 0/1/NaN comparison matrix), or Multiway (rank position matrix) |
| **Engine Compatibility Assessment** | Determine if the spectral engine can process the data directly without transformation |
| **Standardization Decision** | Decide whether standardization is needed (none/standardize/reject) based on data structure |

**Data Format Definitions:**

1. **Pointwise Format**
   - Dense numeric matrix where each row is a sample/observation
   - Columns represent items being ranked (e.g., models, products)
   - Cell values are scores/metrics for that item on that sample
   - Example: LLM benchmark scores, product ratings

2. **Pairwise Format**
   - Standard Matrix: Sparse matrix with 1 = winner, 0 = loser, NaN = not compared
   - Winner/Loser Column: Two columns containing item names (requires standardization)
   - Example: Tennis matches, chess games

3. **Multiway Format**
   - Encodes rankings where multiple items compete simultaneously
   - Values are rank positions (1st, 2nd, 3rd, etc.)
   - Example: Horse race results, league standings

### Data Validation Checks

The Data Agent performs validation checks after format recognition to assess data quality:

| Check | Condition | Severity | User-Facing Message |
|-------|-----------|----------|---------------------|
| **Minimum Items** | `len(ranking_items) < 2` | Critical Error | "We need at least 2 items to create a ranking." |
| **Columns Exist** | Referenced columns missing from data | Critical Error | "Some expected columns are missing from your data." |
| **Sample Size** | `len(df) < 10` | Warning | "Your dataset has only X rows. For more reliable rankings, we recommend at least 10 data points." |
| **Sparsity** | $M < n \log n$ | Warning | "Your data has relatively few comparisons between items. The ranking results may have wider uncertainty ranges." |
| **Connectivity** | Comparison graph is disconnected | Warning | "Some items have no comparisons with others. They cannot be ranked relative to the rest." |

**Design Principle:** All user-facing messages are written in plain, non-technical language per the project goal of making spectral ranking accessible to non-statisticians.

**Sparsity Check Details:**

The sparsity threshold $M > n \log n$ comes from spectral ranking theory (Fan et al. 2023):
- $M$ = total number of pairwise comparisons (calculated as $\sum C(k,2)$ where $k$ = non-null values per row)
- $n$ = number of items to rank
- $n \log n$ = theoretical lower bound for consistent estimation (coupon collector problem)

If $M < n \log n$, the spectral estimates may have high variance and rankings may be unstable.

**Connectivity Check Details:**

Uses `networkx` to verify the comparison graph is connected:
1. Build undirected graph $G$ where nodes = ranking items
2. Add edge $(i, j)$ if items $i$ and $j$ are compared in any row
3. Check `nx.is_connected(G)`
4. If disconnected, rankings are only meaningful within each connected component

---

## 2. Experiments Reference

This experiment corresponds to **Section 4.2.1 Format Recognition & Standardization** of the OmniRank paper.

### 4.2.1 Format Recognition & Standardization (writing.md Reference)

> **Purpose**: Assess the Data Agent's ability to automatically identify and handle diverse data formats.
>
> **What to include**:
> - **Test Datasets**: Pointwise, Pairwise, and Multiway formats
> - **Metrics**: Format detection accuracy (precision/recall per format type)
> - **Results**: Overall accuracy percentage, confusion matrix if applicable

**Evaluation Targets:**

| Target | Description | Labels |
|--------|-------------|--------|
| **Format Detection** | Correctly identify data format | pointwise, pairwise, multiway, invalid |
| **Engine Compatibility** | Assess if spectral engine can process directly | compatible, incompatible |
| **Standardization Decision** | Decide required action | none, standardize, reject |

---

## 3. Experimental Design

### 3.1 Dataset Categories

The experiment uses 41 synthetic datasets across 5 difficulty categories:

| Category | Count | Difficulty | Description |
|----------|-------|------------|-------------|
| **Standard** | 9 | Easy | Clean, unambiguous format examples |
| **Ambiguous** | 8 | Medium-Hard | Edge cases where format could be misinterpreted |
| **Transposed** | 4 | Hard | Data with rows/columns swapped from expected structure |
| **Invalid** | 8 | Easy-Medium | Data that should trigger rejection/errors |
| **Realworld** | 12 | Medium | Formats based on actual internet data sources |

### 3.2 Format Distribution

| Format | Count | Examples |
|--------|-------|----------|
| Pointwise | 19 | Benchmark scores, product ratings, survey responses |
| Pairwise | 9 | Tennis matches (winner/loser), chess games (0/1 matrix) |
| Multiway | 9 | Horse races, league standings |
| Invalid | 4 | Single column, all-text data |

### 3.3 Difficulty Distribution

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 15 | Clean formats, should be correctly identified |
| Medium | 14 | Require semantic understanding |
| Hard | 12 | Ambiguous structures, transposed data |

### 3.4 Challenging Test Cases

**Ambiguous Datasets (designed to cause misclassification):**

| Dataset ID | Challenge | Expected Format | Why Difficult |
|------------|-----------|-----------------|---------------|
| `amb_2way_*` | 2-item races | Multiway | Only values 1,2 per row - looks like pairwise |
| `amb_binary_*` | Binary success/failure | Pointwise | All 0/1 values - identical to pairwise encoding |
| `amb_pointwise_sparse_*` | Missing values | Pointwise | Sparsity pattern resembles pairwise |

**Realworld Datasets (require semantic understanding):**

| Dataset ID | Source Format | Challenge |
|------------|---------------|-----------|
| `real_tennis_*` | ATP matches | winner_name/loser_name columns (not standard matrix) |
| `real_chess_*` | ELO games | Player names as columns (contain spaces) |
| `real_benchmark_*` | ML leaderboards | Messy column names with special characters |

### 3.5 Baseline Comparison

The experiment includes a **Rule-Based Baseline** for comparison:

```python
class RuleBasedDetector:
    """Simple heuristic-based format detection."""
    
    def detect_format(self, df):
        # Rule 1: < 2 numeric columns -> invalid
        # Rule 2: > 50% NaN + only 0/1 values -> pairwise
        # Rule 3: Consecutive integers 1..k -> multiway
        # Rule 4: Default -> pointwise
```

### 3.6 Metrics

| Metric | Scope | Description |
|--------|-------|-------------|
| **Overall Accuracy** | All datasets | Proportion correct across all 41 datasets |
| **Per-Format Precision** | Each format | TP / (TP + FP) |
| **Per-Format Recall** | Each format | TP / (TP + FN) |
| **Per-Format F1** | Each format | Harmonic mean of precision and recall |
| **Category Accuracy** | Each category | Accuracy within standard/ambiguous/etc. |

---

## 4. Experimental Results (Run: 20260126_002143)

### 4.1 Overall Performance

| Method | Format Detection Accuracy |
|--------|---------------------------|
| **Data Agent (LLM)** | **85.4%** |
| Baseline (Rule-based) | 75.6% |

**Improvement: +9.8%**

### 4.2 Per-Format Performance (Data Agent)

| Format | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| Pointwise | 0.792 | 1.000 | 0.884 | 19 |
| Pairwise | 0.900 | 1.000 | 0.947 | 9 |
| Multiway | 1.000 | 0.778 | 0.875 | 9 |
| Invalid | 0.000 | 0.000 | 0.000 | 4 |

### 4.3 Per-Category Performance

| Category | Data Agent | Baseline |
|----------|------------|----------|
| Standard | 100.0% | 88.9% |
| Ambiguous | 87.5% | 100.0% |
| Transposed | 100.0% | 100.0% |
| Invalid | 50.0% | 75.0% |
| Realworld | **91.7%** | 41.7% |

**Key Finding:** Data Agent significantly outperforms baseline on Realworld (semantic understanding required).

### 4.4 Error Analysis

**Misclassified Datasets:**

| Ground Truth | Predicted | Datasets |
|--------------|-----------|----------|
| multiway → pairwise | 1 | amb_2way_02 |
| multiway → pointwise | 1 | real_league_01 |
| invalid → pointwise | 4 | inv_single_col_01/02, inv_all_text_01/02 |

**Failure Modes:**

1. **Invalid Detection Failure (4 errors):** LLM does not reliably distinguish "invalid" data that should be rejected
2. **Ambiguous 2-way Multiway (1 error):** 2-item races with values 1,2 still confused with pairwise

### 4.5 Confusion Matrix

```
                 Predicted
              PW    PA    MW    INV
Actual  PW   [19     0     0     0]
        PA   [ 0     9     0     0]
        MW   [ 1     1     7     0]
        INV  [ 4     0     0     0]
```

---

## 5. Directory Structure

```
experiments/data_agent_f1/
├── README.md                    # This file
├── configs/
│   └── experiment_config.json   # Experiment configuration
├── datasets/
│   ├── manifest.json            # Ground truth labels for all datasets
│   ├── standard/                # 9 clean format examples
│   ├── ambiguous/               # 8 edge case datasets
│   ├── transposed/              # 4 transposed structure datasets
│   ├── invalid/                 # 8 invalid data examples
│   └── realworld/               # 12 realistic internet formats
├── generate_datasets.py         # Dataset generation script
├── run_experiment.py            # Main experiment runner
└── runs/
    └── YYYYMMDD_HHMMSS/         # Timestamped run outputs
        ├── results.json         # Full metrics in JSON
        ├── error_analysis.json  # Detailed error breakdown
        ├── table_4_2_1.tex      # LaTeX table for paper
        ├── table_4_2_1.md       # Markdown table
        └── confusion_matrix.png # Visualization
```

---

## 6. Running the Experiment

### 6.1 Generate Datasets

```bash
cd experiments/data_agent_f1
python generate_datasets.py
```

This creates 41 synthetic datasets across 5 categories with ground truth labels.

### 6.2 Run Experiment

```bash
python run_experiment.py
```

**Configuration (in run_experiment.py):**
- `N_RUNS = 1` for quick testing, set to `5+` for publication results

### 6.3 Output Files

Results are saved to `runs/YYYYMMDD_HHMMSS/`:
- `results.json`: Machine-readable full results
- `table_4_2_1.tex`: Copy directly to paper
- `confusion_matrix.png`: Visualization for paper figure

---

## 7. Configuration Reference

### experiment_config.json

```json
{
  "experiment_id": "data_agent_f1",
  "paper_section": "4.2.1 Format Recognition & Standardization",
  
  "evaluation_targets": {
    "format_detection": {
      "labels": ["pointwise", "pairwise", "multiway", "invalid"]
    },
    "engine_compatibility": {
      "labels": ["compatible", "incompatible"]
    },
    "standardization_decision": {
      "labels": ["none", "standardize", "reject"]
    }
  },
  
  "dataset_categories": {
    "standard": {"expected_accuracy": 0.95},
    "ambiguous": {"expected_accuracy": 0.70},
    "transposed": {"expected_accuracy": 0.60},
    "invalid": {"expected_accuracy": 0.85},
    "realworld": {"expected_accuracy": 0.80}
  }
}
```

---

## 8. Related Code

| File | Description |
|------|-------------|
| `src/api/agents/data_agent.py` | Data Agent implementation |
| `src/api/core/schemas.py` | InferredSchema, DataFormat definitions |

### Data Agent Knowledge Base (System Prompt)

The Data Agent uses a structured system prompt with embedded domain knowledge:

```
# Role
You are a Data Schema Analyst for OmniRank...

## Data Formats
### Pointwise Format
- Dense numeric matrix...

### Pairwise Format
**Standard Matrix Format:**
- Sparse matrix with 0/1/NaN...

**Winner/Loser Column Format (REQUIRES STANDARDIZATION):**
- Two columns: winner_name/loser_name, white/black...
- CRITICAL: This format REQUIRES standardization=true
```

---

## 9. Citation

If you use this experiment in your research, please cite:

```bibtex
@article{omnirank2026,
  title={OmniRank: A Large-Language-Model Agent Platform for Statistically Rigorous Ranking Inference from Arbitrary Multiway Comparisons},
  author={...},
  journal={...},
  year={2026}
}
```
