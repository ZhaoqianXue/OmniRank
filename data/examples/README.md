# Example Data

This directory contains example datasets for testing and demonstrating OmniRank's spectral ranking capabilities.

## Datasets

### `example_data_pointwise.csv`

**Format**: Pointwise scores (continuous values per item)

**Description**: 165 samples, each with scores for 6 models. Higher scores indicate better performance.

**Columns**:
- `sample_id`: Unique identifier for each sample
- `model_1` through `model_6`: Performance scores for each model (float, 0-1 range)
- `description`: Text description of the sample

**Usage**:
```bash
Rscript src/spectral_engine/ranking_cli.R \
    --csv data/examples/example_data_pointwise.csv \
    --bigbetter 1 \
    --B 2000 \
    --seed 42 \
    --out results/
```

### `example_data_pairwise.csv`

**Format**: Pairwise comparison outcomes (binary win/loss indicators)

**Description**: 3001 pairwise comparisons across 6 models (Your Model, ChatGPT, Claude, Gemini, Llama, Qwen) on different task types (code, math, writing).

**Columns**:
- `Task`: Task category (code, math, writing)
- `Your Model`, `ChatGPT`, `Claude`, `Gemini`, `Llama`, `Qwen`: Binary indicators where `1` = winner, `0` = loser

**Format Details**: Each row represents a single pairwise comparison. In each row:
- Exactly two columns have values (1 and 0)
- The column with value `1` is the winner
- The column with value `0` is the loser
- Other columns are empty

**Usage**:
```bash
Rscript src/spectral_engine/ranking_cli.R \
    --csv data/examples/example_data_pairwise.csv \
    --bigbetter 0 \
    --B 2000 \
    --seed 42 \
    --out results/
```

### `example_data_pairwise_long.csv` (Conceptual)

**Format**: Pairwise Long format (`Winner` and `Loser` columns)

**Description**: Each row lists exactly one winner and one loser by name. This format is often easier to export from human-preference tools.

**Columns**:
- `Winner`: Name of the item that won.
- `Loser`: Name of the item that lost.
- (Optional) metadata columns like `Task`.

**Usage**:
The system automatically detects the `Winner` and `Loser` headers and processes them using the spectral engine.

## Data Format Notes

### Pointwise vs Pairwise

| Format | Description | `--bigbetter` Setting |
|--------|-------------|----------------------|
| Pointwise | Direct scores for each item | `1` if higher = better |
| Pairwise (Wide) | Binary win/loss outcomes in columns | `0` |
| Pairwise (Long) | Rows with `Winner` and `Loser` names | `1` (ignored by engine) |

## Validation & Debugging Datasets

These files are used to test the Data Agent's validation logic:

- `validation_sparsity_warning.csv`: Contains 35 items but only 30 comparisons. Used to trigger the **Sparsity Warning** and **Connectivity Warning**.
- `validation_critical_error.csv`: Contains irrelevant data (support tickets) without ranking columns. Used to trigger a **Critical Error** (Structural Failure) to block analysis.

### Expected Output

Both datasets will produce:
- `ranking_results.json`: Structured results with confidence intervals
- `ranking_results.csv`: Tabular results for easy viewing

The output includes:
- Estimated preference scores (θ̂)
- Rank estimates
- 95% confidence intervals (two-sided and left-sided)
- Runtime metadata

## Creating Your Own Data

### For Pointwise Data

Create a CSV where:
1. Each row is one observation/sample
2. Each column (except metadata columns) represents one item to be ranked
3. Values are numeric scores

The script automatically:
- Drops non-numeric columns
- Drops columns named `case_num`, `model`, `description`
- Generates pairwise comparisons from all pairs

### For Pairwise Data

Create a CSV where:
1. Each row is one pairwise comparison
2. Item columns contain `1` (winner), `0` (loser), or empty (not in this comparison)
3. Task/category columns can be included for context
