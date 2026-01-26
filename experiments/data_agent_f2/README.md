# Experiment 4.2.2: Semantic Schema Inference

This experiment evaluates the Data Agent's **Function 2: Semantic Schema Inference** capabilities.

## Paper Section

This experiment corresponds to **Section 4.2.2** in `docs/project/writing.md`.

## Evaluation Targets

| Target | Description | Metric |
|--------|-------------|--------|
| **BigBetter Direction** | Inference of preference direction (1=higher is better, 0=lower is better) | Accuracy |
| **Ranking Items** | Identification of numeric columns representing entities to rank | Jaccard Index |
| **Indicator Column** | Detection of categorical column for stratified analysis | Exact Match Accuracy |
| **Indicator Values** | Extraction of unique values from indicator column | Set F1 |

## Dataset Categories

| Category | Count | Purpose |
|----------|-------|---------|
| `bigbetter_high` | 8 | Column names suggesting higher is better (accuracy, score, f1, etc.) |
| `bigbetter_low` | 10 | Column names suggesting lower is better (error, loss, latency, etc.) |
| `bigbetter_ambiguous` | 6 | Neutral column names (model_A, item_1, etc.) |
| `indicator_clear` | 8 | Clear indicator columns (Task, Category, Domain, etc.) |
| `indicator_none` | 6 | No suitable indicator column |
| `items_complex` | 6 | Mixed metadata and ranking columns |

## Usage

### 1. Generate Datasets

```bash
cd experiments/data_agent_f2
python generate_datasets.py
```

This creates:
- `datasets/` folder with test CSV files organized by category
- `datasets/manifest.json` with ground truth labels

### 2. Run Experiment

```bash
# Set environment variables
export $(cat ../../.env | grep -v '^#' | xargs)

# Run experiment
python run_experiment.py
```

### 3. Configure Number of Runs

Edit `run_experiment.py` and modify `N_RUNS` at the top:
- `N_RUNS = 1` for quick testing
- `N_RUNS = 5` or more for publication-ready results (mean +/- std)

## Output Files

Each run creates a timestamped folder in `runs/` containing:

| File | Description |
|------|-------------|
| `results.json` | Raw metrics in JSON format |
| `error_analysis.json` | Detailed error breakdown |
| `table_4_2_2.tex` | LaTeX table for JASA submission |
| `table_4_2_2.md` | Markdown table for writing.md |

## Copying Results to Paper

After running the experiment:

1. Open `runs/<timestamp>/table_4_2_2.md`
2. Copy the table content
3. Paste into `docs/project/writing.md` Section 4.2.2

## Ground Truth Schema

Each dataset in `manifest.json` includes:

```json
{
  "dataset_id": "bb_high_accuracy_01",
  "category": "bigbetter_high",
  "expected_bigbetter": 1,
  "expected_ranking_items": ["accuracy_A", "accuracy_B", ...],
  "expected_indicator_col": null,
  "expected_indicator_values": [],
  "difficulty": "easy"
}
```
