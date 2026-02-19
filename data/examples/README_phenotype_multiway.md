# Example: Multiway Matrix with Phenotype (PRS Method Ranking)

This dataset demonstrates a multiway matrix with phenotype annotations (wide-table format) for ranking PRS methods across phenotypes. It is reverse-engineered from reference figures to produce matching outputs.

## Reference Image Alignment Checklist

| Element | Reference | Implementation |
|---------|-----------|----------------|
| **Forest Plot** | LDpred2 #1 ... C+T #14 | Overall spectral ranking matches exactly |
| **Violin (A)** | 13 methods (no AnnoPred) | `violin_method_order`, `violin_spec` in meta |
| **Violin K values** | C+T=27, LDpred=13, ... | `method_coverage` + `violin_spec` |
| **Violin lambda_mean** | C+T=11.39, LDpred=9.00, ... | `violin_spec` stores target labels; `heatmap_rank_matrix_*` is the closest discrete approximation under ranks {2,5,7,9,11,13} |
| **Heatmap (B/C)** | 14 methods inc. AnnoPred | `heatmap_method_order` |
| **Heatmap colors** | 2, 5, 7, 9, 11, 13 | `heatmap_discrete_ranks` |
| **Continuous phenotypes** | 15 (BMI, Height, ...) | `continuous_phenotypes` |
| **Binary phenotypes** | 17 | `binary_phenotypes` |
| **Gray cells** | AnnoPred, LDpred-funct sparse | `method_coverage` defines applicability |

## Data Structure

| Column | Type | Description |
|--------|------|-------------|
| phenotype | string | Stratification dimension |
| sample_id | string | Sample identifier within phenotype |
| LDpred2, AnnoPred, ... C+T | numeric | 14 PRS method scores (higher = better, bigbetter=1) |

- **32 phenotypes**: 15 continuous + 17 binary
- **~55 samples per phenotype** -> ~1,760 rows
- **14 methods** total; Violin shows 13 (AnnoPred excluded)

## Regenerate Data

```bash
python scripts/generate_phenotype_multiway_data.py
```

Outputs:
- `example_data_multiway_phenotype.csv`
- `example_data_multiway_phenotype_meta.json` (violin_spec, heatmap_rank_matrix_*, method_coverage)
