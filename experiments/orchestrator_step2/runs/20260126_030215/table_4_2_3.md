## Two-Step Method Triggering Accuracy
*Results over 3 runs (mean +/- std)*

### Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 1.000+/-0.000 |
| Precision (Step 2 Triggered) | 1.000+/-0.000 |
| Recall (Step 2 Triggered) | 1.000+/-0.000 |
| F1 Score | 1.000+/-0.000 |
| Gatekeeper Accuracy | 1.000+/-0.000 |

### Accuracy by Scenario

| Scenario | Expected Step 2 | Accuracy |
|----------|-----------------|----------|
| boundary_sparsity_above | No | 1.000+/-0.000 |
| boundary_sparsity_below | No | 1.000+/-0.000 |
| large_items_sparse | No | 1.000+/-0.000 |
| large_items_sufficient | Yes | 1.000+/-0.000 |
| sparse_heterogeneous | No | 1.000+/-0.000 |
| sparse_homogeneous | No | 1.000+/-0.000 |
| sufficient_both | Yes | 1.000+/-0.000 |
| sufficient_heterogeneity | Yes | 1.000+/-0.000 |
| sufficient_no_trigger | No | 1.000+/-0.000 |
| sufficient_uncertainty | Yes | 1.000+/-0.000 |

### Confusion Matrix

```
                 Predicted
              No Step2   Step2
Actual No    [  75         0]
       Yes   [   0        75]
```