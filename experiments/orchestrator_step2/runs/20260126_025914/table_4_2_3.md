## Two-Step Method Triggering Accuracy
*Results over 1 runs (mean +/- std)*

### Overall Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.860+/-0.000 |
| Precision (Step 2 Triggered) | 1.000+/-0.000 |
| Recall (Step 2 Triggered) | 0.720+/-0.000 |
| F1 Score | 0.837+/-0.000 |
| Gatekeeper Accuracy | 1.000+/-0.000 |

### Accuracy by Scenario

| Scenario | Expected Step 2 | Accuracy |
|----------|-----------------|----------|
| boundary_sparsity_above | No | 0.800+/-0.000 |
| boundary_sparsity_below | No | 1.000+/-0.000 |
| large_items_sparse | No | 1.000+/-0.000 |
| large_items_sufficient | Yes | 0.000+/-0.000 |
| sparse_heterogeneous | No | 1.000+/-0.000 |
| sparse_homogeneous | No | 1.000+/-0.000 |
| sufficient_both | Yes | 1.000+/-0.000 |
| sufficient_heterogeneity | Yes | 0.800+/-0.000 |
| sufficient_no_trigger | No | 1.000+/-0.000 |
| sufficient_uncertainty | Yes | 1.000+/-0.000 |

### Confusion Matrix

```
                 Predicted
              No Step2   Step2
Actual No    [  25         0]
       Yes   [   7        18]
```