# F1 Case Study Assessment for OmniRank

## Verdict

Formula 1 is a strong fit for OmniRank if the case study is defined as **driver-level race classification ranking within a fixed regulation-era window**. For this repository, the retained case-study dataset is now strictly **`2024-2025`**.

Recommended setup:

- **Primary analysis unit:** driver-level Grand Prix classification
- **Primary dataset:** `2024-2025` race results only
- **Primary OmniRank input format:** one row per race, one column per driver, cell value = official finishing position
- **Primary stratification column:** `track_type`

## Why F1 Matches OmniRank

F1 aligns well with OmniRank's current multiway ranking pipeline:

- Each Grand Prix is a natural multiway comparison among roughly 20 drivers.
- Official race classifications are already a rank-position matrix.
- The comparison graph is dense because most drivers repeatedly co-appear.
- The data is reproducible from structured APIs plus official race-result pages.
- Stratification is interpretable: `track_type` is meaningful and low-cardinality.

## Retained Dataset Snapshot

Using the retained `2024-2025` data window:

- 48 Grands Prix
- 958 driver-race result rows
- 27 distinct drivers
- 10 constructors
- 10 sprint weekends
- Drivers per race: min `19`, max `20`, mean `19.96`
- OmniRank comparison count:
  - `M = 9082`
  - `n log n ≈ 88.99` for `n = 27`

Observed `track_type` distribution:

- `permanent`: 32
- `street`: 10
- `temporary_non_street`: 6

Observed status distribution:

- `Finished`: 617
- `Lapped`: 227
- `Retired`: 100
- `Disqualified`: 8
- `Did not start`: 6

## Verified Against OmniRank

The retained upload table was checked against the current OmniRank pipeline:

- detected format: `multiway`
- inferred indicator column: `track_type`
- inferred indicator values: `permanent`, `street`, `temporary_non_street`
- format validation: pass
- quality validation: pass
- connectivity: pass
- sparsity warning: none

The validated `2024-2025` upload table has:

- 48 rows
- 27 rankable driver columns
- 9082 effective pairwise co-comparisons induced by the multiway rows

An end-to-end spectral run on the retained table also completed successfully.

Example run characteristics:

- bootstrap iterations: `500`
- runtime: about `0.68` seconds
- top-5 mean rank-CI width: `3.8`
- top five ranks from the sample run:
  - `Max_Verstappen`: rank 1, 95% rank CI `[1, 4]`
  - `Lando_Norris`: rank 2, 95% rank CI `[1, 5]`
  - `Oscar_Piastri`: rank 3, 95% rank CI `[1, 5]`
  - `Charles_Leclerc`: rank 4, 95% rank CI `[2, 6]`
  - `George_Russell`: rank 5, 95% rank CI `[2, 6]`

This is a strong case-study outcome because OmniRank is not just reproducing standings; it is producing uncertainty-aware rankings on real multiway sports data.

## Window Selection Result

Different season windows were tested directly:

- `2025` only:
  - 24 race rows
  - 21 drivers
  - top-5 mean rank-CI width: `4.6`
- `2024-2025`:
  - 48 race rows
  - 27 drivers
  - top-5 mean rank-CI width: `3.8`
- `2021-2025` full field:
  - 114 race rows
  - 35 drivers
  - top-5 mean rank-CI width: `6.4`
- `2021-2025` recurring-driver subset:
  - 114 race rows
  - 20 drivers
  - top-5 mean rank-CI width: `4.8`

Conclusion:

- `2024-2025` improved precision relative to `2025` alone.
- Raw `2021-2025` pooling made the confidence intervals worse.
- The repository therefore keeps only `2024-2025` as the official F1 case-study window.

## Recommended Data Scheme

### 1. Core OmniRank Input

File: `data/case_study/f1_2024_2025_driver_results_wide.csv`

Shape:

- Rows: 48 races
- Metadata columns: `season_tag`, `round_tag`, `race_name`, and `track_type`
- Item columns: one column per driver

Row semantics:

- One row = one Grand Prix
- Driver cell value = official finishing position from the final classification
- Lower value is better (`bigbetter = 0`)

Recommended columns:

- `season_tag`
- `round_tag`
- `race_name`
- `track_type`
- one driver column per participant, e.g. `Lando_Norris`, `Max_Verstappen`, `Oscar_Piastri`

Recommended `track_type` taxonomy:

- `permanent`: purpose-built permanent circuit
- `street`: public-road street circuit
- `temporary_non_street`: temporary/non-permanent venue that is not best described as a city street circuit

### 2. Supporting Long Table

File: `data/case_study/f1_2024_2025_driver_results_long.csv`

Use this for reproducibility, filtering, and downstream enrichments. Recommended fields:

- race metadata:
  - `season`, `round`, `race_name`, `circuit_id`, `circuit_name`, `locality`, `country`, `date`, `track_type`, `weekend_type`
- driver identity:
  - `driver_id`, `driver_code`, `driver_name`, `driver_slug`, `constructor`
- race outcome:
  - `grid`, `finish_position`, `position_text`, `points`, `laps_completed`, `status`
- pace metadata:
  - `fastest_lap_rank`, `fastest_lap_number`, `fastest_lap_time`, `fastest_lap_speed_kph`

### 3. Driver Lookup

File: `data/case_study/f1_2024_2025_driver_lookup.csv`

This keeps the wide-table columns interpretable and stable:

- `driver_slug`
- `driver_name`
- `driver_code`
- `driver_id`
- `constructor`
- `standing_position`
- `standing_points`
- `wins`

## Cleaning Rules

Recommended defaults for the paper case study:

1. Keep the official classified finishing position as the ranking value.
2. Preserve `status` in the long table instead of silently dropping DNF/DQ/DNS.
3. Use the wide table for OmniRank and the long table for robustness filters.
4. Add an audited manual `track_type` label before pivoting to the upload table.
5. Keep `weekend_type` only in the supporting long table, not as the upload-table stratification field.
6. Keep the wide table limited to `season_tag`, `round_tag`, `race_name`, `track_type`, and driver columns.
7. Under the current schema heuristics, `track_type` is still auto-selected as the indicator because it carries an explicit indicator keyword, while the other retained metadata columns are there for readability and row identification.

Recommended robustness reruns:

1. Re-run after converting `Disqualified` and `Did not start` to `NA`.
2. Re-run within each `track_type`.
3. Re-run on only `standard` weekends as a robustness check, but not as the primary stratification.
4. Re-run on qualifying classifications to separate racecraft from Sunday execution.

## What To Avoid

These designs are weaker for the current project:

- Cross-era pooled driver ranking without explicit era modeling
- Constructor ranking from ad hoc aggregation rules
- Mixing race, sprint, and qualifying into one unlabeled table

## Recommended Paper Narrative

If F1 is used as the sports case study, the cleanest storyline is:

1. Upload the `2024-2025` driver-result wide table.
2. OmniRank detects a multiway rank-position matrix.
3. The system infers `bigbetter = 0` because lower finishing position is better.
4. The system detects `track_type` as the stratification column.
5. OmniRank produces:
   - an overall driver ranking for the retained two-season window
   - a `permanent` vs `street` vs `temporary_non_street` stratified comparison
   - confidence intervals showing whether the top drivers are statistically distinguishable

This is a better fit than treating F1 as a simple standings reproduction task because it exercises:

- multiway format recognition
- semantic inference on ranking direction
- indicator-based stratification
- uncertainty quantification beyond the official points table

## Expansion Plan

After the core race-result case study is stable, add:

1. Qualifying results
2. Sprint results
3. Pit stops and lap timing
4. FastF1-derived telemetry and weather layers
