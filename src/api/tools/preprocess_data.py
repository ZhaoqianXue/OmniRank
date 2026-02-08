"""Tool: preprocess_data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import normalize_column_name, read_table, safe_numeric
from core.schemas import PreprocessResult, SemanticSchema


def _rank_column_order(column: str) -> tuple[int, str]:
    lower = column.lower()
    if lower.startswith("rank_"):
        suffix = lower.split("rank_", 1)[1]
        if suffix.isdigit():
            return int(suffix), lower
    return 9999, lower


def _pivot_long_if_possible(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Convert common long format (item/value) into wide matrix if detected."""
    lower_map = {col.lower(): col for col in df.columns}
    item_col = lower_map.get("item") or lower_map.get("item_name")
    value_col = lower_map.get("value") or lower_map.get("score") or lower_map.get("metric")

    if item_col and value_col:
        id_cols = [col for col in df.columns if col not in {item_col, value_col}]
        if not id_cols:
            df = df.copy()
            df["_row_id"] = range(len(df))
            id_cols = ["_row_id"]
        pivoted = (
            df.pivot_table(index=id_cols, columns=item_col, values=value_col, aggfunc="mean")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        return pivoted, True

    return df, False


def _convert_multiway_rank_columns_if_possible(
    df: pd.DataFrame,
    ranking_items: list[str],
) -> tuple[pd.DataFrame, bool]:
    """Convert rank_1, rank_2, ... (item names) into wide numeric rank matrix."""
    rank_columns = [col for col in df.columns if col.lower().startswith("rank_")]
    if not rank_columns:
        return df, False

    ordered_rank_columns = sorted(rank_columns, key=_rank_column_order)
    try:
        stacked = df[ordered_rank_columns].stack(future_stack=True).dropna()
    except TypeError:
        stacked = df[ordered_rank_columns].stack(dropna=True)
    if stacked.empty:
        return df, False

    item_values = sorted({str(value).strip() for value in stacked.astype(str).tolist() if str(value).strip()})
    if len(item_values) < 2:
        return df, False

    target_items = [item for item in ranking_items if item in item_values]
    if len(target_items) < 2:
        target_items = item_values

    meta_columns = [col for col in df.columns if col not in set(ordered_rank_columns)]
    transformed_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        base: dict[str, object] = {col: row[col] for col in meta_columns}
        for item in target_items:
            base[item] = None
        for rank_index, rank_col in enumerate(ordered_rank_columns, start=1):
            item_name = str(row[rank_col]).strip() if pd.notna(row[rank_col]) else ""
            if item_name and item_name in target_items:
                base[item_name] = rank_index
        transformed_rows.append(base)

    return pd.DataFrame(transformed_rows), True


def _convert_pairwise_long_if_possible(
    df: pd.DataFrame,
    ranking_items: list[str],
) -> tuple[pd.DataFrame, bool]:
    """Convert pairwise long format to wide item-score rows."""
    lower_map = {col.lower(): col for col in df.columns}
    left_col = (
        lower_map.get("item_a")
        or lower_map.get("item1")
        or lower_map.get("left")
        or lower_map.get("player_a")
    )
    right_col = (
        lower_map.get("item_b")
        or lower_map.get("item2")
        or lower_map.get("right")
        or lower_map.get("player_b")
    )
    winner_col = lower_map.get("winner") or lower_map.get("preferred") or lower_map.get("outcome")

    if not left_col or not right_col:
        return df, False

    pairwise_items = pd.concat([df[left_col], df[right_col]], axis=0).dropna().astype(str).tolist()
    unique_items = sorted({item.strip() for item in pairwise_items if item.strip()})
    if len(unique_items) < 2:
        return df, False

    target_items = [item for item in ranking_items if item in unique_items]
    if len(target_items) < 2:
        target_items = unique_items

    structural_cols = {left_col, right_col}
    if winner_col:
        structural_cols.add(winner_col)
    meta_cols = [col for col in df.columns if col not in structural_cols]

    transformed_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        left_item = str(row[left_col]).strip() if pd.notna(row[left_col]) else ""
        right_item = str(row[right_col]).strip() if pd.notna(row[right_col]) else ""
        if left_item not in target_items or right_item not in target_items:
            continue

        left_score = 1.0
        right_score = 0.0
        if winner_col and pd.notna(row[winner_col]):
            winner_raw = row[winner_col]
            winner_text = str(winner_raw).strip().lower()
            if winner_text in {left_item.lower(), "1", "true", "a", "left", "item_a"}:
                left_score, right_score = 1.0, 0.0
            elif winner_text in {right_item.lower(), "0", "false", "b", "right", "item_b"}:
                left_score, right_score = 0.0, 1.0
            elif winner_text in {"tie", "draw", "0.5"}:
                left_score, right_score = 0.5, 0.5
            else:
                parsed_numeric = pd.to_numeric(pd.Series([winner_raw]), errors="coerce").iloc[0]
                if pd.notna(parsed_numeric):
                    if float(parsed_numeric) >= 0.5:
                        left_score, right_score = 1.0, 0.0
                    else:
                        left_score, right_score = 0.0, 1.0

        base: dict[str, object] = {col: row[col] for col in meta_cols}
        for item in target_items:
            base[item] = None
        base[left_item] = left_score
        base[right_item] = right_score
        transformed_rows.append(base)

    if not transformed_rows:
        return df, False

    return pd.DataFrame(transformed_rows), True


def preprocess_data(
    file_path: str,
    schema: SemanticSchema,
    output_dir: str,
) -> PreprocessResult:
    """Apply fixable transformations and persist a preprocessed CSV."""
    df = read_table(file_path)
    transformation_log: list[str] = []
    original_rows = len(df)

    missing_items = [item for item in schema.ranking_items if item not in df.columns]
    if missing_items:
        pairwise_df, changed_pairwise = _convert_pairwise_long_if_possible(df, schema.ranking_items)
        if changed_pairwise:
            df = pairwise_df
            transformation_log.append("Restructured pairwise long format to wide item-score columns.")

    missing_items = [item for item in schema.ranking_items if item not in df.columns]
    if missing_items:
        multiway_df, changed_multiway = _convert_multiway_rank_columns_if_possible(df, schema.ranking_items)
        if changed_multiway:
            df = multiway_df
            transformation_log.append("Converted rank_* multiway columns to wide numeric rank matrix.")

    missing_items = [item for item in schema.ranking_items if item not in df.columns]
    if missing_items:
        pivoted, changed_pivot = _pivot_long_if_possible(df)
        if changed_pivot:
            df = pivoted
            transformation_log.append("Pivoted long format to wide format using item/value columns.")

    # Normalize column names only when conflicts exist with R parsing.
    renamed: dict[str, str] = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        if normalized != col:
            renamed[col] = normalized
    if renamed:
        df = df.rename(columns=renamed)
        transformation_log.append("Normalized column names for R compatibility.")

    ranking_items: list[str] = []
    for item in schema.ranking_items:
        if item in df.columns:
            ranking_items.append(item)
        elif item in renamed:
            ranking_items.append(renamed[item])

    # If schema items were inferred from pairwise/multiway values, keep all numeric item columns.
    if len(ranking_items) < 2:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        ranking_items = [col for col in numeric_cols if col not in {"B", "seed"}]

    # Convert ranking columns to numeric.
    for col in ranking_items:
        before_non_null = int(df[col].notna().sum())
        df[col] = safe_numeric(df[col])
        after_non_null = int(df[col].notna().sum())
        if after_non_null != before_non_null:
            transformation_log.append(
                f"Coerced '{col}' to numeric and dropped {before_non_null - after_non_null} non-numeric cells."
            )

    dropped_rows = 0
    if ranking_items:
        mask = df[ranking_items].notna().sum(axis=1) >= 2
        dropped_rows = int((~mask).sum())
        if dropped_rows > 0:
            transformation_log.append(
                f"Dropped {dropped_rows} rows with fewer than two valid ranking values."
            )
        df = df.loc[mask].reset_index(drop=True)

    output_path = Path(output_dir) / "preprocessed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return PreprocessResult(
        preprocessed_csv_path=str(output_path),
        transformation_log=transformation_log,
        row_count=int(len(df)),
        dropped_rows=dropped_rows if original_rows >= dropped_rows else 0,
    )
