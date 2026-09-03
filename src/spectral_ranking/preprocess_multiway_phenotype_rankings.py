#!/usr/bin/env python3
"""Prepare a phenotype-by-method matrix for multiway spectral ranking."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize multiway phenotype ranking input into a clean wide CSV."
    )
    parser.add_argument(
        "--input",
        default="data/examples/prs_benchmarking_applied.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/examples/example_data_multiway_phenotype.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--indicator-col",
        default="Phenotype",
        help="Phenotype / indicator column to keep as the first column.",
    )
    parser.add_argument(
        "--min-observed-methods",
        type=int,
        default=2,
        help="Drop rows with fewer than this many observed numeric method values.",
    )
    parser.add_argument(
        "--drop-duplicate-rows",
        action="store_true",
        help="Drop exact duplicate rows before writing the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.indicator_col not in df.columns:
        raise ValueError(f"Indicator column not found: {args.indicator_col}")

    output = df.copy()
    if args.drop_duplicate_rows:
        output = output.drop_duplicates()

    numeric_cols = []
    for col in output.columns:
        if col == args.indicator_col:
            continue
        coerced = pd.to_numeric(output[col], errors="coerce")
        if coerced.notna().sum() > 0:
            output[col] = coerced
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        raise ValueError("At least two numeric method columns are required.")

    observed = output[numeric_cols].notna().sum(axis=1)
    output = output.loc[observed >= max(2, int(args.min_observed_methods))].copy()
    output = output[[args.indicator_col, *numeric_cols]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(output)}")
    print(f"Numeric method columns: {len(numeric_cols)}")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
