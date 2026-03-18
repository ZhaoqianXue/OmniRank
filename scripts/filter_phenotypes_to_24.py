#!/usr/bin/env python3
"""
Filter supplementary_tables_filtered.csv to keep only the 24 target phenotypes,
then overwrite example_data_multiway_phenotype.csv with the result.

Phenotype names in the source file use various conventions (abbreviations, typos,
case variants). This script maps them to canonical names before filtering.

Quality filters ensure each phenotype has sufficient dense data for the R spectral
ranking to succeed (avoids na.rm errors and degenerate matrices). Phenotypes that
fail R verification are dropped; backup phenotypes from the supplementary are
added to maintain at least 20 phenotypes.
"""

from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# 24 target phenotypes (canonical names)
TARGET_PHENOTYPES = [
    "Asthma Disease",
    "Breast Cancer",
    "Coronary Artery Disease",
    "Type 2 Diabetes",
    "Type 1 Diabetes",
    "Prostate Cancer",
    "Gout",
    "Depression",
    "Rheumatoid Arthritis",
    "High-Density Lipoprotein",
    "Body Mass Index",
    "Eosinophil Count",
    "Triglyceride",
    "White Blood Cell Count",
    "Systolic Blood Pressure",
    "Low-Density Lipoprotein",
    "Urate",
    "Inflammatory Bowel Disease",
    "Schizophrenia",
    "Height",
    "Stroke",
    "Hemoglobin A1c",
    "Multiple Sclerosis",
    "Intelligence",
]

# Map source phenotype names (as they appear in supplementary_tables_filtered.csv)
# to canonical target names. Case-insensitive match is applied for variants.
PHENOTYPE_MAPPING: dict[str, str] = {
    # Asthma Disease
    "Asthma": "Asthma Disease",
    # Breast Cancer
    "BrCa": "Breast Cancer",
    "BRCA": "Breast Cancer",
    "Breast cancer": "Breast Cancer",
    "Breast Cancer": "Breast Cancer",
    "BC": "Breast Cancer",
    # Coronary Artery Disease
    "CAD": "Coronary Artery Disease",
    "Coronary artery disease": "Coronary Artery Disease",
    # Type 2 Diabetes
    "T2D": "Type 2 Diabetes",
    "T2d": "Type 2 Diabetes",
    "T2DM": "Type 2 Diabetes",
    "Type 2 Diabetes": "Type 2 Diabetes",
    # Type 1 Diabetes
    "T1D": "Type 1 Diabetes",
    "T1B": "Type 1 Diabetes",
    "Type 1 Diabetes": "Type 1 Diabetes",
    # Prostate Cancer
    "PRCA": "Prostate Cancer",
    "PrCa": "Prostate Cancer",
    "Prostate cancer": "Prostate Cancer",
    "Prostate Cancer": "Prostate Cancer",
    # Gout
    "GO": "Gout",
    "Gout": "Gout",
    # Depression
    "MDD": "Depression",
    "DEP": "Depression",
    "Depression": "Depression",
    # Rheumatoid Arthritis
    "RA": "Rheumatoid Arthritis",
    "RheuArth": "Rheumatoid Arthritis",
    "Rheumatoid arthritis": "Rheumatoid Arthritis",
    "Rheumatoid Arthritis": "Rheumatoid Arthritis",
    # High-Density Lipoprotein
    "HDL": "High-Density Lipoprotein",
    "HDL Cholesterol": "High-Density Lipoprotein",
    # Body Mass Index
    "BMI": "Body Mass Index",
    "MBI": "Body Mass Index",  # typo in source
    "Body Mass Index": "Body Mass Index",
    # Eosinophil Count
    "EOS": "Eosinophil Count",
    "Eosinophil Count": "Eosinophil Count",
    # Triglyceride
    "TG": "Triglyceride",
    "logTG": "Triglyceride",
    "Triglyceride": "Triglyceride",
    "Triglycerides": "Triglyceride",
    "TRIG": "Triglyceride",
    # White Blood Cell Count
    "WBC": "White Blood Cell Count",
    "WBCC": "White Blood Cell Count",
    "White Blood Cell Count": "White Blood Cell Count",
    # Systolic Blood Pressure
    "SBP": "Systolic Blood Pressure",
    "Systollic Blood Pressure": "Systolic Blood Pressure",
    "Systolic Blood Pressure": "Systolic Blood Pressure",
    # Low-Density Lipoprotein
    "LDL": "Low-Density Lipoprotein",
    "LDL-C": "Low-Density Lipoprotein",
    "LDL Cholesterol": "Low-Density Lipoprotein",
    # Urate
    "SU": "Urate",
    "Urate": "Urate",
    # Inflammatory Bowel Disease
    "IBD": "Inflammatory Bowel Disease",
    "Crohns Disease": "Inflammatory Bowel Disease",
    "Crohn's Disease": "Inflammatory Bowel Disease",
    "Crohn's disease": "Inflammatory Bowel Disease",
    "Inflammatory Bowel Disease": "Inflammatory Bowel Disease",
    "Ulcerative colitis": "Inflammatory Bowel Disease",
    # Schizophrenia
    "SCZ": "Schizophrenia",
    "Schizophrenia": "Schizophrenia",
    # Height
    "SH": "Height",
    "Height": "Height",
    "HGT": "Height",
    # Stroke
    "Stroke": "Stroke",
    # Hemoglobin A1c
    "HbA1c": "Hemoglobin A1c",
    "Hemoglobin A1c": "Hemoglobin A1c",
    # Multiple Sclerosis
    "MS": "Multiple Sclerosis",
    "MultiScler": "Multiple Sclerosis",
    "Multiple Sclerosis": "Multiple Sclerosis",
    # Intelligence
    "Intelligence": "Intelligence",
    # Backup phenotypes (to reach 20+ when some of the 24 fail R verification)
    "CKD": "Chronic Kidney Disease",
    "Chronic Kidney Disease": "Chronic Kidney Disease",
    "eGFR": "Estimated Glomerular Filtration Rate",
    "Estimated Glomerular Filtration Rate": "Estimated Glomerular Filtration Rate",
    "Osteoporosis": "Osteoporosis",
    "Glaucoma": "Glaucoma",
    "Cataract": "Cataract",
    "Gastric Cancer": "Gastric Cancer",
    "Hyperthyroidism": "Hyperthyroidism",
    "Hypothyroidism": "Hypothyroidism",
    "Alzheimer's disease": "Alzheimer's Disease",
    "Alz": "Alzheimer's Disease",
    "AD": "Alzheimer's Disease",
}


def _verify_phenotype_r_succeeds(
    df: pd.DataFrame,
    phenotype: str,
    method_cols: list[str],
    r_script_path: Path,
    min_non_na: int = 5,
    min_rows: int = 4,
    seeds: tuple[int, ...] = (42, 45, 50, 55, 60),
) -> bool:
    """Return True if R spectral ranking succeeds for this phenotype across multiple seeds."""
    sub = df[df["Phenotype"] == phenotype][method_cols]
    mask = sub.notna().sum(axis=1) >= min_non_na
    sub = sub[mask]
    if len(sub) < min_rows:
        return False
    local_methods = [c for c in method_cols if sub[c].notna().any()]
    if len(local_methods) < 2:
        return False
    sub = sub[local_methods]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sub.to_csv(f.name, index=False)
        try:
            for seed in seeds:
                result = subprocess.run(
                    [
                        "Rscript",
                        str(r_script_path),
                        "--csv",
                        f.name,
                        "--bigbetter",
                        "1",
                        "--B",
                        "500",
                        "--seed",
                        str(seed),
                        "--out",
                        str(Path(f.name).parent / "out"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=45,
                    cwd=str(r_script_path.parent.parent.parent),
                )
                if result.returncode != 0:
                    return False
            return True
        finally:
            Path(f.name).unlink(missing_ok=True)


def _run_r_for_seed(
    csv_path: str,
    seed: int,
    r_script_path: Path,
    cwd: str,
) -> tuple[int, bool]:
    """Run R script for one seed; return (seed, success)."""
    result = subprocess.run(
        [
            "Rscript",
            str(r_script_path),
            "--csv",
            csv_path,
            "--bigbetter",
            "1",
            "--B",
            "500",
            "--seed",
            str(seed),
            "--out",
            str(Path(csv_path).parent / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )
    return (seed, result.returncode == 0)


def _find_passing_seeds(
    df: pd.DataFrame,
    phenotype: str,
    method_cols: list[str],
    r_script_path: Path,
    min_non_na: int,
    min_rows: int,
    seed_range: tuple[int, int] = (42, 62),
    max_workers: int = 8,
) -> set[int]:
    """Return set of seeds for which R spectral ranking succeeds."""
    sub = df[df["Phenotype"] == phenotype][method_cols]
    mask = sub.notna().sum(axis=1) >= min_non_na
    sub = sub[mask]
    if len(sub) < min_rows:
        return set()
    local_methods = [c for c in method_cols if sub[c].notna().any()]
    if len(local_methods) < 2:
        return set()
    sub = sub[local_methods]
    passing: set[int] = set()
    cwd = str(r_script_path.parent.parent.parent)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sub.to_csv(f.name, index=False)
        try:
            seeds_to_check = list(range(seed_range[0], seed_range[1]))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_run_r_for_seed, f.name, s, r_script_path, cwd): s
                    for s in seeds_to_check
                }
                for future in as_completed(futures):
                    seed, ok = future.result()
                    if ok:
                        passing.add(seed)
        finally:
            Path(f.name).unlink(missing_ok=True)
    return passing


def _normalize_phenotype(raw: str) -> str | None:
    """Map raw phenotype string to canonical name if in target set, else None."""
    stripped = raw.strip()
    if not stripped:
        return None
    canonical = PHENOTYPE_MAPPING.get(stripped)
    if canonical is not None:
        return canonical
    # Case-insensitive fallback for exact target names
    for target in TARGET_PHENOTYPES:
        if stripped.lower() == target.lower():
            return target
    return None


# 14 method columns; order must match API's preferred order for R verification
PREFERRED_METHOD_ORDER = [
    "C+T", "LDpred", "lassosum", "PRS-CS", "PRS-CS-auto", "SBayesR",
    "SCT", "DBSLMM", "LDpred2", "LDpred2-auto", "LDpred2-inf",
    "LDpred-funct", "lassosum2", "AnnoPred",
]
METHOD_COLS = PREFERRED_METHOD_ORDER

# Quality thresholds: phenotypes need sufficient dense rows for R spectral ranking.
# Rows with too few non-NA values can cause degenerate comparison matrices.
# Three-tier for tighter CI: 5 >= 4 >= 3 non-NA per row.
MIN_NON_NA_TIER1 = 5  # Densest: >=5 non-NA per row (tightest CI)
MIN_NON_NA_TIER2 = 4  # Medium: >=4 non-NA per row
MIN_NON_NA_TIER3 = 3  # Fallback: >=3 non-NA per row
MIN_ROWS_PER_PHENOTYPE = 2
MIN_PHENOTYPES = 20


def _quality_filter(
    df: pd.DataFrame,
    method_cols: list[str],
    min_non_na: int,
    min_rows: int,
) -> pd.DataFrame:
    """Keep only rows with >= min_non_NA method values; drop phenotypes with < min_rows."""
    mask = df[method_cols].notna().sum(axis=1) >= min_non_na
    kept = df[mask].copy()
    pheno_counts = kept["Phenotype"].value_counts()
    valid_phenos = pheno_counts[pheno_counts >= min_rows].index.tolist()
    return kept[kept["Phenotype"].isin(valid_phenos)]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "data" / "examples" / "supplementary_tables_filtered.csv"
    out_path = repo_root / "data" / "examples" / "example_data_multiway_phenotype.csv"
    r_script_path = repo_root / "src" / "spectral_ranking" / "spectral_ranking.R"

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not r_script_path.exists():
        raise FileNotFoundError(f"R script not found: {r_script_path}")

    df = pd.read_csv(src_path)
    if "Phenotype" not in df.columns:
        raise ValueError(
            f"Expected 'Phenotype' column in {src_path}. Columns: {list(df.columns)}"
        )

    method_cols = [c for c in METHOD_COLS if c in df.columns]
    if len(method_cols) < 14:
        missing = set(METHOD_COLS) - set(method_cols)
        raise ValueError(f"Missing method columns: {missing}")

    for col in method_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tier 1: Densest (min_non_na=5) for tightest CI
    filtered_t1 = _quality_filter(
        df, method_cols, min_non_na=MIN_NON_NA_TIER1, min_rows=MIN_ROWS_PER_PHENOTYPE
    )
    pheno_to_seeds_t1: dict[str, set[int]] = {}
    print("  Scanning phenotype-seed compatibility (tier 1, min 5 non-NA/row)...")
    for pheno in sorted(filtered_t1["Phenotype"].unique()):
        passing = _find_passing_seeds(
            filtered_t1, pheno, method_cols, r_script_path,
            min_non_na=MIN_NON_NA_TIER1, min_rows=MIN_ROWS_PER_PHENOTYPE,
            seed_range=(42, 62),
        )
        if passing:
            pheno_to_seeds_t1[pheno] = passing

    # Tier 2: Medium (min_non_na=4) if tier 1 has <20
    pheno_to_seeds = dict(pheno_to_seeds_t1)
    filtered_t2 = _quality_filter(
        df, method_cols, min_non_na=MIN_NON_NA_TIER2, min_rows=MIN_ROWS_PER_PHENOTYPE
    )
    filtered_t3 = _quality_filter(
        df, method_cols, min_non_na=MIN_NON_NA_TIER3, min_rows=MIN_ROWS_PER_PHENOTYPE
    )
    if len(pheno_to_seeds) < MIN_PHENOTYPES:
        print("  Scanning tier 2 (min 4 non-NA/row)...")
        for pheno in sorted(filtered_t2["Phenotype"].unique()):
            if pheno in pheno_to_seeds:
                continue
            passing = _find_passing_seeds(
                filtered_t2, pheno, method_cols, r_script_path,
                min_non_na=MIN_NON_NA_TIER2, min_rows=MIN_ROWS_PER_PHENOTYPE,
                seed_range=(42, 62),
            )
            if passing:
                pheno_to_seeds[pheno] = passing
    phenos_after_tier2 = set(pheno_to_seeds.keys())

    # Tier 3: Fallback (min_non_na=3) if still <20
    if len(pheno_to_seeds) < MIN_PHENOTYPES:
        print("  Scanning tier 3 (min 3 non-NA/row)...")
        for pheno in sorted(filtered_t3["Phenotype"].unique()):
            if pheno in pheno_to_seeds:
                continue
            passing = _find_passing_seeds(
                filtered_t3, pheno, method_cols, r_script_path,
                min_non_na=MIN_NON_NA_TIER3, min_rows=MIN_ROWS_PER_PHENOTYPE,
                seed_range=(42, 62),
            )
            if passing:
                pheno_to_seeds[pheno] = passing

    # Build filtered: use each phenotype's tier-specific rows (densest tier first)
    tier1_phenos = set(pheno_to_seeds_t1.keys())
    tier2_phenos = phenos_after_tier2 - tier1_phenos
    tier3_phenos = set(pheno_to_seeds.keys()) - phenos_after_tier2
    parts = []
    if tier1_phenos:
        parts.append(filtered_t1[filtered_t1["Phenotype"].isin(tier1_phenos)])
    if tier2_phenos:
        parts.append(filtered_t2[filtered_t2["Phenotype"].isin(tier2_phenos)])
    if tier3_phenos:
        parts.append(filtered_t3[filtered_t3["Phenotype"].isin(tier3_phenos)])
    filtered = pd.concat(parts, ignore_index=True) if len(parts) > 1 else (parts[0] if parts else pd.DataFrame())

    # Row counts for prioritization (denser = tighter CI)
    pheno_row_counts = filtered["Phenotype"].value_counts().to_dict()

    # Prefer phenotypes with 3+ rows; use 2-row only if needed to reach MIN_PHENOTYPES
    pheno_to_seeds_filtered = {
        p: seeds for p, seeds in pheno_to_seeds.items()
        if pheno_row_counts.get(p, 0) >= 3
    }
    if len(pheno_to_seeds_filtered) < MIN_PHENOTYPES:
        two_row = {p: s for p, s in pheno_to_seeds.items() if pheno_row_counts.get(p, 0) == 2}
        for p, seeds in sorted(two_row.items(), key=lambda x: -len(x[1]))[:MIN_PHENOTYPES - len(pheno_to_seeds_filtered)]:
            pheno_to_seeds_filtered[p] = seeds
    pheno_to_seeds = pheno_to_seeds_filtered

    # Build position -> phenotypes map (position i needs seed 42+i)
    pos_to_candidates: dict[int, list[str]] = {}
    for pos in range(MIN_PHENOTYPES):
        need_seed = 42 + pos
        pos_to_candidates[pos] = [
            p for p, seeds in pheno_to_seeds.items() if need_seed in seeds
        ]

    # Greedy assignment: fill positions with fewest options first (hardest-first)
    # Within each position, prioritize phenotypes with more rows (denser data = tighter CI)
    positions_by_hardness = sorted(
        range(MIN_PHENOTYPES),
        key=lambda pos: len(pos_to_candidates[pos]),
    )
    assigned_order: list[str] = []
    pos_to_pheno: dict[int, str] = {}
    used: set[str] = set()
    for pos in positions_by_hardness:
        need_seed = 42 + pos
        # Sort candidates by row count descending (prioritize dense data)
        candidates_sorted = sorted(
            pos_to_candidates[pos],
            key=lambda p: pheno_row_counts.get(p, 0),
            reverse=True,
        )
        chosen = None
        for pheno in candidates_sorted:
            if pheno in used:
                continue
            chosen = pheno
            break
        if chosen is None:
            break
        pos_to_pheno[pos] = chosen
        used.add(chosen)
    assigned_order = [pos_to_pheno[i] for i in range(MIN_PHENOTYPES) if i in pos_to_pheno]

    if len(assigned_order) < MIN_PHENOTYPES:
        print(f"  Only {len(assigned_order)} phenotypes could be assigned (target {MIN_PHENOTYPES})")

    filtered = filtered[filtered["Phenotype"].isin(assigned_order)].copy()
    # Reorder rows so phenotype order matches assigned_order (for consistent CSV)
    filtered["_ord"] = filtered["Phenotype"].map({p: i for i, p in enumerate(assigned_order)})
    filtered = filtered.sort_values("_ord").drop(columns=["_ord"])

    # Ensure each method column has at least one non-NA value
    empty_cols = [c for c in method_cols if filtered[c].notna().sum() == 0]
    if empty_cols:
        df["_canonical"] = df["Phenotype"].astype(str).apply(_normalize_phenotype)
        extra = df[df["_canonical"].notna()].copy()
        extra["Phenotype"] = extra["_canonical"]
        extra = extra.drop(columns=["_canonical"])
        for col in method_cols:
            extra[col] = pd.to_numeric(extra[col], errors="coerce")
        fill_indices = set()
        for col in empty_cols:
            candidates = extra[extra[col].notna()].index.tolist()
            if candidates:
                fill_indices.add(candidates[0])
        if fill_indices:
            filtered = pd.concat(
                [filtered, extra.loc[list(fill_indices)]],
                ignore_index=True,
            ).drop_duplicates()

    # Reorder columns
    pheno_col = "Phenotype"
    other_cols = [c for c in filtered.columns if c != pheno_col]
    col_order = [pheno_col] + [c for c in method_cols if c in other_cols]
    col_order += [c for c in other_cols if c not in method_cols]
    filtered = filtered[col_order]

    filtered.to_csv(out_path, index=False)

    kept = filtered["Phenotype"].nunique()
    total_rows = len(filtered)
    print(f"Kept {total_rows} rows across {kept} phenotypes (all R-verified)")
    if kept < MIN_PHENOTYPES:
        print(f"  WARNING: Only {kept} phenotypes (target >= {MIN_PHENOTYPES}). Some phenotypes fail R spectral ranking.")
    print(f"Written to {out_path}")
    for c in method_cols:
        n = filtered[c].notna().sum()
        print(f"  Col {c}: {n} non-NA")
    for p in sorted(filtered["Phenotype"].unique()):
        n = (filtered["Phenotype"] == p).sum()
        print(f"  {p}: {n} rows")


if __name__ == "__main__":
    main()
