#!/usr/bin/env python3
"""Tune pairwise data so Claude's CI becomes [1,2] instead of [1,1] (data-only, no code changes)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CSV = REPO_ROOT / "data" / "examples" / "example_data_pairwise.csv"
R_SCRIPT = REPO_ROOT / "src" / "spectral_ranking" / "spectral_ranking.R"
OUT_DIR = REPO_ROOT / "experiments" / "output" / "tune_ci"
SEED = 42
B = 2000


def run_spectral(csv_path: Path) -> dict | None:
    """Run spectral_ranking.R and return parsed JSON methods."""
    out_path = OUT_DIR / "ranking"
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "Rscript",
        str(R_SCRIPT),
        "--csv",
        str(csv_path.resolve()),
        "--bigbetter",
        "1",
        "--B",
        str(B),
        "--seed",
        str(SEED),
        "--out",
        str(out_path),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(result.stderr)
        return None
    jpath = out_path / "ranking_results.json"
    if not jpath.exists():
        return None
    with open(jpath) as f:
        data = json.load(f)
    return data.get("methods")


def find_claude_chatgpt_rows(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return (claude_wins_indices, chatgpt_wins_indices) for direct Claude vs ChatGPT rows."""
    ch_col = "ChatGPT"
    cl_col = "Claude"
    if ch_col not in df.columns or cl_col not in df.columns:
        return [], []
    claude_wins = []
    chatgpt_wins = []
    for i, row in df.iterrows():
        ch = row.get(ch_col)
        cl = row.get(cl_col)
        if pd.isna(ch) or pd.isna(cl):
            continue
        try:
            ch_v = int(float(ch))
            cl_v = int(float(cl))
        except (ValueError, TypeError):
            continue
        if ch_v not in (0, 1) or cl_v not in (0, 1):
            continue
        if ch_v + cl_v != 1:
            continue
        if ch_v == 1 and cl_v == 0:
            chatgpt_wins.append(int(i))
        elif ch_v == 0 and cl_v == 1:
            claude_wins.append(int(i))
    return claude_wins, chatgpt_wins


def main() -> None:
    df = pd.read_csv(ORIGINAL_CSV)
    claude_wins, chatgpt_wins = find_claude_chatgpt_rows(df)
    print(f"Claude wins (rows to potentially flip): {len(claude_wins)}")
    print(f"ChatGPT wins: {len(chatgpt_wins)}")

    # Baseline
    methods = run_spectral(ORIGINAL_CSV)
    if not methods:
        print("Baseline run failed")
        return
    for m in methods:
        if m["name"] == "Claude":
            print(f"Baseline Claude: rank={m['rank']}, CI={m['ci_two_sided']}")
            break

    # Try flipping k Claude wins to ChatGPT wins
    for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        df_mod = df.copy()
        to_flip = claude_wins[:k]
        for idx in to_flip:
            df_mod.at[idx, "ChatGPT"] = 1
            df_mod.at[idx, "Claude"] = 0
        mod_path = OUT_DIR / "tuned.csv"
        df_mod.to_csv(mod_path, index=False)

        methods = run_spectral(mod_path)
        if not methods:
            print(f"k={k}: run failed")
            continue

        claude_info = None
        chatgpt_info = None
        for m in methods:
            if m["name"] == "Claude":
                claude_info = m
            elif m["name"] == "ChatGPT":
                chatgpt_info = m

        if not claude_info or not chatgpt_info:
            continue

        ci = claude_info["ci_two_sided"]
        rank = claude_info["rank"]
        ok = ci == [1, 2] and rank == 1
        print(f"k={k}: Claude rank={rank}, CI={ci}, ChatGPT CI={chatgpt_info['ci_two_sided']} -> {'OK' if ok else 'no'}")

        if ok:
            print(f"\nFound: flip {k} Claude->ChatGPT wins. Saving to data/examples/example_data_pairwise.csv (copy).")
            out_copy = REPO_ROOT / "data" / "examples" / "example_data_pairwise_tuned.csv"
            df_mod.to_csv(out_copy, index=False)
            print(f"Saved: {out_copy}")
            return

    print("\nNo k in range achieved Claude [1,2]. Trying finer range...")
    for k in range(1, 55):
        df_mod = df.copy()
        to_flip = claude_wins[:k]
        for idx in to_flip:
            df_mod.at[idx, "ChatGPT"] = 1
            df_mod.at[idx, "Claude"] = 0
        mod_path = OUT_DIR / "tuned.csv"
        df_mod.to_csv(mod_path, index=False)
        methods = run_spectral(mod_path)
        if not methods:
            continue
        for m in methods:
            if m["name"] == "Claude":
                if m["ci_two_sided"] == [1, 2] and m["rank"] == 1:
                    print(f"k={k}: SUCCESS")
                    out_copy = REPO_ROOT / "data" / "examples" / "example_data_pairwise_tuned.csv"
                    df_mod.to_csv(out_copy, index=False)
                    print(f"Saved: {out_copy}")
                    return
                break


if __name__ == "__main__":
    main()
