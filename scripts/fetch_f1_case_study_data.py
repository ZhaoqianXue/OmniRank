#!/usr/bin/env python3
"""Fetch real F1 race results and materialize OmniRank-ready case study tables.

This script uses the Jolpica F1 API (Ergast-compatible) to download official race
classification data for one season or a fixed season window and then writes:

1. A long table with detailed per-driver race results.
2. A wide OmniRank-ready table where each row is one Grand Prix and each driver
   column stores the official finishing position for that race.
3. A driver lookup table for mapping wide-table columns back to readable names.

The wide table is intentionally designed to match OmniRank's preferred multiway
rank-position format: lower values are better and missing values indicate that a
driver did not meaningfully participate in that comparison row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_BASE = "https://api.jolpi.ca/ergast/f1"
DEFAULT_SEASONS = [2024, 2025]
DEFAULT_OUTPUT_DIR = Path("data/case_study")
DEFAULT_CACHE_DIR = Path(".cache/jolpica_f1")

# Manually audited circuit taxonomy for the supported F1 calendars used in this case study.
# Categories are intentionally conservative:
# - permanent: purpose-built permanent racing facility
# - street: public-road street circuit
# - temporary_non_street: temporary/non-permanent venue that is not best described
#   as a public city street circuit (e.g. park roads, stadium complex roads)
TRACK_TYPE_BY_CIRCUIT_ID = {
    "albert_park": "temporary_non_street",
    "shanghai": "permanent",
    "suzuka": "permanent",
    "bahrain": "permanent",
    "portimao": "permanent",
    "istanbul": "permanent",
    "jeddah": "street",
    "miami": "temporary_non_street",
    "imola": "permanent",
    "monaco": "street",
    "catalunya": "permanent",
    "villeneuve": "temporary_non_street",
    "red_bull_ring": "permanent",
    "silverstone": "permanent",
    "spa": "permanent",
    "paul_ricard": "permanent",
    "ricard": "permanent",
    "hungaroring": "permanent",
    "zandvoort": "permanent",
    "monza": "permanent",
    "baku": "street",
    "marina_bay": "street",
    "sochi": "temporary_non_street",
    "americas": "permanent",
    "rodriguez": "permanent",
    "interlagos": "permanent",
    "vegas": "street",
    "losail": "permanent",
    "yas_marina": "permanent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Single F1 season to fetch. Ignored when --seasons is provided.",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated seasons to combine. Defaults to 2024,2025.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV outputs will be written.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
        help="Delay between API requests to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP request timeout.",
    )
    return parser.parse_args()


def parse_season_list(season: int | None, seasons_arg: str | None) -> list[int]:
    if seasons_arg:
        parsed = [int(part.strip()) for part in seasons_arg.split(",") if part.strip()]
        if not parsed:
            raise ValueError("--seasons was provided but no valid years were parsed.")
        return parsed

    if season is not None:
        return [season]

    return DEFAULT_SEASONS.copy()


def _safe_slug(text: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "_", normalized.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def fetch_json(
    url: str,
    *,
    timeout_seconds: float,
    max_attempts: int = 6,
    initial_backoff: float = 1.0,
) -> dict[str, Any]:
    """Fetch JSON with simple retry/backoff for 429 and transient failures."""
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = DEFAULT_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    backoff = initial_backoff
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            if response.status_code == 429:
                raise RuntimeError(f"Rate limited by Jolpica API at {url}")
            response.raise_for_status()
            payload = response.json()
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_calendar(season: int, timeout_seconds: float) -> list[dict[str, Any]]:
    payload = fetch_json(
        f"{API_BASE}/{season}/races.json?limit=100",
        timeout_seconds=timeout_seconds,
    )
    return payload["MRData"]["RaceTable"]["Races"]


def fetch_sprint_rounds(season: int, timeout_seconds: float) -> set[int]:
    payload = fetch_json(
        f"{API_BASE}/{season}/sprint.json?limit=100",
        timeout_seconds=timeout_seconds,
    )
    races = payload["MRData"]["RaceTable"]["Races"]
    return {int(race["round"]) for race in races if race.get("SprintResults")}


def fetch_driver_standings(season: int, timeout_seconds: float) -> pd.DataFrame:
    payload = fetch_json(
        f"{API_BASE}/{season}/driverstandings.json",
        timeout_seconds=timeout_seconds,
    )
    standings = payload["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    rows: list[dict[str, Any]] = []
    for row in standings:
        driver = row["Driver"]
        rows.append(
            {
                "driver_id": driver["driverId"],
                "driver_code": driver.get("code"),
                "driver_name": f"{driver['givenName']} {driver['familyName']}",
                "standing_position": int(row["position"]),
                "standing_points": float(row["points"]),
                "wins": int(row["wins"]),
                "constructor": row["Constructors"][0]["name"],
            }
        )
    return pd.DataFrame(rows)


def infer_track_type(circuit_id: str) -> str:
    try:
        return TRACK_TYPE_BY_CIRCUIT_ID[circuit_id]
    except KeyError as exc:
        raise KeyError(
            f"Missing audited track_type mapping for circuit_id='{circuit_id}'."
        ) from exc


def build_long_results(
    season: int,
    *,
    timeout_seconds: float,
    pause_seconds: float,
) -> pd.DataFrame:
    sprint_rounds = fetch_sprint_rounds(season, timeout_seconds)
    time.sleep(pause_seconds)
    calendar = fetch_calendar(season, timeout_seconds)
    time.sleep(pause_seconds)

    rows: list[dict[str, Any]] = []
    for race in calendar:
        round_number = int(race["round"])
        race_payload = fetch_json(
            f"{API_BASE}/{season}/{round_number}/results.json?limit=100",
            timeout_seconds=timeout_seconds,
        )
        race_result = race_payload["MRData"]["RaceTable"]["Races"][0]
        circuit = race_result["Circuit"]
        location = circuit["Location"]
        weekend_type = "sprint" if round_number in sprint_rounds else "standard"
        track_type = infer_track_type(circuit["circuitId"])

        for entry in race_result["Results"]:
            driver = entry["Driver"]
            fastest_lap = entry.get("FastestLap") or {}
            fastest_lap_time = fastest_lap.get("Time") or {}
            fastest_lap_speed = fastest_lap.get("AverageSpeed") or {}
            driver_name = f"{driver['givenName']} {driver['familyName']}"

            rows.append(
                {
                    "season": int(race_result["season"]),
                    "round": round_number,
                    "race_name": race_result["raceName"],
                    "circuit_id": circuit["circuitId"],
                    "circuit_name": circuit["circuitName"],
                    "locality": location["locality"],
                    "country": location["country"],
                    "date": race_result["date"],
                    "track_type": track_type,
                    "weekend_type": weekend_type,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code"),
                    "driver_name": driver_name,
                    "driver_slug": _safe_slug(driver_name),
                    "constructor": entry["Constructor"]["name"],
                    "grid": pd.to_numeric(entry.get("grid"), errors="coerce"),
                    "finish_position": pd.to_numeric(entry.get("position"), errors="coerce"),
                    "position_text": entry.get("positionText"),
                    "points": pd.to_numeric(entry.get("points"), errors="coerce"),
                    "laps_completed": pd.to_numeric(entry.get("laps"), errors="coerce"),
                    "status": entry.get("status"),
                    "fastest_lap_rank": pd.to_numeric(fastest_lap.get("rank"), errors="coerce"),
                    "fastest_lap_number": pd.to_numeric(fastest_lap.get("lap"), errors="coerce"),
                    "fastest_lap_time": fastest_lap_time.get("time"),
                    "fastest_lap_speed_kph": pd.to_numeric(
                        fastest_lap_speed.get("speed"),
                        errors="coerce",
                    ),
                }
            )
        time.sleep(pause_seconds)

    return (
        pd.DataFrame(rows)
        .sort_values(["season", "round", "finish_position", "driver_slug"])
        .reset_index(drop=True)
    )


def build_driver_lookup(long_df: pd.DataFrame, standings_df: pd.DataFrame) -> pd.DataFrame:
    lookup = (
        long_df[
            [
                "driver_slug",
                "driver_name",
                "driver_code",
                "driver_id",
                "constructor",
            ]
        ]
        .drop_duplicates(subset=["driver_slug"])
        .copy()
    )
    merged = lookup.merge(
        standings_df,
        on=["driver_id", "driver_code", "driver_name"],
        how="left",
        suffixes=("", "_standings"),
    )
    merged["standing_position"] = pd.to_numeric(merged["standing_position"], errors="coerce")
    merged = merged.sort_values(
        by=["standing_position", "driver_slug"],
        na_position="last",
    ).reset_index(drop=True)
    return merged


def build_wide_results(long_df: pd.DataFrame, driver_lookup: pd.DataFrame) -> pd.DataFrame:
    export_df = long_df.copy()
    export_df["season_tag"] = export_df["season"].map(lambda value: f"S{int(value)}")
    export_df["round_tag"] = export_df["round"].map(lambda value: f"R{int(value):02d}")

    metadata_cols = [
        "season_tag",
        "round_tag",
        "race_name",
        "track_type",
    ]
    wide = export_df.pivot_table(
        index=metadata_cols,
        columns="driver_slug",
        values="finish_position",
        aggfunc="first",
    ).reset_index()

    ordered_driver_cols = driver_lookup["driver_slug"].tolist()
    existing_driver_cols = [col for col in ordered_driver_cols if col in wide.columns]
    remaining_cols = [col for col in wide.columns if col not in set(metadata_cols + existing_driver_cols)]
    wide = wide[metadata_cols + existing_driver_cols + remaining_cols]
    return wide.sort_values(["season_tag", "round_tag"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seasons = parse_season_list(args.season, args.seasons)

    long_parts: list[pd.DataFrame] = []
    standings_parts: list[pd.DataFrame] = []
    for season in seasons:
        long_parts.append(
            build_long_results(
                season,
                timeout_seconds=args.timeout_seconds,
                pause_seconds=args.pause_seconds,
            )
        )
        standings = fetch_driver_standings(season, args.timeout_seconds).copy()
        standings["season"] = season
        standings_parts.append(standings)

    long_df = pd.concat(long_parts, ignore_index=True)
    standings_df = pd.concat(standings_parts, ignore_index=True).sort_values(
        ["season", "standing_position"]
    )
    latest_standings = standings_df.sort_values("season").drop_duplicates(
        subset=["driver_id"],
        keep="last",
    )
    driver_lookup = build_driver_lookup(long_df, latest_standings.drop(columns=["season"]))
    wide_df = build_wide_results(long_df, driver_lookup)

    prefix = "f1_" + "_".join(str(season) for season in seasons)
    long_path = args.output_dir / f"{prefix}_driver_results_long.csv"
    wide_path = args.output_dir / f"{prefix}_driver_results_wide.csv"
    lookup_path = args.output_dir / f"{prefix}_driver_lookup.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    driver_lookup.to_csv(lookup_path, index=False)

    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {lookup_path}")
    print(
        "Summary:",
        {
            "seasons": seasons,
            "race_rows": int(wide_df.shape[0]),
            "drivers": int(driver_lookup.shape[0]),
            "constructors": int(long_df["constructor"].nunique()),
            "track_type_counts": wide_df["track_type"].value_counts().to_dict(),
            "sprint_weekends": int(
                (
                    long_df[["season", "round", "weekend_type"]]
                    .drop_duplicates()["weekend_type"]
                    == "sprint"
                ).sum()
            ),
        },
    )


if __name__ == "__main__":
    main()
