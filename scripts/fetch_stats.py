"""
fetch_stats.py
--------------
Fetch per-player games played and TOI from the NHL club-stats API for every
team/season combo that exists in data/raw/rosters/ (excluding the active-season
filter directory). Saves raw JSON to data/raw/stats/{season}/{team}.json.

Usage:
    python scripts/fetch_stats.py
"""

import json
import time
from pathlib import Path

import requests

BASE_URL = "https://api-web.nhle.com/v1"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
ACTIVE_SEASON = "20252026"
SLEEP_BETWEEN_REQUESTS = 0.3


def fetch_club_stats(team: str, season: str, session: requests.Session) -> dict | None:
    url = f"{BASE_URL}/club-stats/{team}/{season}/2"
    resp = session.get(url, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def main() -> None:
    rosters_dir = RAW_DIR / "rosters"
    stats_dir = RAW_DIR / "stats"

    # Collect all team/season combos from existing roster files, skip active season
    combos = [
        (p.parent.name, p.stem)
        for p in rosters_dir.glob("*/*.json")
        if p.parent.name != ACTIVE_SEASON
    ]
    combos.sort()
    print(f"Fetching club stats for {len(combos)} team/season combos…")

    session = requests.Session()
    fetched = skipped = not_found = errors = 0

    for season, team in combos:
        out_path = stats_dir / season / f"{team}.json"
        if out_path.exists():
            skipped += 1
            continue

        try:
            data = fetch_club_stats(team, season, session)
            if data is None:
                not_found += 1
            else:
                save_json(data, out_path)
                n_skaters = len(data.get("skaters", []))
                n_goalies = len(data.get("goalies", []))
                fetched += 1
                print(f"  {season}/{team} — {n_skaters} skaters, {n_goalies} goalies")
        except Exception as exc:
            errors += 1
            print(f"  ERROR {season}/{team}: {exc}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"\nDone. fetched={fetched}, skipped={skipped}, not_found={not_found}, errors={errors}")


if __name__ == "__main__":
    main()
