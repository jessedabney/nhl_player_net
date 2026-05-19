"""
fetch_rosters.py
----------------
Fetch NHL roster data from the NHL Stats API for all teams and seasons
from 2000-01 through 2024-25, saving raw JSON to data/raw/rosters/{season}/{team}.json.

Usage:
    python scripts/fetch_rosters.py
    python scripts/fetch_rosters.py --start-season 20102011 --end-season 20232024
"""

import argparse
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://api-web.nhle.com/v1"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
TEAMS_FILE = RAW_DIR / "teams.json"
SLEEP_BETWEEN_REQUESTS = 0.3  # seconds — be polite to the API

# All team abbreviations that have existed in the NHL from 2000-01 onward.
# Teams that no longer exist are kept so historical rosters can be fetched.
ALL_TEAMS = [
    "ANA",  # Anaheim Ducks (formerly Mighty Ducks of Anaheim)
    "ATL",  # Atlanta Thrashers (1999–2011, became WPG)
    "BOS",  # Boston Bruins
    "BUF",  # Buffalo Sabres
    "CGY",  # Calgary Flames
    "CAR",  # Carolina Hurricanes
    "CHI",  # Chicago Blackhawks
    "COL",  # Colorado Avalanche
    "CBJ",  # Columbus Blue Jackets (joined 2000–01)
    "DAL",  # Dallas Stars
    "DET",  # Detroit Red Wings
    "EDM",  # Edmonton Oilers
    "FLA",  # Florida Panthers
    "LAK",  # Los Angeles Kings
    "MIN",  # Minnesota Wild (joined 2000–01)
    "MTL",  # Montreal Canadiens
    "NSH",  # Nashville Predators
    "NJD",  # New Jersey Devils
    "NYI",  # New York Islanders
    "NYR",  # New York Rangers
    "OTT",  # Ottawa Senators
    "PHI",  # Philadelphia Flyers
    "PHX",  # Phoenix Coyotes (became ARI in 2014–15)
    "ARI",  # Arizona Coyotes (2014–2024, relocated to Utah)
    "PIT",  # Pittsburgh Penguins
    "SJS",  # San Jose Sharks
    "SEA",  # Seattle Kraken (joined 2021–22)
    "STL",  # St. Louis Blues
    "TBL",  # Tampa Bay Lightning
    "TOR",  # Toronto Maple Leafs
    "UTA",  # Utah Hockey Club (joined 2024–25)
    "VAN",  # Vancouver Canucks
    "VGK",  # Vegas Golden Knights (joined 2017–18)
    "WSH",  # Washington Capitals
    "WPG",  # Winnipeg Jets (returned 2011–12)
]

DEFAULT_START = 20002001
DEFAULT_END = 20242025


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seasons_in_range(start: int, end: int) -> list[str]:
    """Return a list of season strings (e.g. '20002001') between start and end."""
    seasons = []
    start_year = int(str(start)[:4])
    end_year = int(str(end)[:4])
    for y in range(start_year, end_year + 1):
        seasons.append(f"{y}{y + 1}")
    return seasons


def fetch_roster(team: str, season: str, session: requests.Session) -> dict | None:
    """Fetch a roster from the NHL API. Returns None on 404 (team/season doesn't exist)."""
    url = f"{BASE_URL}/roster/{team}/{season}"
    resp = session.get(url, timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(start_season: int, end_season: int) -> None:
    seasons = seasons_in_range(start_season, end_season)
    print(f"Fetching rosters for {len(ALL_TEAMS)} teams × {len(seasons)} seasons "
          f"({start_season}–{end_season})")

    # Save the canonical team list once
    TEAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not TEAMS_FILE.exists():
        with open(TEAMS_FILE, "w") as f:
            json.dump(ALL_TEAMS, f, indent=2)
        print(f"Wrote team list → {TEAMS_FILE}")

    session = requests.Session()
    fetched = skipped = errors = not_found = 0

    for season in seasons:
        for team in ALL_TEAMS:
            out_path = RAW_DIR / "rosters" / season / f"{team}.json"

            if out_path.exists():
                skipped += 1
                continue

            try:
                data = fetch_roster(team, season, session)
                if data is None:
                    not_found += 1
                else:
                    save_json(data, out_path)
                    fetched += 1
                    print(f"  {season}/{team} — {sum(len(data.get(g, [])) for g in ('forwards','defensemen','goalies'))} players")
            except Exception as exc:
                errors += 1
                print(f"  ERROR {season}/{team}: {exc}")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"\nDone. fetched={fetched}, skipped={skipped}, not_found={not_found}, errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=DEFAULT_START,
                        help="First season as 8-digit int, e.g. 20002001")
    parser.add_argument("--end-season", type=int, default=DEFAULT_END,
                        help="Last season as 8-digit int, e.g. 20242025")
    args = parser.parse_args()
    main(args.start_season, args.end_season)
