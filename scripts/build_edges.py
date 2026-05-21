"""
build_edges.py
--------------
Parse raw roster JSON files and produce two CSVs:

  data/processed/player_team_seasons.csv  — one row per (player, team, season)
                                            includes games_played and avg_toi_sec if stats are available
  data/processed/team_edges.csv           — one row per team pair, with shared player count

Usage:
    python scripts/build_edges.py                # full historical
    python scripts/build_edges.py --active-only  # current players only (requires 20252026 rosters)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ACTIVE_SEASON = "20252026"


# ---------------------------------------------------------------------------
# Parse raw JSON
# ---------------------------------------------------------------------------

def parse_roster(data: dict, team: str, season: str) -> list[dict]:
    """Flatten a roster API response into a list of player records."""
    records = []
    for group in ("forwards", "defensemen", "goalies"):
        for player in data.get(group, []):
            records.append({
                "player_id": player.get("id"),
                "first_name": player.get("firstName", {}).get("default", ""),
                "last_name": player.get("lastName", {}).get("default", ""),
                "position": player.get("positionCode", ""),
                "team": team,
                "season": season,
            })
    return records


def get_active_player_map() -> dict[int, str]:
    """Return a player_id → current team mapping from the (20252026) season rosters."""
    active_dir = RAW_DIR / "rosters" / ACTIVE_SEASON
    if not active_dir.exists():
        raise FileNotFoundError(
            f"No {ACTIVE_SEASON} roster data found. "
            f"Run: python scripts/fetch_rosters.py --start-season {ACTIVE_SEASON} --end-season {ACTIVE_SEASON}"
        )
    active_map: dict[int, str] = {}
    for path in active_dir.glob("*.json"):
        team = path.stem
        with open(path) as f:
            data = json.load(f)
        for group in ("forwards", "defensemen", "goalies"):
            for player in data.get(group, []):
                pid = player.get("id")
                if pid is not None:
                    active_map[int(pid)] = team
    print(f"Active players in {ACTIVE_SEASON}: {len(active_map)}")
    return active_map


def build_player_team_seasons(active_only: bool = False) -> pd.DataFrame:
    rosters_dir = RAW_DIR / "rosters"
    if not rosters_dir.exists():
        raise FileNotFoundError(f"No roster data found at {rosters_dir}. Run fetch_rosters.py first.")

    all_records = []
    json_files = list(rosters_dir.glob("*/*.json"))
    print(f"Parsing {len(json_files)} roster files…")

    for path in json_files:
        season = path.parent.name
        team = path.stem
        with open(path) as f:
            data = json.load(f)
        all_records.extend(parse_roster(data, team, season))

    df = pd.DataFrame(all_records)
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)
    df = df.drop_duplicates(subset=["player_id", "team", "season"])

    if active_only:
        active_map = get_active_player_map()
        before = len(df)
        df = df[df["player_id"].isin(active_map)]
        df["current_team"] = df["player_id"].map(active_map)
        print(f"Filtered to active players: {before:,} → {len(df):,} records")
    else:
        df["current_team"] = None

    # Merge in games_played and avg_toi_sec if stats data is available
    stats = load_stats()
    if not stats.empty:
        df = df.merge(stats, on=["player_id", "team", "season"], how="left")
        df["games_played"] = df["games_played"].fillna(0).astype(int)
        print(f"Merged stats: {df['games_played'].gt(0).sum():,} records have games played data")
    else:
        df["games_played"] = 0
        df["avg_toi_sec"] = None

    return df


# ---------------------------------------------------------------------------
# Merge stats (games played, TOI)
# ---------------------------------------------------------------------------

def load_stats() -> pd.DataFrame:
    """
    Parse all club-stats JSON files into a DataFrame with columns:
    player_id, team, season, games_played, avg_toi_sec
    """
    stats_dir = RAW_DIR / "stats"
    if not stats_dir.exists():
        return pd.DataFrame(columns=["player_id", "team", "season", "games_played", "avg_toi_sec"])

    records = []
    for path in stats_dir.glob("*/*.json"):
        season = path.parent.name
        team = path.stem
        with open(path) as f:
            data = json.load(f)
        for player in data.get("skaters", []):
            records.append({
                "player_id": player.get("playerId"),
                "team": team,
                "season": season,
                "games_played": player.get("gamesPlayed", 0),
                "avg_toi_sec": player.get("avgTimeOnIcePerGame"),
            })
        for goalie in data.get("goalies", []):
            records.append({
                "player_id": goalie.get("playerId"),
                "team": team,
                "season": season,
                "games_played": goalie.get("gamesPlayed", 0),
                "avg_toi_sec": goalie.get("avgTimeOnIcePerGame"),
            })

    if not records:
        return pd.DataFrame(columns=["player_id", "team", "season", "games_played", "avg_toi_sec"])

    df = pd.DataFrame(records)
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Build team-pair edge list
# ---------------------------------------------------------------------------

def build_team_edges(pts: pd.DataFrame) -> pd.DataFrame:
    """
    For each currently active player, create edges between their current team
    and every historical team they appeared on. This anchors the network to
    current rosters: an edge A–B means a player *currently on* A (or B) has
    previously played for B (or A).

    Requires the current_team column to be populated (run with --active-only).
    Returns a DataFrame with columns: team_a, team_b, shared_players, shared_player_names.
    """
    if "current_team" not in pts.columns or pts["current_team"].isna().all():
        raise ValueError(
            "current_team column is missing or empty. "
            "Run: python scripts/build_edges.py --active-only"
        )

    # player_id → full name lookup
    name_lookup = (
        (pts["first_name"] + " " + pts["last_name"])
        .groupby(pts["player_id"])
        .first()
        .to_dict()
    )

    # player_id → current team (active players only)
    current_team_lookup = (
        pts.dropna(subset=["current_team"])
        .drop_duplicates("player_id")
        .set_index("player_id")["current_team"]
        .to_dict()
    )

    # player_id → all teams ever
    player_all_teams = pts.groupby("player_id")["team"].apply(set).to_dict()

    print(f"Active players with current team: {len(current_team_lookup)}")

    pair_counts: dict[tuple[str, str], set[int]] = {}
    for player_id, current in current_team_lookup.items():
        historical = player_all_teams.get(player_id, set()) - {current}
        for h in historical:
            key = tuple(sorted([current, h]))
            pair_counts.setdefault(key, set()).add(player_id)

    multi_team_players = sum(1 for pid, ct in current_team_lookup.items()
                             if len(player_all_teams.get(pid, set()) - {ct}) > 0)
    print(f"Active players with 2+ teams in career: {multi_team_players}")

    rows = [
        {
            "team_a": t1,
            "team_b": t2,
            "shared_players": len(pids),
            "shared_player_names": "|".join(
                sorted(name_lookup.get(pid, str(pid)) for pid in pids)
            ),
        }
        for (t1, t2), pids in pair_counts.items()
    ]
    edges = pd.DataFrame(rows).sort_values("shared_players", ascending=False).reset_index(drop=True)
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--active-only",
        action="store_true",
        help=f"Limit edges to players currently on a {ACTIVE_SEASON} roster",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pts = build_player_team_seasons(active_only=args.active_only)
    pts_path = PROCESSED_DIR / "player_team_seasons.csv"
    pts.to_csv(pts_path, index=False)
    print(f"Wrote {len(pts):,} player-team-season records → {pts_path}")

    edges = build_team_edges(pts)
    edges_path = PROCESSED_DIR / "team_edges.csv"
    edges.to_csv(edges_path, index=False)
    print(f"Wrote {len(edges):,} team-pair edges → {edges_path}")
    print(f"\nTop 10 most-connected team pairs:")
    print(edges.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
