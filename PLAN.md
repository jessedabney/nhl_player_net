# NHL Team Network — Project Roadmap

## Context
Build an interactive network graph where NHL teams are nodes and weighted edges represent the number of players two teams have shared over time. The project is greenfield (empty repo). The plan breaks work into three stages: data acquisition, cleaning/formatting, and visualization.

---

## Stage 1: Data Acquisition

**Goal:** Retrieve historical roster data for all NHL teams.

**Source:** NHL Stats API (free, no auth required)
- Base URL: `https://api-web.nhle.com/v1/`
- Key endpoints:
  - `/roster/{team}/{season}` — full roster per team per season
  - `/standings/now` or team list endpoint — get all active team abbreviations

**Approach:**
- Write a Python script (`scripts/fetch_rosters.py`) that iterates over all teams and a configurable range of seasons (e.g., 2000–2024)
- Store raw API responses as JSON in `data/raw/rosters/{season}/{team}.json`
- Keep a simple `data/raw/teams.json` with the full team list

**Deliverable:** Raw JSON roster files, one per team per season.

---

## Stage 2: Data Cleaning & Formatting

**Goal:** Transform raw rosters into a player–team edge list, then compute team–team shared-player counts.

**Steps:**
1. Parse raw JSON → flat records: `(player_id, player_name, team, season)`
2. Save as `data/processed/player_team_seasons.csv`
3. For each pair of teams, count distinct players who appeared on both:
   - Group by `player_id`, collect set of teams → generate all team-pair combinations
   - Aggregate to produce `data/processed/team_edges.csv` with columns: `team_a, team_b, shared_players`
4. Optionally filter by season range or minimum shared-player threshold

**Script:** `scripts/build_edges.py`

**Deliverable:** `team_edges.csv` — the edge list that drives the network.

---

## Stage 3: Network Visualization

**Goal:** Interactive graph where users can explore team connections.

**Tooling:** Streamlit + NetworkX + Plotly

**Features to build toward:**
- Node size ~ number of connections (or total players)
- Edge weight/thickness ~ number of shared players
- Hover tooltip: team name, top shared players
- Filter controls: season range, minimum edge weight, specific team highlight

**Entry point:** `app/app.py`

**Deliverable:** Streamlit web app showing the interactive team network.

---

## Directory Structure (proposed)
```
nhl_teamNet/
├── data/
│   ├── raw/
│   │   ├── teams.json
│   │   └── rosters/{season}/{team}.json
│   └── processed/
│       ├── player_team_seasons.csv
│       └── team_edges.csv
├── scripts/
│   ├── fetch_rosters.py
│   └── build_edges.py
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Decisions
- **Season scope:** 2000–present (~25 seasons)
- **Visualization target:** Streamlit app (interactive filters, Python-native)
- **Edge definition:** Any roster appearance — a player counts if they appeared on both teams in any season, regardless of games played
