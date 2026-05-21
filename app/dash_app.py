"""
dash_app.py
-----------
Dash + Cytoscape app: interactive NHL team network graph.
Nodes = teams, edges = shared players (weighted, directed).

Run:
    python app/dash_app.py
"""

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html

EDGES_PATH = Path(__file__).parent.parent / "data" / "processed" / "team_edges.csv"
PTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "player_team_seasons.csv"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    edges = pd.read_csv(EDGES_PATH)
    pts = pd.read_csv(PTS_PATH, dtype={"season": str})
    pts["full_name"] = pts["first_name"] + " " + pts["last_name"]
    return edges, pts


EDGES_DF, PTS_DF = load_data()
HAS_GP = "games_played" in PTS_DF.columns and PTS_DF["games_played"].gt(0).any()
ALL_TEAMS = sorted(set(EDGES_DF["from_team"]) | set(EDGES_DF["to_team"]))
MAX_GP = int(PTS_DF["games_played"].max()) if HAS_GP else 0

# Team city coordinates (lat, lon)
TEAM_COORDS = {
    "ANA": (33.81, -117.88),   # Anaheim
    "ARI": (33.53, -112.07),   # Glendale/Tempe (historical)
    "ATL": (33.76, -84.39),    # Atlanta (historical)
    "BOS": (42.37, -71.06),    # Boston
    "BUF": (42.87, -78.88),    # Buffalo
    "CAR": (35.80, -78.72),    # Raleigh
    "CBJ": (39.97, -83.01),    # Columbus
    "CGY": (51.04, -114.06),   # Calgary
    "CHI": (41.88, -87.67),    # Chicago
    "COL": (39.75, -105.00),   # Denver
    "DAL": (32.79, -96.81),    # Dallas
    "DET": (42.34, -83.06),    # Detroit
    "EDM": (53.55, -113.50),   # Edmonton
    "FLA": (26.16, -80.33),    # Sunrise
    "LAK": (34.04, -118.27),   # Los Angeles
    "MIN": (44.95, -93.10),    # St. Paul
    "MTL": (45.50, -73.57),    # Montreal
    "NJD": (40.73, -74.07),    # Newark
    "NSH": (36.16, -86.78),    # Nashville
    "NYI": (40.72, -73.73),    # Elmont (UBS Arena)
    "NYR": (40.75, -73.99),    # Manhattan (MSG)
    "OTT": (45.30, -75.93),    # Ottawa
    "PHI": (39.90, -75.17),    # Philadelphia
    "PHX": (33.45, -112.07),   # Phoenix (historical)
    "PIT": (40.44, -80.00),    # Pittsburgh
    "SEA": (47.62, -122.35),   # Seattle
    "SJS": (37.33, -121.90),   # San Jose
    "STL": (38.63, -90.20),    # St. Louis
    "TBL": (27.94, -82.45),    # Tampa
    "TOR": (43.64, -79.38),    # Toronto
    "UTA": (40.77, -111.90),   # Salt Lake City
    "VAN": (49.28, -123.11),   # Vancouver
    "VGK": (36.10, -115.18),   # Las Vegas
    "WPG": (49.89, -97.14),    # Winnipeg
    "WSH": (38.90, -77.02),    # Washington DC
}


def _geo_layout(G):
    raw = {}
    for node in G.nodes():
        lat, lon = TEAM_COORDS.get(node, (39.0, -98.0))
        raw[node] = (lon, -lat)
    if not raw:
        return raw
    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1
    scale = max(x_range, y_range)
    return {
        node: ((x - (x_min + x_max) / 2) / scale * 2, (y - (y_min + y_max) / 2) / scale * 2)
        for node, (x, y) in raw.items()
    }


LAYOUTS = {
    "Map": _geo_layout,
    "Circular": lambda G: nx.circular_layout(G),
    "Spring": lambda G: nx.spring_layout(G, seed=42, weight="weight", k=3, iterations=100),
}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_graph(edges: pd.DataFrame, min_shared: int) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in edges[edges["shared_players"] >= min_shared].iterrows():
        G.add_edge(row["from_team"], row["to_team"], weight=row["shared_players"])
    return G


def compute_edges_filtered(pts_filtered: pd.DataFrame) -> pd.DataFrame:
    if "current_team" not in pts_filtered.columns or pts_filtered["current_team"].isna().all():
        return pd.DataFrame(columns=["from_team", "to_team", "shared_players"])
    current_team_lookup = (
        pts_filtered.dropna(subset=["current_team"])
        .drop_duplicates("player_id")
        .set_index("player_id")["current_team"]
        .to_dict()
    )
    player_all_teams = pts_filtered.groupby("player_id")["team"].apply(set).to_dict()
    pair_counts: dict[tuple[str, str], set[int]] = {}
    for player_id, current in current_team_lookup.items():
        historical = player_all_teams.get(player_id, set()) - {current}
        for h in historical:
            pair_counts.setdefault((h, current), set()).add(player_id)
    rows = [
        {"from_team": src, "to_team": dst, "shared_players": len(pids)}
        for (src, dst), pids in pair_counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["from_team", "to_team", "shared_players"])
    return pd.DataFrame(rows).sort_values("shared_players", ascending=False).reset_index(drop=True)


def graph_to_cytoscape(G: nx.DiGraph, pos: dict, highlight_team: str | None = None) -> list[dict]:
    """Convert a NetworkX DiGraph + positions into Cytoscape elements."""
    elements = []
    max_weight = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)

    for node in G.nodes():
        x, y = pos[node]
        weighted_deg = G.degree(node, weight="weight")
        elements.append({
            "data": {
                "id": node,
                "label": node,
                "weighted_degree": weighted_deg,
                "size": max(30, 20 + weighted_deg / 4),
            },
            "position": {"x": x * 500, "y": y * 500},
            "classes": (
                "highlighted" if node == highlight_team
                else "neighbor" if highlight_team and G.has_node(highlight_team) and (
                    G.has_edge(highlight_team, node) or G.has_edge(node, highlight_team)
                )
                else ""
            ),
        })

    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        elements.append({
            "data": {
                "source": u,
                "target": v,
                "weight": weight,
                "norm_weight": max(1, weight * 8 / max_weight),
            },
            "classes": (
                "highlighted-edge" if highlight_team and highlight_team in (u, v)
                else ""
            ),
        })

    return elements


# ---------------------------------------------------------------------------
# Cytoscape stylesheet
# ---------------------------------------------------------------------------

CYTO_STYLESHEET = [
    # Default node
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "width": "data(size)",
            "height": "data(size)",
            "background-color": "#3498db",
            "color": "#ffffff",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            "font-weight": "bold",
            "border-width": 2,
            "border-color": "#ffffff",
            "text-outline-width": 2,
            "text-outline-color": "#1a1a2e",
        },
    },
    # Highlighted node
    {
        "selector": "node.highlighted",
        "style": {
            "background-color": "#e67e22",
            "border-color": "#ffffff",
            "border-width": 3,
        },
    },
    # Neighbor of highlighted node
    {
        "selector": "node.neighbor",
        "style": {
            "background-color": "#3498db",
        },
    },
    # Default edge
    {
        "selector": "edge",
        "style": {
            "width": "data(norm_weight)",
            "line-color": "#555555",
            "target-arrow-color": "#555555",
            "target-arrow-shape": "triangle",
            "arrow-scale": 1.2,
            "curve-style": "bezier",
            "opacity": 0.6,
        },
    },
    # Highlighted edge
    {
        "selector": "edge.highlighted-edge",
        "style": {
            "line-color": "#e67e22",
            "target-arrow-color": "#e67e22",
            "opacity": 0.9,
        },
    },
]


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="NHL Team Network",
)

sidebar = dbc.Card(
    [
        html.H5("Filters", className="mb-3"),

        dbc.Label("Layout"),
        dbc.RadioItems(
            id="layout-selector",
            options=[{"label": k, "value": k} for k in LAYOUTS],
            value="Map",
            inline=True,
            className="mb-3",
        ),

        dbc.Label("Min games played per team"),
        dcc.Slider(
            id="min-games-slider",
            min=0,
            max=min(MAX_GP, 82) if HAS_GP else 0,
            value=0,
            step=1,
            marks={0: "0", 20: "20", 40: "40", 60: "60", 82: "82"} if HAS_GP else {0: "0"},
            tooltip={"placement": "bottom"},
            disabled=not HAS_GP,
            className="mb-3",
        ),

        dbc.Label("Highlight team"),
        dcc.Dropdown(
            id="highlight-team",
            options=[{"label": t, "value": t} for t in ALL_TEAMS],
            value=None,
            placeholder="(none)",
            className="mb-3",
        ),

        html.Hr(),
        html.H5("Team comparison", className="mb-3"),

        dbc.Label("Team A"),
        dcc.Dropdown(
            id="compare-a",
            options=[{"label": t, "value": t} for t in ALL_TEAMS],
            value=None,
            placeholder="(none)",
            className="mb-3",
        ),
        dbc.Label("Team B"),
        dcc.Dropdown(
            id="compare-b",
            options=[{"label": t, "value": t} for t in ALL_TEAMS],
            value=None,
            placeholder="(none)",
            className="mb-3",
        ),

        html.Hr(),
        html.H5("Player lookup", className="mb-3"),

        dbc.Input(
            id="player-search",
            placeholder="e.g. Crosby",
            type="text",
            className="mb-2",
        ),
        dcc.Dropdown(
            id="player-select",
            options=[],
            value=None,
            placeholder="(search above)",
            className="mb-3",
        ),
    ],
    body=True,
    className="bg-dark",
)

main_content = html.Div([
    cyto.Cytoscape(
        id="team-graph",
        elements=[],
        stylesheet=CYTO_STYLESHEET,
        layout={"name": "preset"},
        style={"width": "100%", "height": "700px", "backgroundColor": "#0e1117"},
        minZoom=0.3,
        maxZoom=3.0,
        userPanningEnabled=True,
        userZoomingEnabled=True,
        boxSelectionEnabled=False,
    ),
    html.Div(id="metrics-row", className="mt-3"),
    html.Div(id="career-panel", className="mt-3"),
    html.Div(id="comparison-panel", className="mt-3"),
    html.Div([
        html.H5("Top team connections"),
        dash_table.DataTable(
            id="top-connections-table",
            style_header={"backgroundColor": "#1a1a2e", "color": "white", "fontWeight": "bold"},
            style_cell={"backgroundColor": "#0e1117", "color": "white", "border": "1px solid #333"},
            page_size=20,
        ),
    ], className="mt-3"),
])

app.layout = dbc.Container(
    [
        html.H2("NHL Team Network", className="mt-3 mb-1"),
        html.P(
            "Nodes = teams. Arrows point from a player's former team to their current team. "
            "Edge weight = number of active players.",
            className="text-muted mb-3",
        ),
        dbc.Row([
            dbc.Col(sidebar, width=3),
            dbc.Col(main_content, width=9),
        ]),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("team-graph", "elements"),
    Output("metrics-row", "children"),
    Output("top-connections-table", "data"),
    Output("top-connections-table", "columns"),
    Input("layout-selector", "value"),
    Input("min-games-slider", "value"),
    Input("highlight-team", "value"),
)
def update_graph(layout_name, min_games, highlight_team):
    min_shared = 1
    # Recompute edges if min_games filter is active
    if min_games and min_games > 0 and HAS_GP:
        pts_filtered = PTS_DF[PTS_DF["games_played"] >= min_games]
        edges = compute_edges_filtered(pts_filtered)
    else:
        edges = EDGES_DF

    if edges.empty:
        return [], html.P("No edges with current filters."), [], []

    G = build_graph(edges, min_shared)
    if len(G.nodes) == 0:
        return [], html.P("No edges meet the current filter."), [], []

    # Filter to highlight team neighborhood
    if highlight_team and highlight_team in G:
        keep = {highlight_team} | set(G.predecessors(highlight_team)) | set(G.successors(highlight_team))
        G = G.subgraph(keep).copy()
        G.remove_edges_from([(u, v) for u, v in G.edges() if highlight_team not in (u, v)])

    pos = LAYOUTS[layout_name](G)
    elements = graph_to_cytoscape(G, pos, highlight_team)

    # Metrics
    metrics = dbc.Row([
        dbc.Col(dbc.Card([
            html.H4(len(G.nodes), className="text-center"),
            html.P("Teams shown", className="text-center text-muted mb-0"),
        ], body=True, className="bg-dark"), width=6),
        dbc.Col(dbc.Card([
            html.H4(len(G.edges), className="text-center"),
            html.P("Connections shown", className="text-center text-muted mb-0"),
        ], body=True, className="bg-dark"), width=6),
    ])

    # Top connections table
    top = edges[edges["shared_players"] >= min_shared].head(20)
    table_data = top[["from_team", "to_team", "shared_players"]].to_dict("records")
    table_cols = [
        {"name": "From", "id": "from_team"},
        {"name": "To (current)", "id": "to_team"},
        {"name": "Shared Players", "id": "shared_players"},
    ]

    return elements, metrics, table_data, table_cols


@callback(
    Output("player-select", "options"),
    Output("player-select", "value"),
    Input("player-search", "value"),
)
def update_player_search(search_text):
    if not search_text or not search_text.strip():
        return [], None
    name_lower = search_text.strip().lower()
    matches = PTS_DF[PTS_DF["full_name"].str.lower().str.contains(name_lower, na=False)]
    unique = matches.drop_duplicates("player_id")[["player_id", "full_name"]].sort_values("full_name")
    if unique.empty:
        return [{"label": "No players found", "value": "", "disabled": True}], None
    options = [
        {"label": row["full_name"], "value": int(row["player_id"])}
        for _, row in unique.iterrows()
    ]
    return options, None


@callback(
    Output("career-panel", "children"),
    Input("player-select", "value"),
    Input("team-graph", "tapNodeData"),
)
def update_career_panel(selected_player_id, tap_data):
    # If a player is selected via search, show their career
    if selected_player_id:
        player_rows = PTS_DF[PTS_DF["player_id"] == selected_player_id]
        if player_rows.empty:
            return []
        player_name = player_rows["full_name"].iloc[0]
        career = (
            player_rows.sort_values("season")[["season", "team", "games_played"]]
            .copy()
        )
        career["season"] = career["season"].str[:4]
        cols = [
            {"name": "Season", "id": "season"},
            {"name": "Team", "id": "team"},
        ]
        if HAS_GP and not career["games_played"].eq(0).all():
            cols.append({"name": "GP", "id": "games_played"})

        return [
            html.H5(f"Career path -- {player_name}"),
            dash_table.DataTable(
                data=career.to_dict("records"),
                columns=cols,
                style_header={"backgroundColor": "#1a1a2e", "color": "white", "fontWeight": "bold"},
                style_cell={"backgroundColor": "#0e1117", "color": "white", "border": "1px solid #333"},
            ),
        ]
    return []


@callback(
    Output("comparison-panel", "children"),
    Input("compare-a", "value"),
    Input("compare-b", "value"),
)
def update_comparison(compare_a, compare_b):
    if not compare_a or not compare_b or compare_a == compare_b:
        return []

    a_ids = set(PTS_DF[PTS_DF["team"] == compare_a]["player_id"])
    b_ids = set(PTS_DF[PTS_DF["team"] == compare_b]["player_id"])
    shared_ids = a_ids & b_ids

    if not shared_ids:
        return html.P(f"No shared players between {compare_a} and {compare_b}.")

    rows = []
    for pid in sorted(shared_ids):
        a_rows = PTS_DF[(PTS_DF["player_id"] == pid) & (PTS_DF["team"] == compare_a)]
        b_rows = PTS_DF[(PTS_DF["player_id"] == pid) & (PTS_DF["team"] == compare_b)]
        name = a_rows["full_name"].iloc[0]
        position = a_rows["position"].iloc[0]
        a_seasons = ", ".join(sorted(a_rows["season"].str[:4]))
        b_seasons = ", ".join(sorted(b_rows["season"].str[:4]))
        current_team = a_rows["current_team"].iloc[0] if "current_team" in a_rows.columns and pd.notna(a_rows["current_team"].iloc[0]) else "-"
        row = {
            "Player": name,
            "Pos": position,
            "Current Team": current_team,
            f"{compare_a} seasons": a_seasons,
            f"{compare_b} seasons": b_seasons,
        }
        if HAS_GP:
            row[f"{compare_a} GP"] = int(a_rows["games_played"].sum())
            row[f"{compare_b} GP"] = int(b_rows["games_played"].sum())
        rows.append(row)

    comparison_df = pd.DataFrame(rows).sort_values("Player")
    cols = [{"name": c, "id": c} for c in comparison_df.columns]

    return [
        html.H5(f"Shared players -- {compare_a} & {compare_b}"),
        dash_table.DataTable(
            data=comparison_df.to_dict("records"),
            columns=cols,
            sort_action="native",
            style_header={"backgroundColor": "#1a1a2e", "color": "white", "fontWeight": "bold"},
            style_cell={"backgroundColor": "#0e1117", "color": "white", "border": "1px solid #333"},
        ),
    ]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
