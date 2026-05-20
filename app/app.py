"""
app.py
------
Streamlit app: interactive NHL team network graph.
Nodes = teams, edges = shared players (weighted).

Run:
    streamlit run app/app.py
"""

from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EDGES_PATH = Path(__file__).parent.parent / "data" / "processed" / "team_edges.csv"
PTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "player_team_seasons.csv"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    edges = pd.read_csv(EDGES_PATH)
    pts = pd.read_csv(PTS_PATH)
    return edges, pts


# ---------------------------------------------------------------------------
# Edge computation (runs on filtered pts so min_games works interactively)
# ---------------------------------------------------------------------------

@st.cache_data
def compute_edges(pts_filtered: pd.DataFrame) -> pd.DataFrame:
    player_teams = (
        pts_filtered.groupby("player_id")["team"]
        .apply(set)
        .reset_index()
        .rename(columns={"team": "teams"})
    )
    pair_counts: dict[tuple[str, str], set[int]] = {}
    for _, row in player_teams[player_teams["teams"].apply(len) > 1].iterrows():
        for t1, t2 in combinations(sorted(row["teams"]), 2):
            pair_counts.setdefault((t1, t2), set()).add(row["player_id"])
    rows = [
        {"team_a": t1, "team_b": t2, "shared_players": len(pids)}
        for (t1, t2), pids in pair_counts.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["team_a", "team_b", "shared_players"])
    return pd.DataFrame(rows).sort_values("shared_players", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(edges: pd.DataFrame, min_shared: int) -> nx.Graph:
    G = nx.Graph()
    for _, row in edges[edges["shared_players"] >= min_shared].iterrows():
        G.add_edge(row["team_a"], row["team_b"], weight=row["shared_players"])
    return G


def spring_layout(G: nx.Graph) -> dict:
    return nx.spring_layout(G, seed=42, weight="weight")


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------

def build_figure(
    G: nx.Graph,
    pos: dict,
    highlight_team: str | None,
    player_teams: set[str],
    edges_df: pd.DataFrame,
) -> go.Figure:
    # Build a lookup from (team_a, team_b) → shared_player_names
    names_lookup: dict[tuple[str, str], str] = {}
    if "shared_player_names" in edges_df.columns:
        for _, row in edges_df.iterrows():
            names_lookup[(row["team_a"], row["team_b"])] = row["shared_player_names"]
            names_lookup[(row["team_b"], row["team_a"])] = row["shared_player_names"]

    edge_traces = []
    midpoint_x, midpoint_y, midpoint_hover = [], [], []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data["weight"]

        is_player_edge = u in player_teams and v in player_teams
        is_highlight_edge = highlight_team and highlight_team in (u, v)

        if is_player_edge:
            color = "rgba(50,205,50,0.8)"
            width = max(1.5, weight / 10)
        elif is_highlight_edge:
            color = "rgba(255,100,0,0.7)"
            width = max(1.0, weight / 12)
        else:
            color = "rgba(200,200,200,0.25)"
            width = max(0.5, weight / 15)

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
            )
        )

        # Midpoint hover point
        names_raw = names_lookup.get((u, v), "")
        names_list = names_raw.split("|") if names_raw else []
        hover_text = f"<b>{u} ↔ {v}</b> ({weight} players)<br>" + "<br>".join(names_list)
        midpoint_x.append((x0 + x1) / 2)
        midpoint_y.append((y0 + y1) / 2)
        midpoint_hover.append(hover_text)

    edge_traces.append(
        go.Scatter(
            x=midpoint_x,
            y=midpoint_y,
            mode="markers",
            marker=dict(size=8, color="rgba(0,0,0,0)"),
            hovertext=midpoint_hover,
            hoverinfo="text",
            hoverlabel=dict(bgcolor="#1e2530", font=dict(color="white")),
        )
    )

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        weighted_degree = G.degree(node, weight="weight")
        team_count = G.degree(node)
        neighbors = sorted(G[node].items(), key=lambda kv: kv[1]["weight"], reverse=True)
        top_neighbors = "<br>".join(f"  {nb}: {d['weight']} players" for nb, d in neighbors[:5])
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"<b>{node}</b><br>Connected to {team_count} teams<br>{weighted_degree} total shared players"
            f"<br><br>Top connections:<br>{top_neighbors}"
        )
        node_size.append(12 + weighted_degree / 18)

        if node in player_teams:
            node_color.append("#2ecc71")       # green — player's team
        elif node == highlight_team:
            node_color.append("#e67e22")       # orange — highlighted team
        else:
            node_color.append("#3498db")       # default blue

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="white"),
        ),
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="NHL Team Network", layout="wide")
    st.title("NHL Team Network")
    st.caption("Nodes = teams · Edge weight = number of currently active players shared between two teams")

    if not EDGES_PATH.exists():
        st.error(
            "No data found. Run the data pipeline first:\n\n"
            "```\npython scripts/fetch_rosters.py\n"
            "python scripts/fetch_stats.py\n"
            "python scripts/build_edges.py --active-only\n```"
        )
        return

    edges_base, pts = load_data()
    has_gp = "games_played" in pts.columns and pts["games_played"].gt(0).any()

    # ---- Sidebar ----
    st.sidebar.header("Filters")

    # Games played filter
    if has_gp:
        max_gp = int(pts["games_played"].max())
        min_games = st.sidebar.slider(
            "Min games played per team",
            min_value=0,
            max_value=min(max_gp, 82),
            value=0,
            step=1,
            help="Only count a player as 'shared' if they played at least this many games for each team.",
        )
    else:
        min_games = 0

    # Recompute edges if games played filter is active
    if min_games > 0 and has_gp:
        pts_filtered = pts[pts["games_played"] >= min_games]
        edges = compute_edges(pts_filtered)
    else:
        edges = edges_base

    if edges.empty:
        st.warning("No edges with current filters — try lowering the minimum games played.")
        return

    # Min shared players filter
    max_shared = int(edges["shared_players"].max())
    min_shared = st.sidebar.slider(
        "Min shared players per edge",
        min_value=1,
        max_value=max_shared,
        value=max(1, max_shared // 4),
        step=1,
        help="Hide edges between teams that share fewer than this many players.",
    )

    # Team highlight
    all_teams = sorted(set(edges["team_a"]) | set(edges["team_b"]))
    highlight_team = st.sidebar.selectbox("Highlight team", options=["(none)"] + all_teams)
    if highlight_team == "(none)":
        highlight_team = None

    # Team comparison
    st.sidebar.markdown("---")
    st.sidebar.subheader("Team comparison")
    all_teams_sorted = sorted(set(pts["team"].unique()))
    compare_a = st.sidebar.selectbox("Team A", options=["(none)"] + all_teams_sorted, key="cmp_a")
    compare_b = st.sidebar.selectbox("Team B", options=["(none)"] + all_teams_sorted, key="cmp_b")

    # Player lookup
    st.sidebar.markdown("---")
    st.sidebar.subheader("Player lookup")
    name_filter = st.sidebar.text_input("Search player name", placeholder="e.g. Crosby")

    selected_player_id = None
    player_teams: set[str] = set()

    if name_filter.strip():
        name_lower = name_filter.strip().lower()
        pts["full_name"] = pts["first_name"] + " " + pts["last_name"]
        matches = pts[pts["full_name"].str.lower().str.contains(name_lower, na=False)]
        unique_players = (
            matches.drop_duplicates("player_id")[["player_id", "full_name"]]
            .sort_values("full_name")
        )
        if unique_players.empty:
            st.sidebar.caption("No players found.")
        else:
            options = ["(select)"] + [
                f"{row['full_name']} ({row['player_id']})"
                for _, row in unique_players.iterrows()
            ]
            choice = st.sidebar.selectbox("Matching players", options=options)
            if choice != "(select)":
                selected_player_id = int(choice.split("(")[-1].rstrip(")"))
                player_rows = pts[pts["player_id"] == selected_player_id]
                player_teams = set(player_rows["team"].unique())

    # ---- Graph ----
    G = build_graph(edges, min_shared)
    if len(G.nodes) == 0:
        st.warning("No edges meet the current filter — try lowering the minimum shared players.")
        return

    pos = spring_layout(G)
    fig = build_figure(G, pos, highlight_team, player_teams & set(G.nodes), edges)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Metrics ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Teams shown", len(G.nodes))
    col2.metric("Connections shown", len(G.edges))
    col3.metric("Min shared players", min_shared)

    # ---- Player career panel ----
    if selected_player_id is not None:
        player_name = pts[pts["player_id"] == selected_player_id]["full_name"].iloc[0]
        st.subheader(f"Career path — {player_name}")
        career = (
            pts[pts["player_id"] == selected_player_id]
            .sort_values("season")[["season", "team", "games_played"]]
            .rename(columns={"season": "Season", "team": "Team", "games_played": "GP"})
        )
        career["Season"] = career["Season"].str[:4]
        if not has_gp or career["GP"].eq(0).all():
            career = career.drop(columns=["GP"])
        st.dataframe(career, use_container_width=True, hide_index=True)

    # ---- Team comparison table ----
    if compare_a != "(none)" and compare_b != "(none)" and compare_a != compare_b:
        st.subheader(f"Shared players — {compare_a} & {compare_b}")
        pts["full_name"] = pts["first_name"] + " " + pts["last_name"]
        a_players = pts[pts["team"] == compare_a].groupby("player_id")
        b_players = pts[pts["team"] == compare_b].groupby("player_id")
        shared_ids = set(a_players.groups) & set(b_players.groups)

        if not shared_ids:
            st.caption("No shared players found between these two teams.")
        else:
            rows = []
            for pid in sorted(shared_ids):
                a_rows = pts[(pts["player_id"] == pid) & (pts["team"] == compare_a)]
                b_rows = pts[(pts["player_id"] == pid) & (pts["team"] == compare_b)]
                name = a_rows["full_name"].iloc[0]
                position = a_rows["position"].iloc[0]
                a_seasons = ", ".join(sorted(a_rows["season"].astype(str).str[:4]))
                b_seasons = ", ".join(sorted(b_rows["season"].astype(str).str[:4]))
                a_gp = int(a_rows["games_played"].sum()) if has_gp else None
                b_gp = int(b_rows["games_played"].sum()) if has_gp else None
                current_team = a_rows["current_team"].iloc[0] if "current_team" in a_rows.columns else None
                row = {
                    "Player": name,
                    "Pos": position,
                    "Current Team": current_team if pd.notna(current_team) else "—",
                    f"{compare_a} seasons": a_seasons,
                    f"{compare_b} seasons": b_seasons,
                }
                if has_gp:
                    row[f"{compare_a} GP"] = a_gp
                    row[f"{compare_b} GP"] = b_gp
                rows.append(row)

            comparison_df = pd.DataFrame(rows).sort_values("Player")
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ---- Top connections table ----
    st.subheader("Top team connections")
    top = edges.head(20)[["team_a", "team_b", "shared_players"]].copy()
    top.columns = ["Team A", "Team B", "Shared Players"]
    st.dataframe(top, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
