"""
app.py
------
Streamlit app: interactive NHL team network graph.
Nodes = teams, edges = shared players (weighted).

Run:
    streamlit run app/app.py
"""

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
    pts = pd.read_csv(PTS_PATH, dtype={"season": str})
    return edges, pts


# ---------------------------------------------------------------------------
# Edge computation (runs on filtered pts so min_games works interactively)
# ---------------------------------------------------------------------------

@st.cache_data
def compute_edges(pts_filtered: pd.DataFrame) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(edges: pd.DataFrame, min_shared: int) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in edges[edges["shared_players"] >= min_shared].iterrows():
        G.add_edge(row["from_team"], row["to_team"], weight=row["shared_players"])
    return G


LAYOUTS = {
    "Spectral":       lambda G: nx.spectral_layout(G, weight="weight"),
    "Spring":         lambda G: nx.spring_layout(G, seed=42, weight="weight", k=3, iterations=100),
    "Circular":       lambda G: nx.circular_layout(G),
    "Kamada-Kawai":   lambda G: nx.kamada_kawai_layout(G, weight="weight"),
    "Shell":          lambda G: nx.shell_layout(G),
}

def get_layout(G: nx.Graph, name: str) -> dict:
    return LAYOUTS[name](G)


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
    # Build a lookup from (from_team, to_team) → shared_player_names
    names_lookup: dict[tuple[str, str], str] = {}
    if "shared_player_names" in edges_df.columns:
        for _, row in edges_df.iterrows():
            names_lookup[(row["from_team"], row["to_team"])] = row["shared_player_names"]

    # Pre-compute node sizes for arrowhead standoff
    node_sizes_map = {node: 50 + G.degree(node, weight="weight") / 8 for node in G.nodes()}

    edge_traces = []
    midpoint_x, midpoint_y, midpoint_hover = [], [], []
    annotations = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data["weight"]

        is_player_edge = u in player_teams and v in player_teams
        is_highlight_edge = highlight_team and highlight_team in (u, v)

        if is_player_edge:
            color = "rgba(50,205,50,0.8)"
            width = max(7.5, weight / 2)
        elif is_highlight_edge:
            color = "rgba(255,100,0,0.7)"
            width = max(5.0, weight * 5 / 12)
        else:
            color = "rgba(200,200,200,0.25)"
            width = max(2.5, weight / 3)

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
            )
        )

        # Arrowhead pointing at the target node
        standoff = node_sizes_map.get(v, 60) / 2
        annotations.append(dict(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=max(1.5, width * 0.4),
            arrowcolor=color,
            standoff=standoff,
            text="",
        ))

        # Midpoint hover point
        names_raw = names_lookup.get((u, v), "")
        names_list = names_raw.split("|") if names_raw else []
        hover_text = f"<b>{u} → {v}</b> ({weight} players)<br>" + "<br>".join(names_list)
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
        # Aggregate weights across both directions for the tooltip
        neighbor_weights: dict[str, int] = {}
        for nb, d in G[node].items():  # outgoing
            neighbor_weights[nb] = neighbor_weights.get(nb, 0) + d["weight"]
        for nb in G.predecessors(node):  # incoming
            neighbor_weights[nb] = neighbor_weights.get(nb, 0) + G[nb][node]["weight"]
        neighbors = sorted(neighbor_weights.items(), key=lambda kv: kv[1], reverse=True)
        top_neighbors = "<br>".join(f"  {nb}: {w} players" for nb, w in neighbors[:5])
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"<b>{node}</b><br>Connected to {team_count} teams<br>{weighted_degree} total shared players"
            f"<br><br>Top connections:<br>{top_neighbors}"
        )
        node_size.append(50 + weighted_degree / 8)

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
            opacity=1.0,
            line=dict(width=2, color="white"),
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
            height=1000,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            annotations=annotations,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="NHL Team Network", layout="wide")
    st.title("NHL Team Network")
    st.caption("Nodes = teams · Arrows point from a player's former team to their current team · Edge weight = number of active players")

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

    # Layout selector
    layout_name = st.sidebar.selectbox("Layout", options=list(LAYOUTS.keys()), index=0)

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
    all_teams = sorted(set(edges["from_team"]) | set(edges["to_team"]))
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

    pos = get_layout(G, layout_name)
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
        career["Season"] = career["Season"].astype(str).str[:4]
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
                a_seasons = ", ".join(sorted(a_rows["season"].str[:4]))
                b_seasons = ", ".join(sorted(b_rows["season"].str[:4]))
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
    top = edges.head(20)[["from_team", "to_team", "shared_players"]].copy()
    top.columns = ["From", "To (current)", "Shared Players"]
    st.dataframe(top, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
