import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Texas Interview Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLING — RETRO LIGHT THEME
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #fcfaf6;
        color: #5b2c06;
    }

    section[data-testid="stSidebar"] {
        background: #f4ede2;
        border-right: 1px solid rgba(191,87,0,0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #6a3410 !important;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(180deg, #fffaf3 0%, #f7efe2 100%);
        border: 2px solid #bf5700;
        border-radius: 24px;
        padding: 2rem 2rem 1.7rem 2rem;
        box-shadow: 0 10px 28px rgba(191,87,0,0.10);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        color: #bf5700;
        font-size: 2.55rem;
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.02em;
        font-family: Georgia, "Times New Roman", serif;
    }

    .hero p {
        margin-top: 0.95rem;
        margin-bottom: 0;
        color: #7a4a1b;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 980px;
    }

    .card {
        background: #fffdf9;
        border: 1px solid rgba(191,87,0,0.16);
        border-radius: 22px;
        padding: 1.15rem 1.15rem 1rem 1.15rem;
        box-shadow: 0 8px 20px rgba(191,87,0,0.08);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, #fffdf9 0%, #fbf4e8 100%);
        border: 1px solid rgba(191,87,0,0.16);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 8px 20px rgba(191,87,0,0.08);
    }

    .metric-label {
        font-size: 0.76rem;
        color: #9b5b22;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #8a3e00;
        line-height: 1.1;
        font-family: Georgia, "Times New Roman", serif;
    }

    .metric-small {
        margin-top: 0.35rem;
        color: #7a4a1b;
        font-size: 0.88rem;
    }

    .section-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #a54700;
        margin-top: 0.2rem;
        margin-bottom: 0.7rem;
        letter-spacing: -0.01em;
        font-family: Georgia, "Times New Roman", serif;
    }

    .subtle {
        color: #7a4a1b;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .profile-name {
        font-size: 2rem;
        font-weight: 800;
        color: #8f3f00;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
        font-family: Georgia, "Times New Roman", serif;
    }

    .pill {
        display: inline-block;
        padding: 0.42rem 0.85rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: #fff0e2;
        color: #bf5700;
        border: 1px solid rgba(191,87,0,0.22);
        margin-bottom: 0.85rem;
    }

    .callout-list {
        margin-top: 0.7rem;
        padding-left: 1rem;
    }

    .callout-list li {
        margin-bottom: 0.35rem;
        color: #6a3410;
    }

    .footer-note {
        color: #8a5c32;
        font-size: 0.86rem;
        margin-top: 0.65rem;
    }

    .stDataFrame, .stTable {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid rgba(191,87,0,0.14);
    }

    h1, h2, h3 {
        color: #a54700 !important;
        font-family: Georgia, "Times New Roman", serif !important;
    }

    [data-testid="stMetric"] {
        background: transparent !important;
    }

    hr {
        border-color: rgba(191,87,0,0.14) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        border-bottom: 2px solid rgba(191,87,0,0.2);
        background: transparent;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f7efe2;
        border-radius: 10px 10px 0 0;
        color: #7a4a1b;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.01em;
        padding: 0.55rem 1.25rem;
        border: 1px solid rgba(191,87,0,0.18);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: #bf5700 !important;
        color: #fff !important;
        border-color: #bf5700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none;}

    /* style the first (controls) column as a panel */
    [data-testid="column"]:first-child {
        background: #fffdf9;
        border: 1px solid rgba(191,87,0,0.16);
        border-radius: 18px;
        padding: 1.2rem 1rem 1.4rem 1rem !important;
        box-shadow: 0 4px 14px rgba(191,87,0,0.07);
    }
    .ctrl-label {
        font-size: 0.72rem;
        font-weight: 800;
        color: #9b5b22;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
        margin-top: 0.9rem;
    }
    .ctrl-label:first-child { margin-top: 0; }

    .navbar {
        background: linear-gradient(90deg, #bf5700 0%, #a34a00 100%);
        padding: 1rem 2rem;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(191,87,0,0.25);
    }
    .nav-brand {
        color: #fff;
        font-size: 1.5rem;
        font-weight: 800;
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: -0.02em;
    }
    .nav-sub {
        color: rgba(255,255,255,0.78);
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    [data-baseweb="tag"] {
        background: rgba(191,87,0,0.12) !important;
        border: 1px solid rgba(191,87,0,0.38) !important;
    }
    [data-baseweb="tag"] span {
        color: #7a3800 !important;
    }
    [data-baseweb="tag"] svg {
        fill: #7a3800 !important;
    }

    /* white select/input boxes */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="base-input"],
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        border-color: rgba(191,87,0,0.3) !important;
        color: #5b2c06 !important;
    }
    [data-baseweb="select"] svg,
    [data-baseweb="input"] svg {
        fill: #bf5700 !important;
    }
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: #fff0e2 !important;
    }
    [data-baseweb="option"] {
        color: #5b2c06 !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    required = [
        "player_archetype_profiles.csv",
        "question_cluster_summary.csv",
        "interview_rows_with_auto_question_types.csv",
        "regression_test_results.csv"
    ]
    for f in required:
        if not (DATA_DIR / f).exists():
            st.error(f"Missing file: {DATA_DIR / f}")
            st.stop()

    players = pd.read_csv(DATA_DIR / "player_archetype_profiles.csv")
    questions = pd.read_csv(DATA_DIR / "question_cluster_summary.csv")
    rows = pd.read_csv(DATA_DIR / "interview_rows_with_auto_question_types.csv")
    reg = pd.read_csv(DATA_DIR / "regression_test_results.csv")

    if "season" in rows.columns:
        rows["season"] = pd.to_numeric(rows["season"], errors="coerce")

    if "interview_date" in rows.columns:
        rows["interview_date"] = pd.to_datetime(rows["interview_date"], errors="coerce")

    numeric_player_cols = [
        "avg_word_count", "sd_word_count", "avg_pred_word_count", "pc1", "pc2",
        "self_rate", "team_rate", "hedge_rate", "confidence_rate",
        "gratitude_rate", "accountability_rate", "coachspeak_rate", "type_token_ratio"
    ]
    for col in numeric_player_cols:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")

    for col in ["word_count_final", "pred_word_count"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
        if col in reg.columns:
            reg[col] = pd.to_numeric(reg[col], errors="coerce")

    return players, questions, rows, reg

players, questions, rows, reg = load_data()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def clean_years(series):
    vals = []
    for x in sorted(series.dropna().unique()):
        try:
            vals.append(int(x))
        except Exception:
            pass
    return vals

def archetype_description(name):
    x = str(name).lower()
    if "team-oriented" in x:
        return "This player tends to sound collaborative, steady, and team-first. Answers often lean on group identity, shared responsibility, and classic program language."
    if "concise" in x:
        return "This player tends to be more direct and economical. Answers are shorter, cleaner, and often come across as blunt, controlled, or accountability-driven."
    if "detailed" in x:
        return "This player tends to elaborate more than others. Answers usually include more explanation, storytelling, and visible personality in public interview settings."
    return "This player shows a balanced public communication profile without one dominant speaking pattern."

def style_callouts(profile):
    out = []
    if profile["team_rate"] > profile["self_rate"]:
        out.append("More team-focused than self-focused")
    if profile["avg_word_count"] < 60:
        out.append("Usually concise in interviews")
    elif profile["avg_word_count"] > 120:
        out.append("Usually detailed and expansive")
    if profile["accountability_rate"] > 0.002:
        out.append("Shows accountability language")
    if profile["coachspeak_rate"] > 0.002:
        out.append("Uses classic coach-speak / program language")
    if profile["confidence_rate"] > 0.015:
        out.append("Displays confidence-oriented phrasing")
    if not out:
        out.append("Balanced style without one dominant language signal")
    return out

def get_profile_from_filtered(player_name, player_row, player_rows):
    if player_rows.empty:
        return {
            "player_name": player_name,
            "archetype": player_row["archetype"],
            "n_responses": int(player_row["n_responses"]),
            "n_seasons": int(player_row["n_seasons"]),
            "avg_word_count": float(player_row["avg_word_count"]),
            "avg_pred_word_count": float(player_row.get("avg_pred_word_count", np.nan)),
            "self_rate": float(player_row["self_rate"]),
            "team_rate": float(player_row["team_rate"]),
            "hedge_rate": float(player_row["hedge_rate"]),
            "confidence_rate": float(player_row["confidence_rate"]),
            "gratitude_rate": float(player_row["gratitude_rate"]),
            "accountability_rate": float(player_row["accountability_rate"]),
            "coachspeak_rate": float(player_row["coachspeak_rate"]),
            "type_token_ratio": float(player_row["type_token_ratio"]),
            "dominant_question_type": player_row["dominant_question_type"]
        }

    dominant_q = (
        player_rows["pred_question_type"].mode().iloc[0]
        if "pred_question_type" in player_rows.columns and not player_rows["pred_question_type"].mode().empty
        else player_row["dominant_question_type"]
    )

    return {
        "player_name": player_name,
        "archetype": player_row["archetype"],
        "n_responses": int(len(player_rows)),
        "n_seasons": int(player_rows["season"].nunique()) if "season" in player_rows.columns else int(player_row["n_seasons"]),
        "avg_word_count": float(player_rows["word_count_final"].mean()) if "word_count_final" in player_rows.columns else float(player_row["avg_word_count"]),
        "avg_pred_word_count": float(player_rows["pred_word_count"].mean()) if "pred_word_count" in player_rows.columns else float(player_row.get("avg_pred_word_count", np.nan)),
        "self_rate": float(player_rows["self_rate"].mean()) if "self_rate" in player_rows.columns else float(player_row["self_rate"]),
        "team_rate": float(player_rows["team_rate"].mean()) if "team_rate" in player_rows.columns else float(player_row["team_rate"]),
        "hedge_rate": float(player_rows["hedge_rate"].mean()) if "hedge_rate" in player_rows.columns else float(player_row["hedge_rate"]),
        "confidence_rate": float(player_rows["confidence_rate"].mean()) if "confidence_rate" in player_rows.columns else float(player_row["confidence_rate"]),
        "gratitude_rate": float(player_rows["gratitude_rate"].mean()) if "gratitude_rate" in player_rows.columns else float(player_row["gratitude_rate"]),
        "accountability_rate": float(player_rows["accountability_rate"].mean()) if "accountability_rate" in player_rows.columns else float(player_row["accountability_rate"]),
        "coachspeak_rate": float(player_rows["coachspeak_rate"].mean()) if "coachspeak_rate" in player_rows.columns else float(player_row["coachspeak_rate"]),
        "type_token_ratio": float(player_rows["type_token_ratio"].mean()) if "type_token_ratio" in player_rows.columns else float(player_row["type_token_ratio"]),
        "dominant_question_type": dominant_q
    }

def metric_card(label, value, small=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-small">{small}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# LAYOUT: left controls + right main
# --------------------------------------------------
col_ctrl, col_main = st.columns([1, 4], gap="medium")

with col_ctrl:
    player_list = sorted(players["player_name"].dropna().astype(str).unique())
    all_years = clean_years(rows["season"]) if "season" in rows.columns else []
    all_qtypes = sorted(rows["pred_question_type"].dropna().astype(str).unique()) if "pred_question_type" in rows.columns else []

    st.markdown('<div class="ctrl-label">Player</div>', unsafe_allow_html=True)
    selected_player = st.selectbox("Player", player_list, label_visibility="collapsed")

    st.markdown('<div class="ctrl-label">Seasons</div>', unsafe_allow_html=True)
    selected_years = st.multiselect("Seasons", all_years, default=all_years, label_visibility="collapsed")

    if all_qtypes:
        st.markdown('<div class="ctrl-label">Question types</div>', unsafe_allow_html=True)
        selected_qtypes = st.multiselect("Question types", all_qtypes, default=all_qtypes, label_visibility="collapsed")
    else:
        selected_qtypes = []

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------
player_row = players.loc[players["player_name"] == selected_player].iloc[0].copy()

player_rows = rows.loc[rows["player_name"] == selected_player].copy()
if selected_years and "season" in player_rows.columns:
    player_rows = player_rows[player_rows["season"].isin(selected_years)]
if selected_qtypes and "pred_question_type" in player_rows.columns:
    player_rows = player_rows[player_rows["pred_question_type"].isin(selected_qtypes)]

profile = get_profile_from_filtered(selected_player, player_row, player_rows)

with col_main:
    # --------------------------------------------------
    # HERO
    # --------------------------------------------------
    st.markdown(
        """
        <div class="navbar">
            <div class="nav-brand">BOSSO Texas Football NLP Project</div>
            <div class="nav-sub">Player Communication Scouting · 2021–2025</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # KPI ROW
    # --------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Players profiled", f"{players.shape[0]}", "Cross-year player profiles")
    with k2:
        metric_card("Interview responses", f"{rows.shape[0]}", "Cleaned usable rows")
    with k3:
        metric_card("Years covered", f"{len(all_years)}", "2021 through 2025")
    with k4:
        metric_card("Archetypes", f"{players['archetype'].nunique()}", "Communication-style groupings")

    st.markdown("")

    tab_overview, tab_analytics, tab_regression, tab_log, tab_method, tab_data = st.tabs([
        "Overview", "Analytics", "Regression", "Interview Log", "Methodology", "Data"
    ])

# --------------------------------------------------
# TAB: OVERVIEW
# --------------------------------------------------
with tab_overview:
    left, right = st.columns([1.1, 1])

    with left:
        callouts = style_callouts(profile)
        callout_items = "".join([f"<li>{x}</li>" for x in callouts])
        st.markdown(
            f"""
            <div class="card">
                <div class="pill">{profile["archetype"]}</div>
                <div class="profile-name">{selected_player}</div>
                <div class="subtle">
                    <strong>{profile["n_responses"]}</strong> filtered responses across
                    <strong>{profile["n_seasons"]}</strong> season(s). Most common question context:
                    <strong>{profile["dominant_question_type"]}</strong>.
                </div>
                <div class="subtle" style="margin-top:0.9rem;">{archetype_description(profile["archetype"])}</div>
                <div class="section-title" style="margin-top:1rem;">Scouting readout</div>
                <ul class="callout-list">{callout_items}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        fig_map = px.scatter(
            players,
            x="pc1",
            y="pc2",
            color="archetype",
            size="n_responses",
            hover_name="player_name",
            hover_data=["avg_word_count", "n_seasons", "dominant_question_type"],
            title="Communication archetype map"
        )
        sel = players[players["player_name"] == selected_player]
        fig_map.add_scatter(
            x=sel["pc1"],
            y=sel["pc2"],
            mode="markers+text",
            text=sel["player_name"],
            textposition="top center",
            marker=dict(size=24, symbol="diamond"),
            name="Selected player"
        )
        fig_map.update_layout(
            template="simple_white",
            height=560,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffdf9",
            font=dict(color="#3d1f00", size=13),
            title_font=dict(color="#a54700", size=15),
            legend=dict(font=dict(size=12, color="#3d1f00")),
            xaxis=dict(tickfont=dict(color="#3d1f00", size=12), title_font=dict(color="#3d1f00")),
            yaxis=dict(tickfont=dict(color="#3d1f00", size=12), title_font=dict(color="#3d1f00")),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig_map.update_traces(textfont=dict(color="#3d1f00", size=12))
        st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# TAB: ANALYTICS
# --------------------------------------------------
with tab_analytics:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Avg words", f"{profile['avg_word_count']:.1f}", "Response length")
    with m2:
        metric_card("Pred avg words", f"{profile['avg_pred_word_count']:.1f}", "Regression estimate")
    with m3:
        metric_card("Self rate", f"{profile['self_rate']:.3f}", "I / me / my usage")
    with m4:
        metric_card("Team rate", f"{profile['team_rate']:.3f}", "We / us / our usage")

    st.markdown("")

    style_df = pd.DataFrame({
        "feature": [
            "Self focus", "Team focus", "Hedge language", "Confidence language",
            "Gratitude language", "Accountability language", "Coach-speak", "Vocabulary diversity"
        ],
        "value": [
            profile["self_rate"], profile["team_rate"], profile["hedge_rate"],
            profile["confidence_rate"], profile["gratitude_rate"],
            profile["accountability_rate"], profile["coachspeak_rate"], profile["type_token_ratio"]
        ]
    })
    fig_style = px.bar(style_df, x="feature", y="value", title="Style signal profile")
    fig_style.update_layout(
        template="simple_white",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffdf9",
        xaxis_title="",
        yaxis_title="Value",
        font=dict(color="#3d1f00", size=13),
        title_font=dict(color="#a54700", size=15),
        xaxis=dict(tickangle=-35, tickfont=dict(size=12, color="#3d1f00"), title_font=dict(color="#3d1f00")),
        yaxis=dict(tickfont=dict(size=12, color="#3d1f00"), title_font=dict(color="#3d1f00")),
        margin=dict(l=20, r=20, t=60, b=100)
    )
    fig_style.update_traces(textfont=dict(color="#3d1f00"))
    st.plotly_chart(fig_style, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-title">Question-type distribution</div>', unsafe_allow_html=True)
        if not player_rows.empty and "pred_question_type" in player_rows.columns:
            qmix = player_rows["pred_question_type"].value_counts().reset_index()
            qmix.columns = ["question_type", "count"]
            fig_qmix = px.pie(qmix, names="question_type", values="count")
            fig_qmix.update_layout(
                template="simple_white",
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf9",
                font=dict(color="#3d1f00", size=12),
                legend=dict(font=dict(size=11, color="#3d1f00")),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_qmix.update_traces(
                textfont=dict(size=12, color="#3d1f00"),
                textinfo="percent+label",
                insidetextorientation="radial"
            )
            st.plotly_chart(fig_qmix, use_container_width=True)
        else:
            st.warning("No rows available under the current filter selection.")

    with col2:
        st.markdown('<div class="section-title">Answer length by question type</div>', unsafe_allow_html=True)
        if not player_rows.empty and "pred_question_type" in player_rows.columns:
            fig_box = px.box(player_rows, x="pred_question_type", y="word_count_final")
            fig_box.update_layout(
                template="simple_white",
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf9",
                xaxis_title="",
                yaxis_title="Word count",
                font=dict(color="#3d1f00", size=12),
                xaxis=dict(tickangle=-30, tickfont=dict(size=11, color="#3d1f00"), title_font=dict(color="#3d1f00")),
                yaxis=dict(tickfont=dict(size=12, color="#3d1f00"), title_font=dict(color="#3d1f00")),
                margin=dict(l=20, r=20, t=20, b=80)
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No rows available under the current filter selection.")

    with col3:
        st.markdown('<div class="section-title">Most similar players</div>', unsafe_allow_html=True)
        feature_cols = [
            "avg_word_count", "sd_word_count", "avg_pred_word_count",
            "self_rate", "team_rate", "hedge_rate", "confidence_rate",
            "gratitude_rate", "accountability_rate", "coachspeak_rate", "type_token_ratio"
        ]
        usable = players[["player_name", "archetype"] + feature_cols].copy().dropna()
        if selected_player in usable["player_name"].values:
            base = usable.loc[usable["player_name"] == selected_player, feature_cols].iloc[0].values.astype(float)
            comp = usable.copy()
            mat = comp[feature_cols].values.astype(float)
            comp["similarity"] = 1 / (1 + np.sqrt(((mat - base) ** 2).sum(axis=1)))
            comp = comp[comp["player_name"] != selected_player].sort_values("similarity", ascending=False).head(8)
            comp["similarity"] = comp["similarity"].map(lambda x: f"{x:.3f}")
            st.dataframe(comp[["player_name", "archetype", "similarity"]].reset_index(drop=True), use_container_width=True, height=340)
        else:
            st.warning("Not enough feature data.")

# --------------------------------------------------
# TAB: REGRESSION
# --------------------------------------------------
with tab_regression:
    rmse = float(np.sqrt(np.mean((reg["actual_word_count"] - reg["pred_word_count"]) ** 2)))
    mae = float(np.mean(np.abs(reg["actual_word_count"] - reg["pred_word_count"])))
    corr = float(reg["actual_word_count"].corr(reg["pred_word_count"]))

    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("RMSE", f"{rmse:.2f}", "Average prediction error scale")
    with r2:
        metric_card("MAE", f"{mae:.2f}", "Average absolute error")
    with r3:
        metric_card("Correlation", f"{corr:.2f}", "Actual vs predicted alignment")

    st.markdown("")
    fig_reg = px.scatter(reg, x="actual_word_count", y="pred_word_count", title="Actual vs predicted answer length")
    fig_reg.add_shape(
        type="line",
        x0=reg["actual_word_count"].min(),
        y0=reg["actual_word_count"].min(),
        x1=reg["actual_word_count"].max(),
        y1=reg["actual_word_count"].max()
    )
    fig_reg.update_layout(
        template="simple_white",
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffdf9",
        font=dict(color="#3d1f00", size=13),
        title_font=dict(color="#a54700", size=15),
        xaxis=dict(title="Actual word count", tickfont=dict(size=12, color="#3d1f00"), title_font=dict(color="#3d1f00")),
        yaxis=dict(title="Predicted word count", tickfont=dict(size=12, color="#3d1f00"), title_font=dict(color="#3d1f00")),
        margin=dict(l=60, r=20, t=60, b=60)
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# --------------------------------------------------
# TAB: INTERVIEW LOG
# --------------------------------------------------
with tab_log:
    show_cols = [
        c for c in [
            "season", "interview_date", "pred_question_type",
            "word_count_final", "pred_word_count",
            "cleaned_question_text", "cleaned_player_answer_text",
            "source_link", "notes"
        ] if c in player_rows.columns
    ]
    st.markdown('<div class="section-title">Interview evidence</div>', unsafe_allow_html=True)
    if not player_rows.empty:
        sort_cols = [c for c in ["season", "interview_date"] if c in player_rows.columns]
        st.dataframe(
            player_rows[show_cols].sort_values(sort_cols, ascending=False),
            use_container_width=True,
            height=560
        )
    else:
        st.warning("No interview rows match the current filter.")

# --------------------------------------------------
# TAB: METHODOLOGY
# --------------------------------------------------
with tab_method:
    st.markdown("""
    <div class="card">
        <div class="section-title">Model 1 — Question Clustering</div>
        <div class="subtle">
        Each unique question in the log is vectorized using <strong>TF-IDF</strong> (300 features, 1–2 word ngrams,
        English stop words removed). A <strong>KMeans</strong> model is then fit across candidate cluster counts
        k = 3, 4, 5, 6. The best k is selected by silhouette score. Each cluster is given a heuristic label
        based on its top terms:
        <ul class="callout-list">
            <li><strong>mistake_adversity</strong> — questions about errors, turnovers, tough moments</li>
            <li><strong>leadership_team</strong> — questions about teammates, locker room, culture</li>
            <li><strong>preparation_process</strong> — questions about practice, game week, process</li>
            <li><strong>opponent_game_context</strong> — questions about specific opponents or matchups</li>
            <li><strong>performance</strong> — general performance and execution questions</li>
        </ul>
        Each row in the dataset is then tagged with its predicted question type, which is used downstream
        in both the regression and archetype models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-title">Model 2 — Answer Length Regression</div>
        <div class="subtle">
        A <strong>Ridge regression</strong> model predicts how many words a player will use in response to a
        given question. Features fed into the model:
        <ul class="callout-list">
            <li>Question text (TF-IDF, 500 features, 1–2 ngrams)</li>
            <li>Player name (one-hot encoded)</li>
            <li>Predicted question type (one-hot encoded)</li>
            <li>Season (numeric, median-imputed)</li>
        </ul>
        The dataset is split 80/20 for training and evaluation. RMSE, MAE, and correlation between actual
        and predicted word count are reported. Predicted word counts are also generated for every row and
        used as a feature in the archetype clustering step.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-title">Model 3 — Player Archetype Clustering</div>
        <div class="subtle">
        Each player is summarized into a single feature vector combining:
        <ul class="callout-list">
            <li>13 aggregated style metrics (avg word count, SD, self/team/hedge/confidence/gratitude/
            accountability/coachspeak rates, type-token ratio, predicted word count, season count)</li>
            <li>TF-IDF of all their combined interview text (150 features, 1–2 ngrams)</li>
        </ul>
        Style features are standardized; text features are appended. <strong>PCA</strong> reduces this
        combined space to 2D for visualization. <strong>KMeans</strong> (k = 3–6, best by silhouette score)
        assigns each player to a cluster. Cluster labels are generated from quantile thresholds on the
        cluster centroid values — e.g. a cluster at the 75th percentile for team_rate and coachspeak_rate
        is labeled <em>Team-oriented / Coach-speak</em>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-title">Language signals</div>
        <div class="subtle">
        Every player answer is tokenized and scanned for the following signal categories. All rates are
        computed as (signal token count) / (total token count).
        <ul class="callout-list">
            <li><strong>Self focus</strong> — I, me, my, mine, myself</li>
            <li><strong>Team focus</strong> — we, us, our, ours, ourselves, team</li>
            <li><strong>Hedge language</strong> — maybe, kind, sort, probably, possibly, guess, perhaps, think, feel, felt, might, little</li>
            <li><strong>Confidence language</strong> — definitely, obviously, confident, ready, always, never, absolutely, really, sure</li>
            <li><strong>Gratitude language</strong> — thank, blessed, grateful, honor, honored, appreciate</li>
            <li><strong>Accountability phrases</strong> — "starts with me", "my fault", "on me", "need to be better", "i need to", "i have to", "got to play better"</li>
            <li><strong>Coach-speak phrases</strong> — "control what we can control", "one day at a time", "next man up", "play fast", "stay focused", "execute", "get better"</li>
            <li><strong>Vocabulary diversity (TTR)</strong> — unique tokens / total tokens</li>
        </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# TAB: DATA
# --------------------------------------------------
with tab_data:
    st.markdown("""
    <div class="card">
        <div class="section-title">Source file</div>
        <div class="subtle">
        All data originates from a single Excel workbook: <strong>BOSSO Interview Master Log.xlsx</strong>.
        The file is organized into one sheet per season. Each sheet follows the same column schema and is
        concatenated at load time. Rows with empty player names, very short questions (&lt;5 chars), or
        very short answers (&lt;10 chars) are dropped before any modeling occurs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Column schema</div>', unsafe_allow_html=True)
    schema = pd.DataFrame({
        "Column": [
            "season", "player_name", "interview_date", "source_link",
            "cleaned_question_text", "cleaned_player_answer_text", "word_count"
        ],
        "Type": ["numeric", "text", "date", "url", "text", "text", "numeric"],
        "Description": [
            "Season year (e.g. 2023)",
            "Full player name as it appears in the source transcript",
            "Date of the media availability or press conference",
            "URL to the source video or transcript",
            "Reporter question, lightly cleaned for punctuation/formatting",
            "Player response, lightly cleaned for punctuation/formatting",
            "Word count — manually entered or left blank for auto-computation"
        ]
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="card">
        <div class="section-title">Collection process</div>
        <div class="subtle">
        Interviews were sourced from publicly available Texas Longhorns media availability recordings and
        official press conference transcripts spanning the 2021–2025 seasons. Each Q&A exchange was logged
        manually as a single row. Questions and answers were lightly cleaned to remove filler artifacts
        (e.g. "[laughter]", "[crosstalk]") while preserving the player's original phrasing as closely as
        possible. No semantic editing was performed on player responses.
        <br><br>
        Word count is either taken from the <em>word_count</em> column if manually recorded, or computed
        automatically using a token regex (<code>\\b[\\w']+\\b</code>) applied to the cleaned answer text.
        </div>
    </div>
    """, unsafe_allow_html=True)
