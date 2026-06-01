"""
app.py — Season-aware Streamlit Dashboard
FIX: Shows debug info when squad is empty so you know exactly why
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import (
    load_all_players, load_season_data,
    get_all_seasons, get_clubs_for_season,
    get_squad_for_season, predict_match,
    FORMATIONS, get_season_range, PLAYER_DATA_DIR
)

st.set_page_config(
    page_title="⚽ Match Predictor",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data
def load_all():
    players     = load_all_players()
    season_data = load_season_data()
    return players, season_data

PLAYERS, SEASON_DATA = load_all()
ALL_SEASONS = get_all_seasons(SEASON_DATA)

st.title("⚽ Football Match Predictor")
st.markdown(
    "Select a **season**, pick teams and a **match date**, "
    "choose your **formation** and **Playing XI**. "
    "The model uses **only data strictly before the selected date**."
)

if not ALL_SEASONS:
    st.error("❌ SEASON_DATA.csv not found in PLAYER DATA/ folder.")
    st.stop()
if not PLAYERS:
    st.error("❌ No player CSVs found in PLAYER DATA/ folder.")
    st.stop()

# ── Debug expander: show all loaded players ──
with st.expander("🔍 Debug — Loaded Players & SEASON_DATA", expanded=False):
    st.write(f"**Player CSVs loaded:** {len(PLAYERS)}")
    st.write(sorted(PLAYERS.keys()))
    st.write(f"**SEASON_DATA seasons:** {sorted(SEASON_DATA.keys())}")

st.divider()

# ── STEP 1: SEASON ──
st.markdown("### 📅 Step 1 — Select Season")
selected_season = st.selectbox("Season", ALL_SEASONS, index=0)

season_start, season_end = get_season_range(selected_season)
clubs_in_season = get_clubs_for_season(SEASON_DATA, selected_season)

if not clubs_in_season:
    st.warning(f"No clubs found for season {selected_season} in SEASON_DATA.csv.")
    st.stop()

st.divider()

# ── STEP 2: TEAMS & DATE ──
st.markdown("### 🏟️ Step 2 — Select Teams & Match Date")
c1, c2, c3 = st.columns(3)
with c1:
    home_team = st.selectbox("🏠 Home Team", clubs_in_season, index=0)
with c2:
    away_opts = [c for c in clubs_in_season if c != home_team]
    away_team = st.selectbox("✈️ Away Team", away_opts, index=0)
with c3:
    match_date = st.date_input(
        "📅 Match Date",
        value=season_start.date(),
        min_value=season_start.date(),
        max_value=season_end.date(),
    )

st.caption(f"ℹ️ Only data **strictly before {match_date}** will be used for prediction.")
st.divider()

# ── STEP 3: FORMATIONS ──
st.markdown("### 🗂️ Step 3 — Select Formations")
fc1, fc2 = st.columns(2)
with fc1:
    home_formation = st.selectbox(f"🏠 {home_team} Formation", list(FORMATIONS.keys()), index=0, key="hf")
with fc2:
    away_formation = st.selectbox(f"✈️ {away_team} Formation", list(FORMATIONS.keys()), index=0, key="af")

home_slots = FORMATIONS[home_formation]
away_slots = FORMATIONS[away_formation]
st.divider()

# ── STEP 4: PLAYING XI ──
st.markdown("### 👕 Step 4 — Select Playing XI")
st.caption("Players shown are those who played for this club in the selected season. Each player can only be selected once.")

home_squad = get_squad_for_season(SEASON_DATA, PLAYERS, home_team, selected_season)
away_squad = get_squad_for_season(SEASON_DATA, PLAYERS, away_team, selected_season)

home_xi = []
away_xi = []

col_home, col_away = st.columns(2)


def _squad_debug_info(team, season, squad, season_data, players_dict):
    """Return a helpful message explaining why squad is empty."""
    # Check if team is in SEASON_DATA for this season
    in_sd = team in season_data.get(season, {})
    sd_players = season_data.get(season, {}).get(team, [])

    lines = [f"**Team in SEASON_DATA for {season}:** {'✅ Yes' if in_sd else '❌ No'}"]
    if in_sd:
        lines.append(f"**Players listed in SEASON_DATA:** {sd_players if sd_players else '(none)'}")

    # Show which CSVs are loaded
    lines.append(f"**Total player CSVs loaded:** {len(players_dict)}")

    # Check if any CSV name contains part of the team name
    team_words = [w.lower() for w in team.split() if len(w) > 2]
    possible = [p for p in players_dict if any(w in p.lower() for w in team_words)]
    if possible:
        lines.append(f"**Possible matching CSVs (by team name keywords):** {possible}")
    else:
        lines.append(f"**No CSV filenames contain words from '{team}'** — "
                     f"check your SEASON_DATA.csv player names match the CSV filenames exactly.")

    return "\n\n".join(lines)


with col_home:
    st.markdown(f"#### 🏠 {home_team} — {home_formation}")
    if not home_squad:
        st.error(
            f"❌ No player CSVs found for **{home_team}** in {selected_season}. "
            f"Make sure:\n"
            f"1. Player CSV files are in the `PLAYER DATA/` folder.\n"
            f"2. `SEASON_DATA.csv` lists these players under **{home_team}** / **{selected_season}**.\n"
            f"3. Player names in SEASON_DATA.csv **exactly match** the CSV filenames (before ' - Sheet1')."
        )
        with st.expander(f"🔍 Debug info for {home_team}"):
            st.markdown(_squad_debug_info(home_team, selected_season, home_squad, SEASON_DATA, PLAYERS))
    else:
        for i, slot in enumerate(home_slots):
            already  = [p for p in home_xi if p]
            options  = [p for p in home_squad if p not in already]
            if not options:
                options = home_squad
            player = st.selectbox(f"**{slot}** — Slot {i+1}", options=options, index=0, key=f"h_{i}")
            home_xi.append(player)

with col_away:
    st.markdown(f"#### ✈️ {away_team} — {away_formation}")
    if not away_squad:
        st.error(
            f"❌ No player CSVs found for **{away_team}** in {selected_season}. "
            f"Make sure:\n"
            f"1. Player CSV files are in the `PLAYER DATA/` folder.\n"
            f"2. `SEASON_DATA.csv` lists these players under **{away_team}** / **{selected_season}**.\n"
            f"3. Player names in SEASON_DATA.csv **exactly match** the CSV filenames (before ' - Sheet1')."
        )
        with st.expander(f"🔍 Debug info for {away_team}"):
            st.markdown(_squad_debug_info(away_team, selected_season, away_squad, SEASON_DATA, PLAYERS))
    else:
        for i, slot in enumerate(away_slots):
            already  = [p for p in away_xi if p]
            options  = [p for p in away_squad if p not in already]
            if not options:
                options = away_squad
            player = st.selectbox(f"**{slot}** — Slot {i+1}", options=options, index=0, key=f"a_{i}")
            away_xi.append(player)

st.divider()

# ── VALIDATION ──
home_dups = [p for p in set(home_xi) if home_xi.count(p) > 1]
away_dups = [p for p in set(away_xi)  if away_xi.count(p)  > 1]

if home_dups:
    st.error(f"❌ {home_team}: **{', '.join(home_dups)}** selected more than once.")
if away_dups:
    st.error(f"❌ {away_team}: **{', '.join(away_dups)}** selected more than once.")

can_predict = bool(home_squad and away_squad and not home_dups and not away_dups)

# ── PREDICT BUTTON ──
predict_btn = st.button(
    "🔮 PREDICT MATCH",
    type="primary",
    use_container_width=True,
    disabled=not can_predict
)

if not can_predict and (not home_squad or not away_squad):
    missing = []
    if not home_squad: missing.append(home_team)
    if not away_squad: missing.append(away_team)
    st.warning(f"⚠️ Prediction disabled — missing player CSVs for: **{', '.join(missing)}**")

if predict_btn and can_predict:
    valid_home = [p for p in home_xi if p]
    valid_away = [p for p in away_xi  if p]

    with st.spinner("Running prediction..."):
        try:
            result = predict_match(
                PLAYERS, home_team, away_team,
                valid_home, valid_away,
                str(match_date), selected_season
            )

            st.divider()
            st.markdown("## 📊 Prediction Results")
            st.markdown(
                f"**{home_team}** ({home_formation}) vs "
                f"**{away_team}** ({away_formation}) | "
                f"📅 {match_date} | 🗓️ {selected_season}"
            )

            st.markdown("### ⚽ Expected Goals")
            xc1,xc2,xc3 = st.columns(3)
            xc1.metric(f"🏠 {home_team}", result["xg_home"])
            xc2.metric("VS","—")
            xc3.metric(f"✈️ {away_team}", result["xg_away"])
            st.divider()

            st.markdown("### 📈 Match Probabilities")
            pc1,pc2,pc3 = st.columns(3)
            pc1.metric(f"🏠 {home_team} Win", f"{result['home_win']}%")
            pc2.metric("🤝 Draw",              f"{result['draw']}%")
            pc3.metric(f"✈️ {away_team} Win",  f"{result['away_win']}%")
            prob_df = pd.DataFrame({
                "Outcome":[f"{home_team} Win","Draw",f"{away_team} Win"],
                "Probability (%)":[result["home_win"],result["draw"],result["away_win"]]
            })
            st.bar_chart(prob_df.set_index("Outcome"))
            st.divider()

            st.markdown("### 🎯 Top 5 Most Likely Scorelines")
            st.caption(f"Home: {home_team} — Away: {away_team}")
            st.dataframe(
                pd.DataFrame(result["top5"], columns=["Scoreline","Probability (%)"]),
                use_container_width=True, hide_index=True
            )
            st.divider()

            st.markdown("### 🔥 Scoreline Probability Heatmap (%)")
            matrix_pct = np.round(result["matrix"]*100, 2)
            heatmap_df = pd.DataFrame(
                matrix_pct,
                index=[f"{home_team} {i}" for i in range(7)],
                columns=[f"{away_team} {j}" for j in range(7)]
            )
            st.dataframe(
                heatmap_df.style.background_gradient(cmap="YlOrRd"),
                use_container_width=True
            )
            st.divider()

            st.markdown("### 📋 Playing XI Used")
            xi1,xi2 = st.columns(2)
            with xi1:
                st.markdown(f"**🏠 {home_team} ({home_formation})**")
                for slot, player in zip(home_slots, home_xi):
                    st.write(f"**{slot}** — {player}")
            with xi2:
                st.markdown(f"**✈️ {away_team} ({away_formation})**")
                for slot, player in zip(away_slots, away_xi):
                    st.write(f"**{slot}** — {player}")
            st.divider()

            st.markdown("### ℹ️ Data Used")
            st.info(f"Only matches strictly before **{match_date}** were used.")
            mdt = pd.to_datetime(str(match_date))
            info_rows = []
            for name, df in PLAYERS.items():
                past = df[df["Date"] < mdt]
                if len(past) > 0:
                    info_rows.append({
                        "Player": name,
                        "Matches Used": len(past),
                        "Latest Match Used": str(past["Date"].max().date())
                    })
            if info_rows:
                st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)

st.divider()
st.markdown("📌 **Add more players:** Upload CSV to `PLAYER DATA/` in GitHub. Update `SEASON_DATA.csv` with their club & season.")
