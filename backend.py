"""
backend.py — Rebuilt using original core logic + player-level data
Original logic: Rolling stats → Traditional xG → RF + LR ensemble
                (1/MAE weighted) → Poisson scoreline matrix
Extra additions: Player CSVs, Playing XI, Season filter, Date filter
"""

import os, math, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SEASON_DATES    = {
    "2024-2025": ("2024-07-01","2025-06-30"),
    "2023-2024": ("2023-07-01","2024-06-30"),
    "2022-2023": ("2022-07-01","2023-06-30"),
    "2021-2022": ("2021-07-01","2022-06-30"),
    "2020-2021": ("2020-07-01","2021-06-30"),
    "2019-2020": ("2019-07-01","2020-06-30"),
    "2018-2019": ("2018-07-01","2019-06-30"),
    "2017-2018": ("2017-07-01","2018-06-30"),
    "2016-2017": ("2016-07-01","2017-06-30"),
    "2015-2016": ("2015-07-01","2016-06-30"),
    "2014-2015": ("2014-07-01","2015-06-30"),
    "2013-2014": ("2013-07-01","2014-06-30"),
    "2012-2013": ("2012-07-01","2013-06-30"),
    "2011-2012": ("2011-07-01","2012-06-30"),
    "2010-2011": ("2010-07-01","2011-06-30"),
    "2009-2010": ("2009-07-01","2010-06-30"),
    "2008-2009": ("2008-07-01","2009-06-30"),
}

# ── All La Liga teams by season (hardcoded fallback) ──
LALIGA_TEAMS = {
    "2024-2025": [
        "Barcelona","Real Madrid","Atletico Madrid","Athletic Club","Villarreal",
        "Real Betis","Celta Vigo","Rayo Vallecano","Osasuna","Mallorca",
        "Real Sociedad","Valencia","Getafe","Espanyol","Deportivo Alaves",
        "Girona","Sevilla","Leganes","Las Palmas","Valladolid"
    ],
    "2023-2024": [
        "Real Madrid","Barcelona","Girona","Atletico Madrid","Athletic Club",
        "Real Sociedad","Real Betis","Villarreal","Valencia","Deportivo Alaves",
        "Osasuna","Las Palmas","Getafe","Celta Vigo","Sevilla",
        "Cadiz","Mallorca","Granada","Almeria","Rayo Vallecano"
    ],
    "2022-2023": [
        "Barcelona","Real Madrid","Atletico Madrid","Real Sociedad","Villarreal",
        "Real Betis","Osasuna","Athletic Club","Rayo Vallecano","Mallorca",
        "Girona","Cadiz","Almeria","Getafe","Espanyol",
        "Celta Vigo","Sevilla","Valencia","Valladolid","Elche"
    ],
    "2021-2022": [
        "Real Madrid","Barcelona","Atletico Madrid","Sevilla","Real Betis",
        "Real Sociedad","Villarreal","Athletic Club","Valencia","Osasuna",
        "Celta Vigo","Rayo Vallecano","Elche","Getafe","Espanyol",
        "Deportivo Alaves","Levante","Mallorca","Granada","Cadiz"
    ],
    "2020-2021": [
        "Atletico Madrid","Real Madrid","Barcelona","Sevilla","Real Sociedad",
        "Real Betis","Villarreal","Celta Vigo","Athletic Club","Cadiz",
        "Levante","Valencia","Osasuna","Granada","Deportivo Alaves",
        "Elche","Getafe","Huesca","Valladolid","Eibar"
    ],
}

FORMATIONS = {
    "4-3-3":   ["GK","RB","CB","CB","LB","CM","CM","CM","RW","ST","LW"],
    "4-4-2":   ["GK","RB","CB","CB","LB","RM","CM","CM","LM","ST","ST"],
    "4-2-3-1": ["GK","RB","CB","CB","LB","DM","DM","AM","RW","LW","ST"],
    "3-5-2":   ["GK","CB","CB","CB","RM","CM","CM","CM","LM","ST","ST"],
    "3-4-3":   ["GK","CB","CB","CB","RM","CM","CM","LM","RW","ST","LW"],
    "5-3-2":   ["GK","RB","CB","CB","CB","LB","CM","CM","CM","ST","ST"],
    "4-1-4-1": ["GK","RB","CB","CB","LB","DM","RM","CM","CM","LM","ST"],
}

SLOT_TO_GROUP = {
    "GK":"GK","CB":"DEF","LB":"DEF","RB":"DEF",
    "DM":"MID","CM":"MID","LM":"MID","RM":"MID","AM":"MID",
    "ST":"FWD","LW":"FWD","RW":"FWD",
}

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def find_player_data_dir():
    for name in ["PLAYER DATA","player_data","Player Data","PLAYER_DATA","data","DATA"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.isdir(p) and glob.glob(os.path.join(p,"*.csv")):
            return p
    for item in os.listdir(BASE_DIR):
        full = os.path.join(BASE_DIR, item)
        if os.path.isdir(full) and glob.glob(os.path.join(full,"*.csv")):
            return full
    return BASE_DIR

PLAYER_DATA_DIR = find_player_data_dir()

def get_season_range(season):
    if season in SEASON_DATES:
        s, e = SEASON_DATES[season]
        return pd.Timestamp(s), pd.Timestamp(e)
    try:
        yr = int(str(season).split("-")[0])
        return pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr+1}-06-30")
    except:
        return pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30")

def season_mask(df, season):
    s, e = get_season_range(season)
    return (df["Date"] >= s) & (df["Date"] <= e)

def clean_player_name(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    if " - " in name:
        name = name.split(" - ")[0]
    return name.strip()

def normalize(name):
    import unicodedata
    name = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in name if not unicodedata.combining(c)).lower().strip()

def safe_sum(df, col):
    return float(df[col].sum()) if col in df.columns else 0.0

def safe_mean(df, col):
    return float(df[col].mean()) if col in df.columns and len(df) > 0 else 0.0

# ─────────────────────────────────────────
# 1. LOAD ALL PLAYER CSVs
# ─────────────────────────────────────────
def load_all_players():
    players = {}
    for path in glob.glob(os.path.join(PLAYER_DATA_DIR, "*.csv")):
        if "SEASON_DATA" in os.path.basename(path).upper():
            continue
        name = clean_player_name(path)
        try:
            df = pd.read_csv(path, encoding="latin1")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).copy()

            # Fix encoding issues in opponent names (from your original code)
            if "Opponent" in df.columns:
                replacements = {
                    "Ã©":"é","Ã¡":"á","Ã­":"í",
                    "Ã³":"ó","Ãº":"ú","Ã±":"ñ","Ã":"Á"
                }
                for bad, good in replacements.items():
                    df["Opponent"] = df["Opponent"].str.replace(bad, good, regex=False)

            # Standardise numeric columns — handle missing gracefully
            for col in ["Goals","Assists","Shots","SoT","Minutes",
                        "TacklesWon","Interceptions","Crosses","Fouls",
                        "TeamGoals","OppGoals","Yellow","Red"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Ensure required columns exist with defaults
            defaults = {
                "Goals":0,"Assists":0,"Shots":0,"SoT":0,
                "Minutes":90,"TacklesWon":0,"Interceptions":0,
                "TeamGoals":0,"OppGoals":0,"Venue":"Home",
                "Opponent":"Unknown","Result":"D"
            }
            for col, default in defaults.items():
                if col not in df.columns:
                    df[col] = default

            # Compute G/Sh and G/SoT per row (like original code)
            df["G/Sh"]  = df.apply(
                lambda r: r["Goals"]/r["Shots"] if r["Shots"] > 0 else 0, axis=1)
            df["G/SoT"] = df.apply(
                lambda r: r["Goals"]/r["SoT"]   if r["SoT"]   > 0 else 0, axis=1)

            players[name] = df
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return players

# ─────────────────────────────────────────
# 2. LOAD SEASON DATA
# ─────────────────────────────────────────
def load_season_data():
    search_paths = [os.path.join(PLAYER_DATA_DIR, "SEASON_DATA.csv"),
                    os.path.join(BASE_DIR, "SEASON_DATA.csv")]
    try:
        for item in os.listdir(BASE_DIR):
            full = os.path.join(BASE_DIR, item)
            if os.path.isdir(full):
                search_paths.append(os.path.join(full, "SEASON_DATA.csv"))
    except:
        pass

    path = next((p for p in search_paths if os.path.exists(p)), None)
    if path is None:
        return {}

    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    result = {}
    for _, row in df.iterrows():
        season = str(row["SEASON"]).strip()
        club   = str(row["TEAM"]).strip()
        player = str(row["PLAYER"]).strip()
        # Skip placeholder entries (added just to show team in dropdown)
        if player.startswith("_placeholder_"):
            result.setdefault(season, {}).setdefault(club, [])
            continue
        result.setdefault(season, {}).setdefault(club, []).append(player)
    return result

def get_all_seasons(season_data):
    # Merge seasons from SEASON_DATA + hardcoded La Liga seasons
    csv_seasons    = set(season_data.keys())
    laliga_seasons = set(LALIGA_TEAMS.keys())
    all_seasons    = csv_seasons | laliga_seasons
    return sorted(all_seasons, reverse=True)

def get_clubs_for_season(season_data, season):
    # Get teams from SEASON_DATA.csv
    csv_teams = set(season_data.get(season, {}).keys())
    # Get hardcoded La Liga teams for this season
    laliga_teams = set(LALIGA_TEAMS.get(season, []))
    # Merge both — show all
    all_teams = csv_teams | laliga_teams
    return sorted(all_teams)

def get_squad_for_season(season_data, players_dict, club, season):
    season_players = season_data.get(season, {}).get(club, [])
    if not season_players:
        return []
    csv_norm = {normalize(n): n for n in players_dict.keys()}
    matched  = []
    for sp in season_players:
        norm_sp = normalize(sp)
        if norm_sp in csv_norm:
            matched.append(csv_norm[norm_sp])
        else:
            for norm_csv, csv_name in csv_norm.items():
                if norm_sp in norm_csv or norm_csv in norm_sp:
                    matched.append(csv_name)
                    break
    return sorted(set(matched))

# ─────────────────────────────────────────
# 3. BUILD TEAM DATAFRAME FROM XI PLAYERS
#    Key improvement: aggregate player CSVs
#    into a team-level match history
# ─────────────────────────────────────────
def build_team_df(players_dict, xi_players, match_date):
    """
    From selected XI player CSVs, build a combined team-level
    match history (one row per match date) strictly before match_date.
    Mirrors your original df structure: Date, Venue, Opponent,
    GF, GA, Sh, SoT, G/Sh, G/SoT, Result
    """
    all_rows = []
    for player in xi_players:
        if not player or player not in players_dict:
            continue
        df = players_dict[player]
        past = df[df["Date"] < match_date].copy()
        if len(past) == 0:
            continue
        all_rows.append(past)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows)

    # Aggregate per match date: sum shots/goals, keep venue/opponent/result
    agg = combined.groupby("Date").agg(
        Venue    = ("Venue",     "first"),
        Opponent = ("Opponent",  "first"),
        Result   = ("Result",    "first"),
        GF       = ("TeamGoals", "first"),  # team goals (same for all players in match)
        GA       = ("OppGoals",  "first"),  # opp goals (same for all players in match)
        Sh       = ("Shots",     "sum"),    # sum player shots
        SoT      = ("SoT",       "sum"),    # sum player shots on target
    ).reset_index().sort_values("Date")

    # Recompute G/Sh and G/SoT at team level (like original)
    agg["G/Sh"]  = agg.apply(lambda r: r["GF"]/r["Sh"]  if r["Sh"]  > 0 else 0, axis=1)
    agg["G/SoT"] = agg.apply(lambda r: r["GF"]/r["SoT"] if r["SoT"] > 0 else 0, axis=1)

    return agg

# ─────────────────────────────────────────
# 4. ORIGINAL ROLLING STATS (from your code)
# ─────────────────────────────────────────
def get_rolling_stats(team_df, venue, opponent, n=5):
    """
    Replicates your original Step 5 + 6:
    - Last N matches at venue
    - Historical record vs opponent at venue
    """
    venue_matches = team_df[team_df["Venue"] == venue]
    last_n        = venue_matches.tail(n)

    if len(last_n) == 0:
        # fallback: use all matches
        last_n = team_df.tail(n)

    avg_shots  = safe_mean(last_n, "Sh")
    avg_sot    = safe_mean(last_n, "SoT")
    avg_g_sh   = safe_mean(last_n, "G/Sh")
    avg_g_sot  = safe_mean(last_n, "G/SoT")
    avg_gf     = safe_mean(last_n, "GF")
    avg_ga     = safe_mean(last_n, "GA")

    # Historical vs opponent at venue
    opp_venue = team_df[(team_df["Opponent"] == opponent) & (team_df["Venue"] == venue)]
    if len(opp_venue) > 0:
        avg_gf_vs_opp = safe_mean(opp_venue, "GF")
    else:
        avg_gf_vs_opp = avg_gf if avg_gf > 0 else 1.2

    return {
        "avg_shots": max(avg_shots, 0.1),
        "avg_sot":   max(avg_sot,   0.1),
        "avg_g_sh":  avg_g_sh,
        "avg_g_sot": avg_g_sot,
        "avg_gf":    max(avg_gf,    0.5),
        "avg_ga":    max(avg_ga,    0.5),
        "avg_gf_vs_opp": max(avg_gf_vs_opp, 0.5),
        "last_n": last_n,
    }

# ─────────────────────────────────────────
# 5. TRADITIONAL xG (from your original Step 7)
# ─────────────────────────────────────────
def traditional_xg(stats):
    xg_shots = stats["avg_shots"] * stats["avg_g_sh"]
    if xg_shots > 0:
        opp_factor  = stats["avg_gf_vs_opp"] / xg_shots
        xg_adjusted = xg_shots * opp_factor
    else:
        xg_adjusted = stats["avg_gf_vs_opp"]
    return max(round(xg_adjusted, 3), 0.1)

# ─────────────────────────────────────────
# 6. ML FEATURES (from your original Step 8)
#    Now built from player CSV aggregation
# ─────────────────────────────────────────
def build_ml_features(team_df, target_col="GF"):
    """
    Replicates your original build_features():
    Features: Sh, SoT, G/Sh, G/SoT, Venue_enc, Opp_enc,
              roll_GF, roll_GA, roll_Sh, roll_SoT
    Target: GF or GA
    """
    data = team_df.copy().reset_index(drop=True)
    if len(data) < 5:
        return None, None, None, None

    # Encode categoricals (from your original)
    venue_enc = LabelEncoder()
    opp_enc   = LabelEncoder()
    data["Venue_enc"] = venue_enc.fit_transform(data["Venue"].fillna("Home"))
    data["Opp_enc"]   = opp_enc.fit_transform(data["Opponent"].fillna("Unknown"))

    # Rolling 5-match form with shift(1) — no leakage (from your original)
    data["roll_GF"]  = data["GF"].shift(1).rolling(5, min_periods=1).mean()
    data["roll_GA"]  = data["GA"].shift(1).rolling(5, min_periods=1).mean()
    data["roll_Sh"]  = data["Sh"].shift(1).rolling(5, min_periods=1).mean()
    data["roll_SoT"] = data["SoT"].shift(1).rolling(5, min_periods=1).mean()

    feature_cols = ["Sh","SoT","G/Sh","G/SoT","Venue_enc","Opp_enc",
                    "roll_GF","roll_GA","roll_Sh","roll_SoT"]

    data = data.dropna(subset=feature_cols + [target_col])
    if len(data) < 5:
        return None, None, None, None

    X = data[feature_cols].values
    y = data[target_col].values

    return X, y, opp_enc, venue_enc

# ─────────────────────────────────────────
# 7. TRAIN RF + LR ENSEMBLE (your original Step 8)
# ─────────────────────────────────────────
def train_and_predict(X, y, query_row):
    """
    Exact replica of your original ML training:
    - Time-ordered train/test split (shuffle=False)
    - RF n_estimators=200, max_depth=6
    - LR baseline
    - Ensemble weighted by 1/MAE
    """
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                random_state=42, n_jobs=-1)
    lr = LinearRegression()
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    if len(X_test) > 0:
        rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
        lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
    else:
        rf_mae = lr_mae = 1.0

    rf_pred = max(float(rf.predict(query_row)[0]), 0.01)
    lr_pred = max(float(lr.predict(query_row)[0]), 0.01)

    # Inverse MAE weighting (your original)
    rf_w = 1 / (rf_mae + 1e-5)
    lr_w = 1 / (lr_mae + 1e-5)
    total_w = rf_w + lr_w

    ml_xg = round((rf_pred * rf_w + lr_pred * lr_w) / total_w, 3)
    return ml_xg, rf_mae, lr_mae

# ─────────────────────────────────────────
# 8. POISSON (your original Step 10-12)
# ─────────────────────────────────────────
def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def scoreline_matrix(lam_h, lam_a, max_goals=6):
    m = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            m[i][j] = poisson_prob(lam_h, i) * poisson_prob(lam_a, j)
    return m

# ─────────────────────────────────────────
# 9. MAIN PREDICT FUNCTION
# ─────────────────────────────────────────
def predict_match(players_dict, home_team, away_team,
                  home_xi, away_xi, match_date, season):
    """
    Full pipeline:
    1. Build team dataframes from XI player CSVs (before match_date)
    2. Compute rolling stats (your original Step 5+6)
    3. Traditional xG (your original Step 7)
    4. ML features + RF+LR ensemble (your original Step 8)
    5. Final xG = 60% ML + 40% traditional
    6. Poisson scoreline matrix (your original Step 10-12)
    """
    match_date = pd.to_datetime(match_date)
    xg_results = {}

    for team, xi, venue in [(home_team, home_xi, "Home"),
                             (away_team, away_xi, "Away")]:
        opponent = away_team if team == home_team else home_team

        # ── Build team-level df from XI player CSVs ──
        team_df = build_team_df(players_dict, xi, match_date)

        if len(team_df) == 0:
            xg_results[team] = 1.2
            continue

        # ── Rolling stats (your original Step 5+6) ──
        stats = get_rolling_stats(team_df, venue, opponent)

        # ── Traditional xG (your original Step 7) ──
        xg_trad = traditional_xg(stats)

        # ── ML features (your original Step 8) ──
        X, y, opp_enc, venue_enc = build_ml_features(team_df, target_col="GF")

        if X is not None and len(X) >= 5:
            # Encode query values
            try:
                opp_code   = opp_enc.transform([opponent])[0]
            except:
                opp_code   = 0
            try:
                venue_code = venue_enc.transform([venue])[0]
            except:
                venue_code = 0

            # Build query row (same features as your original)
            query_row = np.array([[
                stats["avg_shots"],
                stats["avg_sot"],
                stats["avg_g_sh"],
                stats["avg_g_sot"],
                venue_code,
                opp_code,
                stats["avg_gf"],
                stats["avg_ga"],
                stats["avg_shots"],
                stats["avg_sot"],
            ]])

            ml_xg, rf_mae, lr_mae = train_and_predict(X, y, query_row)

            # ── Final xG: 60% ML + 40% traditional (improvement) ──
            xg_final = round(0.6 * ml_xg + 0.4 * xg_trad, 3)
        else:
            # Not enough data → fall back to traditional xG
            xg_final = xg_trad

        xg_results[team] = max(xg_final, 0.1)

    # ── Poisson matrix (your original Step 10-12) ──
    lam_h  = xg_results[home_team]
    lam_a  = xg_results[away_team]
    matrix = scoreline_matrix(lam_h, lam_a)

    win  = float(np.sum(np.tril(matrix, -1)))
    draw = float(np.sum(np.diag(matrix)))
    loss = float(np.sum(np.triu(matrix, 1)))

    scores = [(i, j, matrix[i][j]) for i in range(7) for j in range(7)]
    top5   = sorted(scores, key=lambda x: -x[2])[:5]

    return {
        "xg_home":  lam_h,
        "xg_away":  lam_a,
        "home_win": round(win  * 100, 1),
        "draw":     round(draw * 100, 1),
        "away_win": round(loss * 100, 1),
        "top5":     [(f"{h}-{a}", round(p*100, 2)) for h, a, p in top5],
        "matrix":   matrix,
        "home_team":home_team,
        "away_team":away_team,
    }
