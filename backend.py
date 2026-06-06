"""
backend.py — Bulletproof version
Handles ANY player CSV column structure gracefully
Core logic from original Poisson + RF reports
"""

import os, math, glob, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEASON_DATES = {
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

# All La Liga teams hardcoded — always show in dropdown
LALIGA_TEAMS = {
    "2024-2025": [
        "Barcelona","Real Madrid","Atlético Madrid","Athletic Club","Villarreal",
        "Real Betis","Celta Vigo","Rayo Vallecano","Osasuna","Mallorca",
        "Real Sociedad","Valencia","Getafe","Espanyol","Deportivo Alaves",
        "Girona","Sevilla","Leganes","Las Palmas","Valladolid"
    ],
    "2023-2024": [
        "Real Madrid","Barcelona","Girona","Atlético Madrid","Athletic Club",
        "Real Sociedad","Real Betis","Villarreal","Valencia","Deportivo Alaves",
        "Osasuna","Las Palmas","Getafe","Celta Vigo","Sevilla",
        "Cadiz","Mallorca","Granada","Almeria","Rayo Vallecano"
    ],
    "2022-2023": [
        "Barcelona","Real Madrid","Atlético Madrid","Real Sociedad","Villarreal",
        "Real Betis","Osasuna","Athletic Club","Rayo Vallecano","Mallorca",
        "Girona","Cadiz","Almeria","Getafe","Espanyol",
        "Celta Vigo","Sevilla","Valencia","Valladolid","Elche"
    ],
    "2021-2022": [
        "Real Madrid","Barcelona","Atlético Madrid","Sevilla","Real Betis",
        "Real Sociedad","Villarreal","Athletic Club","Valencia","Osasuna",
        "Celta Vigo","Rayo Vallecano","Elche","Getafe","Espanyol",
        "Deportivo Alaves","Levante","Mallorca","Granada","Cadiz"
    ],
    "2020-2021": [
        "Atlético Madrid","Real Madrid","Barcelona","Sevilla","Real Sociedad",
        "Real Betis","Villarreal","Celta Vigo","Athletic Club","Cadiz",
        "Levante","Valencia","Osasuna","Granada","Deportivo Alaves",
        "Elche","Getafe","Huesca","Valladolid","Eibar"
    ],
}

FORMATIONS = {
    "4-3-3":   ["GK","RB","CB","CB","LB","CM","CM","CM","RW","ST","LW"],
    "4-4-2":   ["GK","RB","CB","CB","LB","RM","CM","CM","LM","ST","ST"],
    "4-2-3-1": ["GK","RB","CB","CB","LB","DM","DM","AM","RW","LW","ST"],
    "4-5-1":   ["GK","RB","CB","CB","LB","RM","CM","CM","CM","LM","ST"],
    "4-1-4-1": ["GK","RB","CB","CB","LB","DM","RM","CM","CM","LM","ST"],
    "3-5-2":   ["GK","CB","CB","CB","RM","CM","CM","CM","LM","ST","ST"],
    "3-4-3":   ["GK","CB","CB","CB","RM","CM","CM","LM","RW","ST","LW"],
    "3-5-1-1": ["GK","CB","CB","CB","RM","CM","CM","CM","LM","AM","ST"],
    "5-3-2":   ["GK","RB","CB","CB","CB","LB","CM","CM","CM","ST","ST"],
    "5-4-1":   ["GK","RB","CB","CB","CB","LB","RM","CM","CM","LM","ST"],
    "4-4-1-1": ["GK","RB","CB","CB","LB","RM","CM","CM","LM","AM","ST"],
}

SLOT_TO_GROUP = {
    "GK":"GK","CB":"DEF","LB":"DEF","RB":"DEF",
    "DM":"MID","CM":"MID","LM":"MID","RM":"MID","AM":"MID",
    "ST":"FWD","LW":"FWD","RW":"FWD",
}

# ─────────────────────────────────────────
# COLUMN NAME ALIASES
# Maps any known variant → standard name
# ─────────────────────────────────────────
COL_ALIASES = {
    # Goals
    "Goals": ["Goals","G","Gls","Goal","goals"],
    # Assists
    "Assists": ["Assists","Ast","A","assists"],
    # Shots
    "Shots": ["Shots","Sh","shots","Shot"],
    # Shots on Target
    "SoT": ["SoT","shots_on_target","Shots on Target","ShotsOnTarget","sot"],
    # Minutes
    "Minutes": ["Minutes","Min","Mins","minutes","min"],
    # Team Goals
    "TeamGoals": ["TeamGoals","GF","team_goals","Goals For","GoalsFor"],
    # Opp Goals
    "OppGoals": ["OppGoals","GA","opp_goals","Goals Against","GoalsAgainst"],
    # Venue
    "Venue": ["Venue","venue","Home/Away","HomeAway","location"],
    # Opponent
    "Opponent": ["Opponent","opponent","Opp","opp","vs","Against"],
    # Result
    "Result": ["Result","result","Res","res","W/D/L"],
    # Tackles
    "TacklesWon": ["TacklesWon","Tkl","tackles_won","Tackles"],
    # Interceptions
    "Interceptions": ["Interceptions","Int","interceptions"],
    # Date
    "Date": ["Date","date","Match Date","MatchDate"],
    # Team
    "Team": ["Team","team","Club","club","Squad"],
    # Season
    "Season": ["Season","season","Szn"],
}

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
    # Extra replacements for characters NFKD does not decompose
    extra = {
        "ø":"o","Ø":"O","ð":"d","Ð":"D","þ":"th","æ":"ae",
        "Æ":"AE","ł":"l","Ł":"L","ß":"ss","đ":"d","ħ":"h",
        "ĸ":"k","ŋ":"n","ŧ":"t","ı":"i",
    }
    name = str(name)
    for char, repl in extra.items():
        name = name.replace(char, repl)
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if not unicodedata.combining(c)).lower().strip()

def safe_col(df, col):
    return float(df[col].sum()) if col in df.columns else 0.0

def safe_mean(df, col, default=0.0):
    return float(df[col].mean()) if col in df.columns and len(df) > 0 else default

# ─────────────────────────────────────────
# STANDARDISE COLUMNS
# Renames any known variant to standard name
# ─────────────────────────────────────────
def standardise_columns(df):
    """Rename any column variants to standard names."""
    rename_map = {}
    existing = set(df.columns.str.strip())

    for std_name, variants in COL_ALIASES.items():
        if std_name in existing:
            continue  # already correct
        for variant in variants:
            if variant in existing:
                rename_map[variant] = std_name
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()

    # Add missing required columns with defaults
    defaults = {
        "Goals": 0, "Assists": 0, "Shots": 0, "SoT": 0,
        "Minutes": 90, "TacklesWon": 0, "Interceptions": 0,
        "TeamGoals": 0, "OppGoals": 0,
        "Venue": "Home", "Opponent": "Unknown", "Result": "D"
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Convert numeric columns
    for col in ["Goals","Assists","Shots","SoT","Minutes","TacklesWon",
                "Interceptions","TeamGoals","OppGoals","Yellow","Red"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fix encoding in Opponent column
    if "Opponent" in df.columns:
        replacements = {"Ã©":"é","Ã¡":"á","Ã­":"í","Ã³":"ó","Ãº":"ú","Ã±":"ñ","Ã":"Á"}
        for bad, good in replacements.items():
            df["Opponent"] = df["Opponent"].str.replace(bad, good, regex=False)

    # Compute G/Sh and G/SoT
    df["G/Sh"]  = df.apply(lambda r: r["Goals"]/r["Shots"] if r["Shots"] > 0 else 0, axis=1)
    df["G/SoT"] = df.apply(lambda r: r["Goals"]/r["SoT"]   if r["SoT"]   > 0 else 0, axis=1)

    return df

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
            df.columns = df.columns.str.strip()

            # Find and parse Date column
            date_col = next((c for c in df.columns
                           for v in COL_ALIASES["Date"] if c == v), None)
            if date_col and date_col != "Date":
                df = df.rename(columns={date_col: "Date"})
            if "Date" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).copy()

            # Standardise all columns
            df = standardise_columns(df)
            players[name] = df

        except Exception as e:
            print(f"Skipping {os.path.basename(path)}: {e}")
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
        if player.startswith("_placeholder_"):
            result.setdefault(season, {}).setdefault(club, [])
            continue
        result.setdefault(season, {}).setdefault(club, []).append(player)
    return result

def get_all_seasons(season_data):
    csv_seasons    = set(season_data.keys())
    laliga_seasons = set(LALIGA_TEAMS.keys())
    return sorted(csv_seasons | laliga_seasons, reverse=True)

def get_clubs_for_season(season_data, season):
    csv_teams    = set(season_data.get(season, {}).keys())
    laliga_teams = set(LALIGA_TEAMS.get(season, []))
    return sorted(csv_teams | laliga_teams)

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
# 3. BUILD TEAM DATAFRAME FROM XI
# ─────────────────────────────────────────
def build_team_df(players_dict, xi_players, match_date):
    all_rows = []
    for player in xi_players:
        if not player or player not in players_dict:
            continue
        df = players_dict[player]
        past = df[df["Date"] < match_date].copy()
        if len(past) > 0:
            all_rows.append(past)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows)
    agg = combined.groupby("Date").agg(
        Venue    = ("Venue",     "first"),
        Opponent = ("Opponent",  "first"),
        Result   = ("Result",    "first"),
        GF       = ("TeamGoals", "first"),
        GA       = ("OppGoals",  "first"),
        Sh       = ("Shots",     "sum"),
        SoT      = ("SoT",       "sum"),
    ).reset_index().sort_values("Date")

    agg["G/Sh"]  = agg.apply(lambda r: r["GF"]/r["Sh"]  if r["Sh"]  > 0 else 0, axis=1)
    agg["G/SoT"] = agg.apply(lambda r: r["GF"]/r["SoT"] if r["SoT"] > 0 else 0, axis=1)
    return agg

# ─────────────────────────────────────────
# 4. ROLLING STATS (original Step 5+6)
# ─────────────────────────────────────────
def get_rolling_stats(team_df, venue, opponent, n=5):
    venue_matches = team_df[team_df["Venue"] == venue]
    last_n        = venue_matches.tail(n) if len(venue_matches) > 0 else team_df.tail(n)

    avg_shots = safe_mean(last_n, "Sh",   0.1)
    avg_sot   = safe_mean(last_n, "SoT",  0.1)
    avg_g_sh  = safe_mean(last_n, "G/Sh", 0.0)
    avg_g_sot = safe_mean(last_n, "G/SoT",0.0)
    avg_gf    = safe_mean(last_n, "GF",   1.2)
    avg_ga    = safe_mean(last_n, "GA",   1.0)

    opp_venue = team_df[(team_df["Opponent"] == opponent) & (team_df["Venue"] == venue)]
    avg_gf_vs_opp = safe_mean(opp_venue, "GF", avg_gf) if len(opp_venue) > 0 else avg_gf

    return {
        "avg_shots":     max(avg_shots, 0.1),
        "avg_sot":       max(avg_sot,   0.1),
        "avg_g_sh":      avg_g_sh,
        "avg_g_sot":     avg_g_sot,
        "avg_gf":        max(avg_gf,    0.5),
        "avg_ga":        max(avg_ga,    0.5),
        "avg_gf_vs_opp": max(avg_gf_vs_opp, 0.5),
        "last_n":        last_n,
    }

# ─────────────────────────────────────────
# 5. TRADITIONAL xG (original Step 7)
# ─────────────────────────────────────────
def traditional_xg(stats):
    xg_shots = stats["avg_shots"] * stats["avg_g_sh"]
    if xg_shots > 0:
        xg_adj = xg_shots * (stats["avg_gf_vs_opp"] / xg_shots)
    else:
        xg_adj = stats["avg_gf_vs_opp"]
    return max(round(xg_adj, 3), 0.1)

# ─────────────────────────────────────────
# 6. ML FEATURES (original Step 8)
# ─────────────────────────────────────────
def build_ml_features(team_df, target_col="GF"):
    data = team_df.copy().reset_index(drop=True)
    if len(data) < 5:
        return None, None, None, None

    venue_enc = LabelEncoder()
    opp_enc   = LabelEncoder()
    data["Venue_enc"] = venue_enc.fit_transform(data["Venue"].fillna("Home"))
    data["Opp_enc"]   = opp_enc.fit_transform(data["Opponent"].fillna("Unknown"))

    data["roll_GF"]  = data["GF"].shift(1).rolling(5, min_periods=1).mean()
    data["roll_GA"]  = data["GA"].shift(1).rolling(5, min_periods=1).mean()
    data["roll_Sh"]  = data["Sh"].shift(1).rolling(5, min_periods=1).mean()
    data["roll_SoT"] = data["SoT"].shift(1).rolling(5, min_periods=1).mean()

    feature_cols = ["Sh","SoT","G/Sh","G/SoT","Venue_enc","Opp_enc",
                    "roll_GF","roll_GA","roll_Sh","roll_SoT"]

    data = data.dropna(subset=feature_cols + [target_col])
    if len(data) < 5:
        return None, None, None, None

    return data[feature_cols].values, data[target_col].values, opp_enc, venue_enc

# ─────────────────────────────────────────
# 7. TRAIN RF + LR ENSEMBLE (original Step 8)
# ─────────────────────────────────────────
def train_and_predict(X, y, query_row):
    split   = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    lr = LinearRegression()
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    rf_mae = mean_absolute_error(y_test, rf.predict(X_test)) if len(X_test) > 0 else 1.0
    lr_mae = mean_absolute_error(y_test, lr.predict(X_test)) if len(X_test) > 0 else 1.0

    rf_pred = max(float(rf.predict(query_row)[0]), 0.01)
    lr_pred = max(float(lr.predict(query_row)[0]), 0.01)

    rf_w = 1 / (rf_mae + 1e-5)
    lr_w = 1 / (lr_mae + 1e-5)
    ml_xg = round((rf_pred * rf_w + lr_pred * lr_w) / (rf_w + lr_w), 3)
    return ml_xg

# ─────────────────────────────────────────
# 8. POISSON (original Steps 10-12)
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
# 9. MAIN PREDICT
# ─────────────────────────────────────────
def predict_match(players_dict, home_team, away_team,
                  home_xi, away_xi, match_date, season):
    match_date = pd.to_datetime(match_date)
    xg_results = {}

    for team, xi, venue in [(home_team, home_xi, "Home"),
                             (away_team, away_xi, "Away")]:
        opponent = away_team if team == home_team else home_team

        team_df = build_team_df(players_dict, xi, match_date)

        if len(team_df) == 0:
            xg_results[team] = 1.2
            continue

        stats   = get_rolling_stats(team_df, venue, opponent)
        xg_trad = traditional_xg(stats)

        X, y, opp_enc, venue_enc = build_ml_features(team_df, "GF")

        if X is not None:
            try:
                opp_code   = opp_enc.transform([opponent])[0]
            except:
                opp_code   = 0
            try:
                venue_code = venue_enc.transform([venue])[0]
            except:
                venue_code = 0

            query_row = np.array([[
                stats["avg_shots"], stats["avg_sot"],
                stats["avg_g_sh"],  stats["avg_g_sot"],
                venue_code, opp_code,
                stats["avg_gf"],    stats["avg_ga"],
                stats["avg_shots"], stats["avg_sot"],
            ]])

            ml_xg    = train_and_predict(X, y, query_row)
            xg_final = round(0.6 * ml_xg + 0.4 * xg_trad, 3)
        else:
            xg_final = xg_trad

        xg_results[team] = max(xg_final, 0.1)

    lam_h  = xg_results[home_team]
    lam_a  = xg_results[away_team]
    matrix = scoreline_matrix(lam_h, lam_a)
    win    = float(np.sum(np.tril(matrix, -1)))
    draw   = float(np.sum(np.diag(matrix)))
    loss   = float(np.sum(np.triu(matrix, 1)))
    scores = [(i, j, matrix[i][j]) for i in range(7) for j in range(7)]
    top5   = sorted(scores, key=lambda x: -x[2])[:5]

    return {
        "xg_home":  lam_h, "xg_away":  lam_a,
        "home_win": round(win*100,1), "draw": round(draw*100,1),
        "away_win": round(loss*100,1),
        "top5":     [(f"{h}-{a}", round(p*100,2)) for h,a,p in top5],
        "matrix":   matrix,
        "home_team":home_team, "away_team":away_team,
    }
