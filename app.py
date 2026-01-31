# app.py  (PSL 2.0 AI Match Predictor + Compliance Monitor)
# ---------------------------------------------------------
# Includes:
# - Two Tabs at top (Predictor / Compliance)
# - Compliance reads PSL02_Compliance_Log.xlsx (Matches + Appearances)
# - FIXED: Name matching via PlayerKey normalization (handles A.H. Asad Mughni vs Asad Mughni, (vc), dots, etc.)
# - FIXED: Clear file diagnostics (shows which file is being read, sheets, and row counts)
# ---------------------------------------------------------

import os, base64, re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="PSL 2.0 AI Match Predictor", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUADS_XLSX = os.path.join(BASE_DIR, "PSL_Team_Players.xlsx")
BG_IMAGE = os.path.join(BASE_DIR, "assets", "bg.jpg")
BRAND_IMAGE = os.path.join(BASE_DIR, "assets", "PSL brand.jpg")
LOGO_DIR = os.path.join(BASE_DIR, "team logos")

S01 = {
    "bat": os.path.join(BASE_DIR, "Season 01", "1441602_batting_leaderboard.csv"),
    "bowl": os.path.join(BASE_DIR, "Season 01", "1441602_bowling_leaderboard.csv"),
    "field": os.path.join(BASE_DIR, "Season 01", "1441602_fielding_leaderboard.csv"),
    "mvp": os.path.join(BASE_DIR, "Season 01", "1441602_mvp_leaderboard.csv"),
}
S02 = {
    "bat": os.path.join(BASE_DIR, "Season 02", "1786448_batting_leaderboard.csv"),
    "bowl": os.path.join(BASE_DIR, "Season 02", "1786448_bowling_leaderboard.csv"),
    "field": os.path.join(BASE_DIR, "Season 02", "1786448_fielding_leaderboard.csv"),
    "mvp": os.path.join(BASE_DIR, "Season 02", "1786448_mvp_leaderboard.csv"),
}

TEAM_LOGOS = {
    "Bubak Blasters": "Bubak.jpg",
    "Fazilpur Falcons": "Fazilpur.jpg",
    "Kot Bahadur Shah Bulls": "KBS.jpg",
    "Keamari Kings": "Keamari.jpg",
    "Mahmoodkot Mavericks": "MKM.jpg",
    "Macchike Mustangs": "Mustangs.jpg",
    "Port Qasim Panthers": "PortQasim.jpg",
    "Shikarpur Stallions": "Shikarpur.jpg",
}

# Compliance file (you update after every match)
COMPLIANCE_XLSX = os.path.join(BASE_DIR, "PSL02_Compliance_Log.xlsx")
MIN_MATCHES_REQUIRED = 2

# ----------------------------
# Model settings
# ----------------------------
W_RECENT = 0.68
W_BAT, W_BOWL, W_FIELD, W_MVP = 0.40, 0.40, 0.10, 0.10
PROB_SCALE = 3.2

# ----------------------------
# Helpers
# ----------------------------
def file_to_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def to_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return 0.0

def zscore(series: pd.Series) -> pd.Series:
    v = series.astype(float)
    mu = v.mean()
    sd = v.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        sd = 1.0
    out = (v - mu) / sd
    return out.replace([np.inf, -np.inf], 0).fillna(0)

def sigmoid(x):
    if np.isnan(x) or np.isinf(x):
        return 0.5
    return float(1 / (1 + np.exp(-x)))

def pick_name_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in cols:
        if str(c).strip().lower() in ["player", "player name", "name", "batsman", "bowler", "fielder"]:
            return c
    for c in cols:
        if "player" in str(c).lower() or "name" in str(c).lower():
            return c
    return cols[0]

def find_col(df: pd.DataFrame, options):
    cols_lower = [str(c).lower() for c in df.columns]
    for opt in options:
        if opt.lower() in cols_lower:
            return df.columns[cols_lower.index(opt.lower())]
    for opt in options:
        for i, c in enumerate(cols_lower):
            if opt.lower() in c:
                return df.columns[i]
    return None

def get_logo(team_name: str):
    fn = TEAM_LOGOS.get(team_name)
    if not fn:
        return None
    path = os.path.join(LOGO_DIR, fn)
    if os.path.exists(path):
        try:
            return Image.open(path)
        except:
            return None
    return None

def best_xi(team_squad, ratings_df, n=11):
    valid = [p for p in team_squad if p in ratings_df.index]
    valid_sorted = sorted(valid, key=lambda p: float(ratings_df.loc[p, "rating"]), reverse=True)
    return valid_sorted[:n]

def team_strength(xi, ratings_df):
    s = 0.0
    for p in xi:
        if p in ratings_df.index:
            v = float(ratings_df.loc[p, "rating"])
            if not np.isnan(v) and not np.isinf(v):
                s += v
    return float(s) if (not np.isnan(s) and not np.isinf(s)) else 0.0

# --- Name normalization to match squad vs appearances ---
# Handles: "A.H. Asad Mughni" vs "Asad Mughni", "(vc)", dots, extra spaces, etc.
def clean_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\(.*?\)", " ", s)          # remove (vc), (c), etc
    s = re.sub(r"[^a-z\s]", " ", s)         # remove dots/numbers/specials
    s = re.sub(r"\s+", " ", s).strip()
    # remove single-letter initials (a h asad -> asad) but keep normal names
    parts = [p for p in s.split() if len(p) > 1]
    return " ".join(parts)

# ----------------------------
# Data Load
# ----------------------------
@st.cache_data(show_spinner=False)
def load_squads():
    df = pd.read_excel(SQUADS_XLSX, sheet_name="Team Players")
    df["Team"] = df["Team"].astype(str).str.strip()
    df["Player"] = df["Player"].astype(str).str.strip()
    df = df[(df["Team"] != "") & (df["Team"].str.lower() != "nan") &
            (df["Player"] != "") & (df["Player"].str.lower() != "nan")]
    return df

# ----------------------------
# Compliance & Participation (PSL02_Compliance_Log.xlsx)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=10)
def load_compliance_log(path: str):
    """Loads Matches + Appearances sheets. TTL keeps it fresh after each match update."""
    if not os.path.exists(path):
        return pd.DataFrame(), pd.DataFrame()

    try:
        matches = pd.read_excel(path, sheet_name="Matches")
    except Exception:
        matches = pd.DataFrame()

    try:
        apps = pd.read_excel(path, sheet_name="Appearances")
    except Exception:
        apps = pd.DataFrame()

    if not apps.empty:
        for c in ["Team", "Player"]:
            if c in apps.columns:
                apps[c] = apps[c].astype(str).str.strip()
        if "MatchID" in apps.columns:
            apps["MatchID"] = pd.to_numeric(apps["MatchID"], errors="coerce")
        apps = apps.dropna(subset=[c for c in ["MatchID", "Team", "Player"] if c in apps.columns])
        apps = apps.drop_duplicates(subset=[c for c in ["MatchID", "Team", "Player"] if c in apps.columns]).reset_index(drop=True)

    return matches, apps


def compliance_matrix_page(squads_df: pd.DataFrame):
    st.subheader("📋 Team Participation Matrix")
    st.markdown(
        '<div class="small">Select a team to see a simple tick/cross view by match. '
        'No percentages — just who played.</div>',
        unsafe_allow_html=True
    )

    # --- Diagnostics (helps avoid reading wrong files) ---
    with st.expander("Diagnostics", expanded=False):
        st.write("BASE_DIR:", BASE_DIR)
        st.write("Compliance file:", COMPLIANCE_XLSX, "✅" if os.path.exists(COMPLIANCE_XLSX) else "❌ (missing)")
        st.write("Squads file:", SQUADS_XLSX, "✅" if os.path.exists(SQUADS_XLSX) else "❌ (missing)")
        if os.path.exists(COMPLIANCE_XLSX):
            try:
                xls = pd.ExcelFile(COMPLIANCE_XLSX)
                st.write("Compliance sheets:", xls.sheet_names)
            except Exception as e:
                st.write("Could not read compliance sheets:", e)

    matches_df, apps_df = load_compliance_log(COMPLIANCE_XLSX)

    if squads_df.empty:
        st.error("Squads file is empty or could not be loaded.")
        return

    teams = sorted(squads_df["Team"].dropna().unique().tolist())
    if not teams:
        st.error("No teams found in squads data.")
        return

    team = st.selectbox("Select Team", teams, index=0)

    # ---- Prefer mapper outputs if present (generated by your mapping script) ----
    player_master_path = os.path.join(BASE_DIR, "player_master.xlsx")
    apps_mapped_path = os.path.join(BASE_DIR, "appearances_mapped.xlsx")

    @st.cache_data(show_spinner=False, ttl=30)
    def load_player_master():
        if os.path.exists(player_master_path):
            pm = pd.read_excel(player_master_path)
            # make sure required cols exist
            if "Player" not in pm.columns and "player_name_raw" in pm.columns:
                pm["Player"] = pm["player_name_raw"]
            if "Team_canonical" not in pm.columns and "Team" in pm.columns:
                pm["Team_canonical"] = pm["Team"].astype(str).str.strip()
            if "player_name_key" not in pm.columns:
                src = "player_name_raw" if "player_name_raw" in pm.columns else "Player"
                pm["player_name_key"] = pm[src].apply(clean_name)
            return pm[["player_id","Team_canonical","Player","player_name_key"]].drop_duplicates()
        # fallback: build from squads
        tmp = squads_df.copy()
        tmp["Team_canonical"] = tmp["Team"].astype(str).str.strip()
        tmp["Player"] = tmp["Player"].astype(str).str.strip()
        tmp["player_name_key"] = tmp["Player"].apply(clean_name)
        keys = tmp[["player_name_key"]].drop_duplicates().sort_values("player_name_key").reset_index(drop=True)
        keys["player_id"] = ["P" + str(i + 1).zfill(4) for i in range(len(keys))]
        pm = tmp.merge(keys, on="player_name_key", how="left")
        return pm[["player_id","Team_canonical","Player","player_name_key"]].drop_duplicates()

    @st.cache_data(show_spinner=False, ttl=30)
    def load_appearances_mapped(apps_raw: pd.DataFrame, pm: pd.DataFrame):
        if os.path.exists(apps_mapped_path):
            am = pd.read_excel(apps_mapped_path)
            if "Team_canonical" not in am.columns and "Team" in am.columns:
                am["Team_canonical"] = am["Team"].astype(str).str.strip()
            if "player_name_key" not in am.columns and "Player" in am.columns:
                am["player_name_key"] = am["Player"].apply(clean_name)
            return am
        # fallback mapping
        if apps_raw.empty:
            return apps_raw
        tmp = apps_raw.copy()
        tmp["Team_canonical"] = tmp["Team"].astype(str).str.strip()
        tmp["player_name_key"] = tmp["Player"].apply(clean_name)
        lookup = pm[["player_id","Team_canonical","player_name_key"]].drop_duplicates()
        tmp = tmp.merge(lookup, on=["Team_canonical","player_name_key"], how="left")
        # global fallback ignoring team
        miss = tmp["player_id"].isna()
        if miss.any():
            gl = pm[["player_id","player_name_key"]].drop_duplicates()
            tmp2 = tmp.loc[miss].merge(gl, on="player_name_key", how="left", suffixes=("", "_g"))
            tmp.loc[miss, "player_id"] = tmp2["player_id_g"].values
        return tmp

    pm = load_player_master()
    apps_mapped = load_appearances_mapped(apps_df, pm)

    # ---- Determine team matches ----
    team_match_ids = []
    if not matches_df.empty and "MatchID" in matches_df.columns:
        md = matches_df.copy()
        md["MatchID"] = pd.to_numeric(md["MatchID"], errors="coerce")
        # Support flexible columns: TeamA/TeamB or Team A/Team B, etc.
        colA = find_col(md, ["TeamA","Team A","Team1","Team 1","Home","HomeTeam"])
        colB = find_col(md, ["TeamB","Team B","Team2","Team 2","Away","AwayTeam","Visitor"])
        if colA and colB:
            team_match_ids = md.loc[(md[colA] == team) | (md[colB] == team), "MatchID"].dropna().unique().tolist()
        elif "Team" in md.columns:
            team_match_ids = md.loc[md["Team"] == team, "MatchID"].dropna().unique().tolist()

    # fallback: derive from appearances
    if not team_match_ids and not apps_df.empty and "MatchID" in apps_df.columns:
        team_match_ids = apps_df.loc[apps_df["Team"] == team, "MatchID"].dropna().unique().tolist()

    team_match_ids = sorted([int(x) for x in team_match_ids if pd.notna(x)])

    if not team_match_ids:
        st.warning(f"No matches found for **{team}** in PSL02_Compliance_Log.xlsx.")
        return

    total_team_matches = len(team_match_ids)

    # Friendly headers: show opponent name (e.g., 'vs Keamari Kings (05-Jan)')
    label_by_mid = {mid: "vs" for mid in team_match_ids}

    if not matches_df.empty and "MatchID" in matches_df.columns:
        md = matches_df.copy()
        md["MatchID"] = pd.to_numeric(md["MatchID"], errors="coerce")
        if "MatchDate" in md.columns:
            md["MatchDate"] = pd.to_datetime(md["MatchDate"], errors="coerce")

        for _, r in md.iterrows():
            if pd.isna(r.get("MatchID")):
                continue
            mid = int(r["MatchID"])
            if mid not in label_by_mid:
                continue

            # Flexible team columns (TeamA/TeamB or Team A/Team B etc.)
            colA = find_col(md, ["TeamA","Team A","Team1","Team 1","Home","HomeTeam"])
            colB = find_col(md, ["TeamB","Team B","Team2","Team 2","Away","AwayTeam","Visitor"])
            opp = ""
            if colA and colB:
                ta = str(r.get(colA, "")).strip()
                tb = str(r.get(colB, "")).strip()
                if ta == str(team).strip():
                    opp = tb
                elif tb == str(team).strip():
                    opp = ta

            dt_txt = ""
            if "MatchDate" in md.columns and pd.notna(r.get("MatchDate")):
                dt_txt = r["MatchDate"].strftime("%d-%b")

            label = f"vs {opp}" if opp else "vs"
            if dt_txt:
                label += f" ({dt_txt})"

            label_by_mid[mid] = label

# ---- Build matrix rows ----
    squad_team = pm.loc[pm["Team_canonical"] == str(team).strip()].copy()
    if squad_team.empty:
        # fallback from squads_df if master doesn't include team
        squad_team = squads_df.loc[squads_df["Team"] == team].copy()
        squad_team["Team_canonical"] = squad_team["Team"].astype(str).str.strip()
        squad_team["Player"] = squad_team["Player"].astype(str).str.strip()
        squad_team["player_name_key"] = squad_team["Player"].apply(clean_name)
        keys = squad_team[["player_name_key"]].drop_duplicates().sort_values("player_name_key").reset_index(drop=True)
        keys["player_id"] = ["P" + str(i + 1).zfill(4) for i in range(len(keys))]
        squad_team = squad_team.merge(keys, on="player_name_key", how="left")

    team_apps = apps_mapped.copy()
    if not team_apps.empty:
        team_apps["MatchID"] = pd.to_numeric(team_apps["MatchID"], errors="coerce")
        team_apps = team_apps.loc[
            (team_apps["Team_canonical"] == str(team).strip()) &
            (team_apps["MatchID"].isin(team_match_ids))
        ].copy()

    rows = []
    for _, r in squad_team.sort_values("Player").iterrows():
        pid = r.get("player_id", "")
        pname = r.get("Player", "")
        played = set()
        if not team_apps.empty and pid:
            played = set(team_apps.loc[team_apps["player_id"] == pid, "MatchID"].dropna().astype(int).tolist())

        row = {"Player's Name": pname}
        for mid in team_match_ids:
            row[label_by_mid[mid]] = "✅" if mid in played else "❌"
        row["Total Team Matches"] = total_team_matches
        row["Matches Played"] = len(played)
        rows.append(row)

    matrix_df = pd.DataFrame(rows)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)
def build_component_scores(bat, bowl, field, mvp):
    # Batting
    bname = pick_name_col(bat)
    runs_c = find_col(bat, ["runs"])
    sr_c   = find_col(bat, ["sr", "strike rate"])
    avg_c  = find_col(bat, ["avg", "average"])
    inns_c = find_col(bat, ["inns", "innings"])
    f50_c  = find_col(bat, ["50s", "fifties"])
    f100_c = find_col(bat, ["100s", "centuries"])

    bat["__player__"] = bat[bname].astype(str).str.strip()
    bat_score = {}
    for _, r in bat.iterrows():
        p = r["__player__"]
        if not p or p.lower() == "nan":
            continue
        runs = to_num(r.get(runs_c, 0)) if runs_c else 0
        sr   = to_num(r.get(sr_c, 0)) if sr_c else 0
        avg  = to_num(r.get(avg_c, 0)) if avg_c else 0
        inns = max(1.0, to_num(r.get(inns_c, 1))) if inns_c else 1.0
        f50  = to_num(r.get(f50_c, 0)) if f50_c else 0
        f100 = to_num(r.get(f100_c, 0)) if f100_c else 0
        bat_score[p] = runs + (sr*0.6) + (avg*0.8) + (f50*10) + (f100*25) + (np.log(inns+1)*2)

    # Bowling
    wname = pick_name_col(bowl)
    wk_c  = find_col(bowl, ["wkts", "wickets"])
    eco_c = find_col(bowl, ["econ", "economy"])
    avg_c2= find_col(bowl, ["avg", "average"])
    sr_c2 = find_col(bowl, ["sr", "strike rate"])
    mat_c = find_col(bowl, ["mat", "matches"])

    bowl["__player__"] = bowl[wname].astype(str).str.strip()
    bowl_score = {}
    for _, r in bowl.iterrows():
        p = r["__player__"]
        if not p or p.lower() == "nan":
            continue
        wk  = to_num(r.get(wk_c, 0)) if wk_c else 0
        eco = to_num(r.get(eco_c, 0)) if eco_c else 0
        avg2= to_num(r.get(avg_c2, 0)) if avg_c2 else 0
        sr2 = to_num(r.get(sr_c2, 0)) if sr_c2 else 0
        mat = max(1.0, to_num(r.get(mat_c, 1))) if mat_c else 1.0
        bowl_score[p] = (wk*25) + (np.log(mat+1)*2) - (eco*8) - (avg2*0.6) - (sr2*0.4)

    # Fielding
    fname = pick_name_col(field)
    ct_c  = find_col(field, ["catches", "ct"])
    ro_c  = find_col(field, ["run out", "runouts", "ro"])

    field["__player__"] = field[fname].astype(str).str.strip()
    field_score = {}
    for _, r in field.iterrows():
        p = r["__player__"]
        if not p or p.lower() == "nan":
            continue
        ct = to_num(r.get(ct_c, 0)) if ct_c else 0
        ro = to_num(r.get(ro_c, 0)) if ro_c else 0
        field_score[p] = (ct*8) + (ro*10)

    # MVP
    mname = pick_name_col(mvp)
    pts_c = find_col(mvp, ["points", "pts", "score"])

    mvp["__player__"] = mvp[mname].astype(str).str.strip()
    mvp_score = {}
    for _, r in mvp.iterrows():
        p = r["__player__"]
        if not p or p.lower() == "nan":
            continue
        mvp_score[p] = to_num(r.get(pts_c, 0)) if pts_c else 0

    return bat_score, bowl_score, field_score, mvp_score

@st.cache_data(show_spinner=False)
def build_player_ratings_and_components():
    s1_maps = build_component_scores(
        pd.read_csv(S01["bat"]), pd.read_csv(S01["bowl"]), pd.read_csv(S01["field"]), pd.read_csv(S01["mvp"])
    )
    s2_maps = build_component_scores(
        pd.read_csv(S02["bat"]), pd.read_csv(S02["bowl"]), pd.read_csv(S02["field"]), pd.read_csv(S02["mvp"])
    )

    players = sorted(set().union(*[set(d.keys()) for d in s1_maps], *[set(d.keys()) for d in s2_maps]))

    def blended(comp_idx):
        v1 = pd.Series({p: float(s1_maps[comp_idx].get(p, 0)) for p in players})
        v2 = pd.Series({p: float(s2_maps[comp_idx].get(p, 0)) for p in players})
        z1 = zscore(v1); z2 = zscore(v2)
        return ((1 - W_RECENT)*z1 + W_RECENT*z2).replace([np.inf, -np.inf], 0).fillna(0)

    bat_z   = blended(0)
    bowl_z  = blended(1)
    field_z = blended(2)
    mvp_z   = blended(3)

    rating = (W_BAT*bat_z) + (W_BOWL*bowl_z) + (W_FIELD*field_z) + (W_MVP*mvp_z)
    rating = rating.replace([np.inf, -np.inf], 0).fillna(0)

    ratings_df = pd.DataFrame({"player": players, "rating": rating.values}).set_index("player")

    comp_df = pd.DataFrame({
        "Batting": bat_z,
        "Bowling": bowl_z,
        "Fielding": field_z,
        "MVP": mvp_z,
        "Overall": rating
    }).fillna(0)

    return ratings_df, comp_df

# ----------------------------
# UI Styling
# ----------------------------
bg_b64 = file_to_base64(BG_IMAGE)
brand_b64 = file_to_base64(BRAND_IMAGE)

st.markdown(
f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
html, body, [class*="css"] {{
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}}

.stApp {{
  background:
    radial-gradient(1200px 700px at 50% -10%, rgba(122,162,255,0.18), rgba(0,0,0,0) 55%),
    radial-gradient(900px 600px at 90% 20%, rgba(255,122,217,0.14), rgba(0,0,0,0) 55%),
    linear-gradient(180deg, rgba(6,10,22,0.38) 0%, rgba(6,10,22,0.62) 55%, rgba(6,10,22,0.74) 100%),
    url("data:image/jpg;base64,{bg_b64}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

.block-container {{
  padding-top: 86px;
  max-width: 1400px;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stToolbar"] {{ visibility:hidden; height:0px; }}

section[data-testid="stSidebar"] {{ display:none !important; }}
button[kind="header"][data-testid="collapsedControl"] {{ display:none !important; }}

.topbar {{
  position: fixed; top:0; left:0; right:0; z-index:999;
  height:60px; display:flex; align-items:center;
  padding: 0 18px;
  background: rgba(255,255,255,0.07);
  border-bottom: 1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(14px);
}}
.brandWrap {{ display:flex; align-items:center; gap:12px; }}
.brandImg {{
  height: 44px; width:auto; display:block;
  filter: drop-shadow(0 10px 24px rgba(0,0,0,0.55));
}}
.appTitle {{
  color: rgba(245,247,255,0.98);
  font-weight: 900;
  font-size: 16px;
  letter-spacing: 0.2px;
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: rgba(15,23,42,0.08);
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.36);
  backdrop-filter: blur(14px);
}}

.small {{
  color: rgba(235,242,255,0.92);
  font-size: 10.5px;
  margin-top: -6px;
}}
h1,h2,h3,h4 {{
  color: rgba(245,247,255,0.98) !important;
}}

.teamDivider {{
  width: 1px;
  height: 100%;
  background: linear-gradient(180deg, rgba(255,255,255,0.0), rgba(255,255,255,0.18), rgba(255,255,255,0.0));
  margin: 0 auto;
  border-radius: 999px;
  filter: drop-shadow(0 0 10px rgba(255,255,255,0.08));
}}

.tile {{
  position: relative;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 16px;
  padding: 10px;
  text-align: center;
  overflow: hidden;
}}
.tile:hover {{
  transform: translateY(-2px);
  box-shadow: 0 18px 60px rgba(0,0,0,0.30);
}}

.tile::before {{
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 12px;
  background: rgba(15,23,42,0.08);
}}

.tileSelected {{
  border: 2px solid rgba(255,122,217,0.95) !important;
  box-shadow: 0 22px 70px rgba(255,122,217,0.20);
  background: rgba(255,122,217,0.12);
}}
.tileSelected::before {{
  background: linear-gradient(90deg, rgba(122,162,255,0.95), rgba(255,122,217,0.95));
  height: 14px;
}}

.logoBox {{
  border-radius: 14px;
  padding: 6px;
  background: rgba(0,0,0,0.12);
  border-top: 3px solid rgba(255,255,255,0.14);
}}
.tileSelected .logoBox {{
  border-top: 3px solid rgba(255,122,217,0.95);
  background: rgba(255,122,217,0.10);
}}

.tileName {{
  margin-top: 8px;
  font-weight: 900;
  color: rgba(245,247,255,0.95);
  font-size: 11.5px;
  line-height: 1.15;
  min-height: 30px;
  display:flex;
  align-items:center;
  justify-content:center;
}}

.selBadge {{
  display:inline-block;
  margin-top: 6px;
  padding: 5px 9px;
  border-radius: 999px;
  font-size: 10.5px;
  font-weight: 900;
  color: rgba(245,247,255,0.94);
  background: rgba(122,162,255,0.22);
  border: 1px solid rgba(122,162,255,0.65);
}}

.tileBtn .stButton > button {{
  width: 100% !important;
  border-radius: 12px !important;
  padding: 8px 10px !important;
  font-weight: 900 !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  background: rgba(10,16,30,0.35) !important;
  color: rgba(245,247,255,0.96) !important;
  box-shadow: none !important;
}}
.tileBtn .stButton > button:hover {{
  background: rgba(255,255,255,0.18) !important;
}}

.stButton > button {{
  border-radius: 14px !important;
  font-weight: 900 !important;
}}
button[kind="primary"] {{
  color: #071021 !important;
  background: linear-gradient(135deg, #7aa2ff 0%, #ff7ad9 100%) !important;
  border: 0 !important;
}}

.predCard {{
  padding: 18px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.12);
}}
.predHead {{
  display:flex; align-items:baseline; justify-content:space-between; gap: 12px;
}}
.predTeam {{
  font-weight: 900; font-size: 16px; color: rgba(245,247,255,0.98);
}}
.predTag {{
  font-size: 11px; font-weight: 900;
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(15,23,42,0.08);
  color: rgba(245,247,255,0.92);
}}
.predPct {{
  font-weight: 900; font-size: 46px; line-height: 1.0; margin-top: 8px;
}}
.predBar {{
  height: 12px; border-radius: 999px;
  background: rgba(15,23,42,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  overflow: hidden; margin-top: 12px;
}}
.predFill {{
  height: 100%; border-radius: 999px;
}}

.appFooter {{
  margin-top: 40px;
  padding: 18px 10px;
  text-align: center;
  font-size: 11.5px;
  color: rgba(235,242,255,0.75);
  border-top: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  backdrop-filter: blur(10px);
}}
.appFooter strong {{
  color: rgba(245,247,255,0.95);
  font-weight: 800;
}}
.appFooter .copyright {{
  margin-top: 4px;
  font-size: 10.5px;
  color: rgba(220,230,255,0.65);
}}

/* Tabs visibility */
div[data-baseweb="tab-list"] {{
  gap: 8px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
  padding: 6px;
  border-radius: 14px;
  backdrop-filter: blur(12px);
}}
button[data-baseweb="tab"] {{
  color: rgba(245,247,255,0.92) !important;
  font-weight: 900 !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}}
button[data-baseweb="tab"]:hover {{
  background: rgba(255,255,255,0.12) !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: rgba(7,16,33,0.98) !important;
  background: linear-gradient(135deg, #7aa2ff 0%, #ff7ad9 100%) !important;
  border: 0 !important;
}}
button[data-baseweb="tab"]:focus {{
  outline: none !important;
  box-shadow: none !important;
}}

/* ---- Tabs visibility ---- */
.stTabs [data-baseweb="tab-list"]{{
  background: rgba(255,255,255,0.28);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 14px;
  padding: 6px 8px;
  backdrop-filter: blur(8px);
}}
.stTabs [data-baseweb="tab"]{{
  color: rgba(255,255,255,0.92) !important;
  font-weight: 800 !important;
  font-size: 15px !important;
  text-shadow: 0 1px 10px rgba(0,0,0,0.55);
}}
.stTabs [aria-selected="true"]{{
  background: rgba(255,255,255,0.20) !important;
  border-radius: 12px !important;
}}



/* --- Readability (safe) --- */
.small, .muted, .hint, .stCaption, [data-testid="stCaptionContainer"] {{
  color: rgba(255,255,255,0.92) !important;
}}
label {{
  color: rgba(255,255,255,0.92) !important;
}}

/* Keep placeholders visible on white inputs */
input::placeholder {{ color: rgba(100,116,139,0.90) !important; }}

/* Ensure button text stays readable on light buttons */
.stButton button {{ color: #0b1220 !important; }}

</style>

<div class="topbar">
  <div class="brandWrap">
    <img class="brandImg" src="data:image/jpg;base64,{brand_b64}" />
    <div class="appTitle">PSL 2.0 AI Match Predictor</div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# ----------------------------
# Tabs
# ----------------------------
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
tab_predictor, tab_compliance = st.tabs(["🏏 Match Predictor", "📋 Compliance"])

# ----------------------------
# UI: Team tiles
# ----------------------------
def team_tile_grid(title, teams, selected_key):
    st.markdown(f"### {title}")
    st.markdown('<div class="small">Click Select under logo</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, t in enumerate(teams):
        with cols[i % 4]:
            selected = (st.session_state.get(selected_key) == t)
            tile_class = "tile tileSelected" if selected else "tile"
            badge_html = '<div class="selBadge">Selected</div>' if selected else ""

            logo = get_logo(t)

            st.markdown(f'<div class="{tile_class}">', unsafe_allow_html=True)
            st.markdown('<div class="logoBox">', unsafe_allow_html=True)
            if logo:
                st.image(logo, use_container_width=True)
            else:
                st.caption("Logo missing")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="tileName">{t}</div>{badge_html}', unsafe_allow_html=True)

            st.markdown('<div class="tileBtn">', unsafe_allow_html=True)
            if st.button("Select", key=f"{selected_key}_{t}"):
                st.session_state[selected_key] = t
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Player stats popover + chart (labels)
# ----------------------------
def stats_chart_with_labels(player, comp_df):
    if player not in comp_df.index:
        st.info("No stats found for this player in S01/S02 leaderboards.")
        return

    s = comp_df.loc[player, ["Batting", "Bowling", "Fielding", "MVP", "Overall"]].astype(float)
    df = pd.DataFrame({"Metric": s.index, "Score": s.values})

    bar = alt.Chart(df).mark_bar().encode(
        x=alt.X("Metric:N", sort=None),
        y=alt.Y("Score:Q"),
        tooltip=["Metric", alt.Tooltip("Score:Q", format=".2f")]
    )

    labels = alt.Chart(df).mark_text(dy=-8, fontWeight="bold").encode(
        x=alt.X("Metric:N", sort=None),
        y=alt.Y("Score:Q"),
        text=alt.Text("Score:Q", format=".2f")
    )

    st.altair_chart((bar + labels).properties(height=220), use_container_width=True)

def player_stats_popover(team_name, squad, comp_df):
    with st.popover("Player stats"):
        p = st.selectbox(f"Pick player ({team_name})", squad, key=f"stats_{team_name}")
        st.caption(p)
        stats_chart_with_labels(p, comp_df)

# ----------------------------
# XI Selector
# ----------------------------
def xi_editor(team_name, squad, ratings_df, state_key, comp_df):
    if state_key not in st.session_state:
        st.session_state[state_key] = best_xi(squad, ratings_df, 11)

    player_stats_popover(team_name, squad, comp_df)

    search = st.text_input(f"Search players ({team_name})", "", key=f"search_{state_key}")

    df = pd.DataFrame({"Player": squad})
    df["Rating"] = df["Player"].apply(lambda p: float(ratings_df.loc[p, "rating"]) if p in ratings_df.index else 0.0)
    df["In XI"] = df["Player"].isin(st.session_state[state_key])
    df = df.sort_values("Rating", ascending=False).reset_index(drop=True)

    if search.strip():
        s = search.strip().lower()
        df = df[df["Player"].str.lower().str.contains(s)].reset_index(drop=True)

    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "In XI": st.column_config.CheckboxColumn("In XI"),
            "Rating": st.column_config.NumberColumn("Rating", format="%.2f"),
        },
        disabled=["Player", "Rating"],
        height=380
    )

    chosen = edited.loc[edited["In XI"] == True, "Player"].tolist()

    if len(chosen) > 11:
        full = pd.DataFrame({"Player": squad})
        full["Rating"] = full["Player"].apply(lambda p: float(ratings_df.loc[p, "rating"]) if p in ratings_df.index else 0.0)
        keep = full[full["Player"].isin(chosen)].sort_values("Rating", ascending=False)["Player"].tolist()[:11]
        chosen = keep
        st.warning("You selected more than 11. I kept the best 11 (by rating).")

    st.session_state[state_key] = chosen
    st.caption(f"Selected: {len(chosen)}/11")
    return chosen

# ----------------------------
# Prediction cards
# ----------------------------
def pred_theme(pct):
    if pct >= 65:
        return ("#7CFFCB", "High chance", "linear-gradient(90deg,#7CFFCB,#7AA2FF)")
    if pct >= 50:
        return ("#FFD98E", "Slight edge", "linear-gradient(90deg,#FFD98E,#FF7AD9)")
    if pct >= 35:
        return ("#FFB3C7", "Underdog", "linear-gradient(90deg,#FFB3C7,#7AA2FF)")
    return ("#FF88A6", "Low chance", "linear-gradient(90deg,#FF88A6,#FF7AD9)")

def prediction_card(team, pct, strength):
    accent, tag, grad = pred_theme(pct)
    st.markdown(
        f"""
        <div class="predCard">
          <div class="predHead">
            <div class="predTeam">{team}</div>
            <div class="predTag">{tag}</div>
          </div>
          <div class="predPct" style="color:{accent};">{pct}%</div>
          <div class="small">XI strength: {strength:.2f}</div>
          <div class="predBar">
            <div class="predFill" style="width:{pct}%; background:{grad};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# TAB 1: Compliance Monitor
# =========================================================
with tab_compliance:
    squads_df = load_squads()
    with st.container(border=True):
        compliance_matrix_page(squads_df)

# =========================================================
# TAB 2: Match Predictor
# =========================================================
with tab_predictor:
    squads_df = load_squads()
    ratings, comp_df = build_player_ratings_and_components()
    teams = sorted(squads_df["Team"].unique().tolist())

    with st.container(border=True):
        st.subheader("Team Selection")

        colA, colMid, colB = st.columns([1, 0.045, 1], vertical_alignment="top")
        with colA:
            team_tile_grid("Team A", teams, "team_a")
        with colMid:
            st.markdown('<div class="teamDivider"></div>', unsafe_allow_html=True)
        with colB:
            team_tile_grid("Team B", teams, "team_b")

        sel_a = st.session_state.get("team_a")
        sel_b = st.session_state.get("team_b")

        go_disabled = (not sel_a) or (not sel_b) or (sel_a == sel_b)
        if sel_a and sel_b and sel_a == sel_b:
            st.warning("Select two different teams.")

        go = st.button("Go", type="primary", disabled=go_disabled)

        # Show quick squad lists as soon as both teams are selected
        if sel_a and sel_b and sel_a != sel_b:
            a_squad = squads_df.loc[squads_df["Team"] == sel_a, "Player"].dropna().astype(str).tolist()
            b_squad = squads_df.loc[squads_df["Team"] == sel_b, "Player"].dropna().astype(str).tolist()
            a_tbl = pd.DataFrame({"Player (Team A)": a_squad})
            b_tbl = pd.DataFrame({"Player (Team B)": b_squad})

    if "go_done" not in st.session_state:
        st.session_state.go_done = False

    if go:
        st.session_state.go_done = True
        for k in ["xi_a", "xi_b", "search_xi_a", "search_xi_b"]:
            if k in st.session_state:
                del st.session_state[k]

    if st.session_state.go_done:
        team_a = st.session_state.get("team_a")
        team_b = st.session_state.get("team_b")

        if not team_a or not team_b or team_a == team_b:
            st.stop()

        squad_a = squads_df.loc[squads_df["Team"] == team_a, "Player"].tolist()
        squad_b = squads_df.loc[squads_df["Team"] == team_b, "Player"].tolist()

        with st.container(border=True):
            st.subheader("Playing XI")
            st.markdown('<div class="small">Tick players in XI (exactly 11). Use Player stats popover for quick stats.</div>', unsafe_allow_html=True)

            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                if st.button(f"Auto-pick Best XI: {team_a}", use_container_width=True):
                    st.session_state["xi_a"] = best_xi(squad_a, ratings, 11)
            with b2:
                if st.button(f"Auto-pick Best XI: {team_b}", use_container_width=True):
                    st.session_state["xi_b"] = best_xi(squad_b, ratings, 11)
            with b3:
                if st.button("Reset Both", use_container_width=True):
                    st.session_state["xi_a"] = best_xi(squad_a, ratings, 11)
                    st.session_state["xi_b"] = best_xi(squad_b, ratings, 11)

            left, right = st.columns(2)

            with left:
                with st.container(border=True):
                    logo = get_logo(team_a)
                    if logo:
                        st.image(logo, width=90)
                    st.markdown(f"### {team_a}")
                    _ = xi_editor(team_a, squad_a, ratings, "xi_a", comp_df)

            with right:
                with st.container(border=True):
                    logo = get_logo(team_b)
                    if logo:
                        st.image(logo, width=90)
                    st.markdown(f"### {team_b}")
                    _ = xi_editor(team_b, squad_b, ratings, "xi_b", comp_df)

        can_predict = (len(st.session_state.get("xi_a", [])) == 11 and len(st.session_state.get("xi_b", [])) == 11)

        with st.container(border=True):
            predict = st.button("Predict", type="primary", disabled=not can_predict)
            if not can_predict:
                st.warning("Select exactly 11 players for both teams.")

        if predict:
            xi_a = st.session_state["xi_a"]
            xi_b = st.session_state["xi_b"]

            sA = team_strength(xi_a, ratings)
            sB = team_strength(xi_b, ratings)

            pA = sigmoid((sA - sB) / PROB_SCALE)
            pctA = int(round(pA * 100))
            pctB = 100 - pctA

            c1, c2 = st.columns(2)
            with c1:
                prediction_card(team_a, pctA, sA)
            with c2:
                prediction_card(team_b, pctB, sB)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <div class="appFooter">
        Created by <strong>IT Digitalization Team</strong> (Arman Bari &amp; M Zohaib Hassan)
        <div class="copyright">
            © 2026 PSL 2.0. All rights reserved.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
