import os, base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="PSL 2.0 AI App", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUADS_XLSX = os.path.join(BASE_DIR, "PSL_Team_Players.xlsx")
BG_IMAGE = os.path.join(BASE_DIR, "assets", "bg.jpg")
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

# Match your folder filenames exactly
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

# ----------------------------
# Hidden model settings (simple + stable)
# ----------------------------
W_RECENT = 0.68
W_BAT, W_BOWL, W_FIELD, W_MVP = 0.40, 0.40, 0.10, 0.10
PROB_SCALE = 3.2

# ----------------------------
# UI Styling
# ----------------------------
def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_b64 = file_to_base64(BG_IMAGE) if os.path.exists(BG_IMAGE) else ""

st.markdown(
f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}

.stApp {{
  background:
    linear-gradient(180deg, rgba(8,12,24,0.68) 0%, rgba(8,12,24,0.84) 55%, rgba(8,12,24,0.92) 100%),
    url("data:image/jpg;base64,{bg_b64}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

.block-container {{ padding-top: 74px; max-width: 1300px; }}

header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stToolbar"] {{ visibility:hidden; height:0px; }}

/* Top bar */
.topbar {{
  position: fixed; top:0; left:0; right:0; z-index:999;
  height:60px; display:flex; align-items:center; justify-content:space-between;
  padding: 0 18px;
  background: rgba(255,255,255,0.07);
  border-bottom: 1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(14px);
}}
.brand {{ display:flex; align-items:center; gap:10px; }}
.badge {{
  width: 34px; height: 34px; border-radius: 12px;
  background: linear-gradient(135deg, #7aa2ff 0%, #ff7ad9 100%);
  box-shadow: 0 16px 40px rgba(0,0,0,0.35);
}}
.title {{
  color: rgba(245,247,255,0.96);
  font-weight: 900;
  font-size: 16px;
  margin: 0;
}}

/* Premium bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: rgba(255,255,255,0.09);
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.36);
  backdrop-filter: blur(14px);
}}

/* Buttons */
.stButton > button {{
  border-radius: 14px !important;
  padding: 11px 14px !important;
  font-weight: 900 !important;
  border: 0 !important;
  color: #0b1020 !important;
  background: linear-gradient(135deg, #ffffff 0%, #dfe7ff 60%, #ffd7f1 100%) !important;
  box-shadow: 0 14px 34px rgba(0,0,0,0.28) !important;
}}
button[kind="primary"] {{
  color: #0b1020 !important;
  background: linear-gradient(135deg, #7aa2ff 0%, #ff7ad9 100%) !important;
}}
.stButton > button:disabled {{ opacity: 0.55 !important; }}

/* Text */
h1,h2,h3,h4 {{ color: rgba(245,247,255,0.96) !important; }}
.small {{ color: rgba(225,232,255,0.78); font-size: 12.5px; }}

/* Team Tile */
.tile {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 10px;
  text-align: center;
}}
.tileSelected {{
  border: 2px solid rgba(255,122,217,0.85) !important;
  box-shadow: 0 18px 50px rgba(255,122,217,0.18);
}}
.tileName {{
  margin-top: 8px;
  font-weight: 900;
  color: rgba(245,247,255,0.92);
  font-size: 12.5px;
}}
/* Fix st.data_editor look slightly */
div[data-testid="stDataFrame"] {{
  background: rgba(255,255,255,0.05);
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.14);
  overflow: hidden;
}}
</style>

<div class="topbar">
  <div class="brand">
    <div class="badge"></div>
    <div class="title">PSL 2.0 AI App</div>
  </div>
</div>
""",
unsafe_allow_html=True
)

# ----------------------------
# Data helpers
# ----------------------------
def pick_name_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in cols:
        if c.strip().lower() in ["player", "player name", "name", "batsman", "bowler", "fielder"]:
            return c
    for c in cols:
        if "player" in c.lower() or "name" in c.lower():
            return c
    return cols[0]

def find_col(df: pd.DataFrame, options):
    cols_lower = [c.lower() for c in df.columns]
    for opt in options:
        if opt.lower() in cols_lower:
            return df.columns[cols_lower.index(opt.lower())]
    for opt in options:
        for i, c in enumerate(cols_lower):
            if opt.lower() in c:
                return df.columns[i]
    return None

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

@st.cache_data(show_spinner=False)
def load_squads():
    df = pd.read_excel(SQUADS_XLSX, sheet_name="Team Players")
    df["Team"] = df["Team"].astype(str).str.strip()
    df["Player"] = df["Player"].astype(str).str.strip()
    return df

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
def build_player_ratings():
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

    bat_z, bowl_z, field_z, mvp_z = blended(0), blended(1), blended(2), blended(3)
    rating = (W_BAT*bat_z) + (W_BOWL*bowl_z) + (W_FIELD*field_z) + (W_MVP*mvp_z)
    rating = rating.replace([np.inf, -np.inf], 0).fillna(0)

    return pd.DataFrame({"player": players, "rating": rating.values}).set_index("player")

def get_logo(team_name: str):
    fn = TEAM_LOGOS.get(team_name)
    if not fn:
        return None
    path = os.path.join(LOGO_DIR, fn)
    if os.path.exists(path):
        return Image.open(path)
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
    if np.isnan(s) or np.isinf(s):
        return 0.0
    return float(s)

# ----------------------------
# Team Picker (logo tiles)
# ----------------------------
def team_tile_grid(title, teams, selected_key):
    st.markdown(f"### {title}")
    st.markdown('<div class="small">Click a logo to select.</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, t in enumerate(teams):
        col = cols[i % 4]
        with col:
            selected = (st.session_state.get(selected_key) == t)

            # Tile container
            tile_class = "tile tileSelected" if selected else "tile"
            st.markdown(f'<div class="{tile_class}">', unsafe_allow_html=True)

            logo = get_logo(t)
            if logo:
                st.image(logo, use_container_width=True)
            else:
                st.caption("Logo missing")

            st.markdown(f'<div class="tileName">{t}</div>', unsafe_allow_html=True)

            # Button under logo (keeps UX consistent)
            if st.button("Select", key=f"{selected_key}_{t}", use_container_width=True):
                st.session_state[selected_key] = t

            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# XI Selector (table + checkbox)
# ----------------------------
def xi_editor(team_name, squad, ratings_df, state_key):
    # Initialize XI
    if state_key not in st.session_state:
        st.session_state[state_key] = best_xi(squad, ratings_df, 11)

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

    # Enforce max 11 (if user checks >11, keep best 11 by rating)
    if len(chosen) > 11:
        # keep top 11 by rating from the FULL squad dataframe
        full = pd.DataFrame({"Player": squad})
        full["Rating"] = full["Player"].apply(lambda p: float(ratings_df.loc[p, "rating"]) if p in ratings_df.index else 0.0)
        keep = full[full["Player"].isin(chosen)].sort_values("Rating", ascending=False)["Player"].tolist()[:11]
        chosen = keep
        st.warning("You selected more than 11. I kept the best 11 (by rating).")

    st.session_state[state_key] = chosen

    st.caption(f"Selected: {len(chosen)}/11")
    return chosen

# ----------------------------
# App
# ----------------------------
tab_pred, tab_rec = st.tabs(["Predictions", "Recommendations"])

with tab_pred:
    squads_df = load_squads()
    ratings = build_player_ratings()

    teams = sorted(squads_df["Team"].unique().tolist())

    with st.container(border=True):
        st.subheader("Team Selection")

        cA, cB = st.columns(2)
        with cA:
            team_tile_grid("Team A", teams, "team_a")
        with cB:
            team_tile_grid("Team B", teams, "team_b")

        sel_a = st.session_state.get("team_a")
        sel_b = st.session_state.get("team_b")

        go_disabled = (not sel_a) or (not sel_b) or (sel_a == sel_b)

        if sel_a and sel_b and sel_a == sel_b:
            st.warning("Select two different teams.")

        go = st.button("Go", type="primary", use_container_width=True, disabled=go_disabled)

    if "go_done" not in st.session_state:
        st.session_state.go_done = False

    if go:
        st.session_state.go_done = True

        # reset XI when teams change
        for k in ["xi_a", "xi_b"]:
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
            st.markdown('<div class="small">Tick players in XI (exactly 11). Use Auto-pick buttons for speed.</div>', unsafe_allow_html=True)

            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                if st.button("Auto-pick Best XI (Team A)", use_container_width=True):
                    st.session_state["xi_a"] = best_xi(squad_a, ratings, 11)
            with b2:
                if st.button("Auto-pick Best XI (Team B)", use_container_width=True):
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
                    xi_a = xi_editor(team_a, squad_a, ratings, "xi_a")

            with right:
                with st.container(border=True):
                    logo = get_logo(team_b)
                    if logo:
                        st.image(logo, width=90)
                    st.markdown(f"### {team_b}")
                    xi_b = xi_editor(team_b, squad_b, ratings, "xi_b")

        can_predict = (len(st.session_state.get("xi_a", [])) == 11 and len(st.session_state.get("xi_b", [])) == 11)

        with st.container(border=True):
            predict = st.button("Predict", type="primary", use_container_width=True, disabled=not can_predict)
            if not can_predict:
                st.warning("Select exactly 11 players for both teams.")

        if predict:
            xi_a = st.session_state["xi_a"]
            xi_b = st.session_state["xi_b"]

            sA = team_strength(xi_a, ratings)
            sB = team_strength(xi_b, ratings)

            pA = sigmoid((sA - sB) / PROB_SCALE)
            pB = 1 - pA

            r1, r2 = st.columns(2)
            with r1:
                with st.container(border=True):
                    st.markdown(f"### {team_a}")
                    st.metric("Win %", f"{int(round(pA*100))}%")
                    st.caption(f"XI strength: {sA:.2f}")

            with r2:
                with st.container(border=True):
                    st.markdown(f"### {team_b}")
                    st.metric("Win %", f"{int(round(pB*100))}%")
                    st.caption(f"XI strength: {sB:.2f}")

with tab_rec:
    with st.container(border=True):
        st.subheader("Recommendations")
        st.caption("Tell me what you want here (batting order, bowling plan, field positions), I’ll implement next.")
