import os, base64
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
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}

.stApp {{
  background:
    radial-gradient(1200px 700px at 50% -10%, rgba(122,162,255,0.30), rgba(0,0,0,0) 55%),
    radial-gradient(900px 600px at 90% 20%, rgba(255,122,217,0.26), rgba(0,0,0,0) 55%),
    linear-gradient(180deg, rgba(6,10,22,0.55) 0%, rgba(6,10,22,0.82) 55%, rgba(6,10,22,0.93) 100%),
    url("data:image/jpg;base64,{bg_b64}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

.block-container {{ padding-top: 86px; max-width: 1320px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stToolbar"] {{ visibility:hidden; height:0px; }}

/* Hide sidebar + collapse arrow */
section[data-testid="stSidebar"] {{ display:none !important; }}
button[kind="header"][data-testid="collapsedControl"] {{ display:none !important; }}

/* Topbar */
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
  background: rgba(255,255,255,0.10);
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

h1,h2,h3,h4 {{ color: rgba(245,247,255,0.98) !important; }}

/* ---------- TEAM TILE ---------- */
.tile {{
  position: relative;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px;
  padding: 12px;
  text-align: center;
  min-height: 255px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease, background .12s ease;
  overflow: hidden;
}}
.tile:hover {{
  transform: translateY(-2px);
  box-shadow: 0 18px 60px rgba(0,0,0,0.30);
}}

/* Top band (always subtle) */
.tile::before {{
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 14px;
  background: rgba(255,255,255,0.10);
}}

.tileSelected {{
  border: 2px solid rgba(255,122,217,0.95) !important;
  box-shadow: 0 22px 70px rgba(255,122,217,0.22);
  background: rgba(255,122,217,0.12);
}}

/* Selected top band fill (the thing you asked) */
.tileSelected::before {{
  background: linear-gradient(90deg, rgba(122,162,255,0.95), rgba(255,122,217,0.95));
  height: 16px;
}}

/* Logo container + top border highlight */
.logoBox {{
  border-radius: 14px;
  padding: 8px;
  background: rgba(0,0,0,0.12);
  border-top: 3px solid rgba(255,255,255,0.14);
}}
.tileSelected .logoBox {{
  border-top: 3px solid rgba(255,122,217,0.95);  /* highlight only top border */
  background: rgba(255,122,217,0.10);           /* slight fill so user KNOWS selected */
}}

.tileName {{
  margin-top: 10px;
  font-weight: 900;
  color: rgba(245,247,255,0.95);
  font-size: 12px;
  min-height: 34px;
  display:flex;
  align-items:center;
  justify-content:center;
}}

.selBadge {{
  display:inline-block;
  margin-top: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 900;
  color: rgba(245,247,255,0.94);
  background: rgba(122,162,255,0.22);
  border: 1px solid rgba(122,162,255,0.65);
}}

/* Invisible click button: cover full tile area */
.tileClickWrap .stButton > button {{
  width: 100% !important;
  height: 255px !important;
  opacity: 0 !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  padding: 0 !important;
  margin-top: -255px !important;  /* overlay */
}}

/* ---------- PREDICTION CARD ---------- */
.predCard {{
  padding: 18px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.12);
}}
.predHead {{ display:flex; align-items:baseline; justify-content:space-between; gap: 12px; }}
.predTeam {{ font-weight: 900; font-size: 16px; color: rgba(245,247,255,0.98); }}
.predTag {{
  font-size: 11px; font-weight: 900;
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.10);
  color: rgba(245,247,255,0.92);
}}
.predPct {{ font-weight: 900; font-size: 46px; line-height: 1.0; margin-top: 8px; }}
.predBar {{
  height: 12px; border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
  overflow: hidden; margin-top: 12px;
}}
.predFill {{ height: 100%; border-radius: 999px; }}
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
# UI: Team tiles (full-tile click, no select button text)
# ----------------------------
def team_tile_grid(title, teams, selected_key):
    st.markdown(f"### {title}")
    st.markdown('<div class="small">Click a logo to select</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, t in enumerate(teams):
        with cols[i % 4]:
            selected = (st.session_state.get(selected_key) == t)
            tile_class = "tile tileSelected" if selected else "tile"
            badge_html = '<div class="selBadge">Selected</div>' if selected else ""

            logo = get_logo(t)

            st.markdown(f'<div class="{tile_class}">', unsafe_allow_html=True)

            # logo block
            st.markdown('<div class="logoBox">', unsafe_allow_html=True)
            if logo:
                st.image(logo, use_container_width=True)
            else:
                st.caption("Logo missing")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="tileName">{t}</div>{badge_html}', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Full-tile invisible click overlay
            st.markdown('<div class="tileClickWrap">', unsafe_allow_html=True)
            if st.button("", key=f"{selected_key}_{t}"):
                st.session_state[selected_key] = t
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
# Prediction card UI
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

# ----------------------------
# APP (Predictions only)
# ----------------------------
squads_df = load_squads()
ratings, comp_df = build_player_ratings_and_components()
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

    go = st.button("Go", type="primary", disabled=go_disabled)

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
