import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from sklearn.linear_model import Ridge

st.set_page_config(page_title="NFL Week Dashboard", page_icon="🏈")

today = dt.date.today()
SEASON = today.year if today.month >= 3 else today.year - 1
COLS = ["season", "week", "season_type", "play_type", "epa", "wp", "posteam",
        "defteam", "success", "yards_gained", "down", "game_id"]


@st.cache_data(ttl=6 * 3600, show_spinner="Loading NFL data...")
def load(season):
    pbp = nfl.load_pbp([season - 1, season]).select(COLS).to_pandas()
    sched = nfl.load_schedules(season).to_pandas()
    return pbp, sched


def fav(a, h, m):
    if pd.isna(m):
        return "n/a"
    if abs(m) < 0.05:
        return "Pick 'em"
    return "{} -{:.1f}".format(h if m > 0 else a, abs(m))


pbp, sched = load(SEASON)
todo = sched[sched["result"].isna() & (sched["game_type"] == "REG")]
if todo.empty:
    st.info("No upcoming regular-season games right now.")
    st.stop()
week = int(todo["week"].min())
now_idx = SEASON * 18 + week

# Passes and runs from last season and this season so far
allp = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["pass", "run"])]
p = allp.dropna(subset=["epa", "wp", "posteam", "defteam", "yards_gained", "success"])
p = p[(p["wp"] >= 0.05) & (p["wp"] <= 0.95)].copy()  # drop garbage time
p["w"] = 0.5 ** ((now_idx - (p["season"] * 18 + p["week"])) / 8)  # recent plays count more

# Opponent-adjusted team ratings
teams = sorted(p["posteam"].unique())
X = np.hstack([
    pd.get_dummies(p["posteam"]).reindex(columns=teams, fill_value=0).astype(float).values,
    pd.get_dummies(p["defteam"]).reindex(columns=teams, fill_value=0).astype(float).values,
])
model = Ridge(alpha=100).fit(X, p["epa"].values, sample_weight=p["w"].values)
net = pd.Series(model.coef_[:len(teams)] - model.coef_[len(teams):], index=teams)

# Team profiles: pass, run, success and big plays, ranked 1 (best) to 32 (worst)
p["wepa"] = p["epa"] * p["w"]
p["wsucc"] = p["success"] * p["w"]
p["wexp"] = np.where(p["play_type"] == "pass", p["yards_gained"] >= 20, p["yards_gained"] >= 10).astype(float) * p["w"]


def profile(col, prefix):
    out = {}
    for kind in ["pass", "run"]:
        gg = p[p["play_type"] == kind].groupby(col)
        out[prefix + kind] = gg["wepa"].sum() / gg["w"].sum()
    gg = p.groupby(col)
    out[prefix + "succ"] = gg["wsucc"].sum() / gg["w"].sum()
    out[prefix + "expl"] = gg["wexp"].sum() / gg["w"].sum()
    return pd.DataFrame(out)


prof = profile("posteam", "off_").join(profile("defteam", "def_"))
ranks = pd.DataFrame(index=prof.index)
for c in prof.columns:
    ranks[c] = prof[c].rank(ascending=c.startswith("def_")).astype(int)
rank_cols = list(ranks.columns)
ranks["plays_pg"] = (allp.groupby("posteam").size() / allp.groupby("posteam")["game_id"].nunique()).round(1)
neutral = allp[(allp["down"] <= 2) & allp["wp"].between(0.2, 0.8)]
ranks["pass_pct"] = (neutral.groupby("posteam")["play_type"].apply(lambda s: (s == "pass").mean()) * 100).round(0)

# This week's games
g = todo[todo["week"] == week].copy().sort_values("gameday")
g["model"] = (g["home_team"].map(net) - g["away_team"].map(net)) * 63 + 1.5
g["gap"] = g["model"] - g["spread_line"]

st.title("NFL Week " + str(week))
tab1, tab2, tab3 = st.tabs(["Games", "Matchups", "Teams"])

with tab1:
    sort_by = st.radio("Sort by", ["Date", "Points difference to spread"], horizontal=True)
    if sort_by == "Date":
        g = g.sort_values("gameday")
    else:
        g = g.reindex(g["gap"].abs().sort_values(ascending=False, na_position="last").index)

    for _, r in g.iterrows():
        a, h = r["away_team"], r["home_team"]
        with st.container(border=True):
            st.subheader(a + " at " + h)
            st.caption(str(r["gameday"]))
            c1, c2, c3 = st.columns(3)
            c1.metric("Market", fav(a, h, r["spread_line"]))
            c2.metric("Model", fav(a, h, r["model"]))
            c3.metric("Total", "-" if pd.isna(r["total_line"]) else str(r["total_line"]))
            if pd.notna(r["gap"]):
                st.write("Model vs market: {:.1f} pts toward {}".format(abs(r["gap"]), h if r["gap"] > 0 else a))
                if abs(r["gap"]) >= 5:
                    st.warning("Big gap. Check injury and lineup news before trusting it.")
            st.caption("{} vs {}. Rest {} and {} days.".format(
                r.get("away_qb_name", "?"), r.get("home_qb_name", "?"),
                r.get("away_rest", "?"), r.get("home_rest", "?")))

with tab2:
    st.caption("A plus number means that offence ranks better than the defence it faces.")
    m = pd.DataFrame({
        "Game": g["away_team"] + " at " + g["home_team"],
        "Away pass": g["home_team"].map(ranks["def_pass"]) - g["away_team"].map(ranks["off_pass"]),
        "Away run": g["home_team"].map(ranks["def_run"]) - g["away_team"].map(ranks["off_run"]),
        "Home pass": g["away_team"].map(ranks["def_pass"]) - g["home_team"].map(ranks["off_pass"]),
        "Home run": g["away_team"].map(ranks["def_run"]) - g["home_team"].map(ranks["off_run"]),
        "Plays": (g["away_team"].map(ranks["plays_pg"]) + g["home_team"].map(ranks["plays_pg"])).round(0),
    })
    edge_cols = ["Away pass", "Away run", "Home pass", "Home run"]
    st.dataframe(m.style.background_gradient(cmap="RdYlGn", vmin=-30, vmax=30, subset=edge_cols)
                 .format("{:.0f}", subset=edge_cols + ["Plays"]),
                 hide_index=True, use_container_width=True)

with tab3:
    st.caption("Ranks from 1 (best) to 32 (worst). O is offence, D is defence, Plays is per game, "
               "Pass % is passing share when the game is close.")
    view = ranks.sort_index()
    view.columns = ["O pass", "O run", "O succ", "O expl", "D pass", "D run", "D succ", "D expl", "Plays", "Pass %"]
    st.dataframe(view.style.background_gradient(cmap="RdYlGn_r", vmin=1, vmax=32, subset=list(view.columns[:8])),
                 use_container_width=True)
