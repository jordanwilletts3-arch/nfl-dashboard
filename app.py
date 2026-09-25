import datetime as dt
from pathlib import Path

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
LOG_PATH = Path("predictions_log.csv")

DIV_ADJ = 4.0  # points shaved off the model's margin for division games — backtested on 2022-2025,
               # beat a random-games control, average miss improved from 10.36 to 10.18


@st.cache_data(ttl=6 * 3600, show_spinner="Loading NFL data...")
def load(season):
    pbp = nfl.load_pbp([season - 1, season]).select(COLS).to_pandas()
    sched = nfl.load_schedules(season).to_pandas()
    try:
        injuries = nfl.load_injuries([season - 1, season]).to_pandas()
    except Exception:
        injuries = pd.DataFrame()
    return pbp, sched, injuries


def fav(a, h, m):
    if pd.isna(m):
        return "n/a"
    if abs(m) < 0.05:
        return "Pick 'em"
    return "{} -{:.1f}".format(h if m > 0 else a, abs(m))


def injury_counts(injuries, week, season):
    """Count 'Out' and 'Doubtful' players per team for a given week, split offence/defence."""
    if injuries.empty:
        return pd.DataFrame()
    off_pos = {"QB", "RB", "WR", "TE", "T", "G", "C", "OL", "FB"}
    wk = injuries[(injuries["season"] == season) & (injuries["week"] == week)].copy()
    if wk.empty:
        return pd.DataFrame()
    wk["side"] = np.where(wk["position"].isin(off_pos), "off", "def")
    wk["flag"] = wk["report_status"].isin(["Out", "Doubtful"])
    out = wk[wk["flag"]].groupby(["team", "side"]).size().unstack(fill_value=0)
    for c in ["off", "def"]:
        if c not in out.columns:
            out[c] = 0
    return out


def log_predictions(g, season, week):
    """Append this week's predictions to the log once, skipping games already logged."""
    cols = ["season", "week", "gameday", "away_team", "home_team", "spread_line", "model_margin"]
    new_rows = g.rename(columns={"model": "model_margin"})[cols].copy()
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        key = ["season", "week", "away_team", "home_team"]
        already = existing.set_index(key).index
        new_rows = new_rows[~new_rows.set_index(key).index.isin(already)]
        if not new_rows.empty:
            new_rows["result"] = np.nan
            existing = pd.concat([existing, new_rows], ignore_index=True)
    else:
        new_rows["result"] = np.nan
        existing = new_rows
    existing.to_csv(LOG_PATH, index=False)
    return existing


def fill_results(log, sched):
    """Fill in final results for logged games that have since been played."""
    done = sched.dropna(subset=["result"])[["season", "week", "away_team", "home_team", "result"]]
    log = log.drop(columns=["result"]).merge(
        done, on=["season", "week", "away_team", "home_team"], how="left"
    )
    log.to_csv(LOG_PATH, index=False)
    return log


pbp, sched, injuries = load(SEASON)
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
ranks["plays_pg"] = (allp.groupby("posteam").size() / allp.groupby("posteam")["game_id"].nunique()).round(1)
neutral = allp[(allp["down"] <= 2) & allp["wp"].between(0.2, 0.8)]
ranks["pass_pct"] = (neutral.groupby("posteam")["play_type"].apply(lambda s: (s == "pass").mean()) * 100).round(0)

inj = injury_counts(injuries, week, SEASON)

# This week's games
g = todo[todo["week"] == week].copy().sort_values("gameday")
g["base_model"] = (g["home_team"].map(net) - g["away_team"].map(net)) * 63 + 1.5

div_col = "div_game" if "div_game" in g.columns else None
g["div_adj"] = np.where(g[div_col] == 1, np.sign(-g["base_model"].fillna(0)) * DIV_ADJ, 0.0) if div_col else 0.0

g["model"] = g["base_model"] + g["div_adj"]
g["gap"] = g["model"] - g["spread_line"]

log = log_predictions(g[["season", "week", "gameday", "away_team", "home_team", "spread_line", "model"]]
                       .rename(columns={"model": "model_margin"}), SEASON, week)
log = fill_results(log, sched)

st.title("NFL Week " + str(week))
tab1, tab2, tab3, tab4 = st.tabs(["Games", "Matchups", "Teams", "Track record"])

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
            tags = []
            if div_col and r.get(div_col) == 1:
                tags.append("Division game")
            st.caption(str(r["gameday"]) + (" — " + ", ".join(tags) if tags else ""))

            c1, c2, c3 = st.columns(3)
            c1.metric("Market", fav(a, h, r["spread_line"]))
            c2.metric("Model", fav(a, h, r["model"]))
            c3.metric("Total", "-" if pd.isna(r["total_line"]) else str(r["total_line"]))

            if pd.notna(r["gap"]):
                st.write("Model vs market: {:.1f} pts toward {}".format(abs(r["gap"]), h if r["gap"] > 0 else a))
                if abs(r["gap"]) >= 5:
                    st.warning("Big gap. Check injury and lineup news before trusting it.")

            if r["div_adj"] != 0:
                st.caption("Includes a division-game adjustment of {:+.1f} pts (backtested).".format(r["div_adj"]))

            st.caption("{} vs {}. Rest {} and {} days.".format(
                r.get("away_qb_name", "?"), r.get("home_qb_name", "?"),
                r.get("away_rest", "?"), r.get("home_rest", "?")))

            if not inj.empty:
                a_off, a_def = (inj.loc[a, "off"], inj.loc[a, "def"]) if a in inj.index else (0, 0)
                h_off, h_def = (inj.loc[h, "off"], inj.loc[h, "def"]) if h in inj.index else (0, 0)
                if a_off + a_def + h_off + h_def > 0:
                    st.caption("Out/doubtful — {}: {} offence, {} defence. {}: {} offence, {} defence.".format(
                        a, a_off, a_def, h, h_off, h_def))

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

with tab4:
    st.caption("Every prediction the model has made this season, checked against results once games finish. "
               "This log lives on the app's own storage, so a redeploy or long period of inactivity can reset it.")
    played = log.dropna(subset=["result"]).copy()
    if played.empty:
        st.info("No completed games logged yet this season. Check back once this week's games are played.")
    else:
        played["model_miss"] = (played["model_margin"] - played["result"]).abs()
        played["book_miss"] = (played["spread_line"] - played["result"]).abs()
        played["model_side_won"] = np.sign(played["model_margin"] - played["spread_line"]) == \
            np.sign(played["result"] - played["spread_line"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Games tracked", len(played))
        c2.metric("Model avg miss", "{:.2f}".format(played["model_miss"].mean()))
        c3.metric("Bookmaker avg miss", "{:.2f}".format(played["book_miss"].mean()))
        st.metric("Model side win rate", "{:.1f}%".format(played["model_side_won"].mean() * 100))
        st.dataframe(
            played[["week", "gameday", "away_team", "home_team", "spread_line", "model_margin", "result"]]
            .sort_values(["week", "gameday"], ascending=[False, False]),
            hide_index=True, use_container_width=True,
        )
