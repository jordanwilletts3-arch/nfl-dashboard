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


@st.cache_data(ttl=6 * 3600, show_spinner="Loading player stats...")
def load_player_stats(season):
    """Each player's rolling 4-game average, as of right now, for rec yards / rush yards / receptions.
    This is a plain rolling average — tested against an opponent-adjusted version and it did not help
    (see notes), so no defensive adjustment is applied here.
    """
    raw = nfl.load_pbp([season - 1, season]).select(
        ["season", "week", "season_type", "play_type", "posteam", "rusher_player_id", "rusher_player_name",
         "receiver_player_id", "receiver_player_name", "yards_gained", "complete_pass"]
    ).to_pandas()
    raw = raw[raw["season_type"] == "REG"].copy()
    raw["idx"] = raw["season"] * 18 + raw["week"]

    def rolling_avg(df, id_col, name_col, value_col, label):
        d = df.dropna(subset=[id_col]).groupby([id_col, name_col, "posteam", "idx"])[value_col].sum().reset_index()
        d = d.sort_values([id_col, "idx"])
        d["avg"] = d.groupby(id_col)[value_col].transform(lambda s: s.rolling(4, min_periods=1).mean())
        last = d.groupby(id_col).tail(1).copy()
        last["stat"] = label
        return last.rename(columns={name_col: "player", "posteam": "team"})[["player", "team", "stat", "avg"]]

    run = raw[raw["play_type"] == "run"]
    rec = raw[raw["play_type"] == "pass"].copy()
    rec["catch"] = rec["complete_pass"].fillna(0)

    out = pd.concat([
        rolling_avg(run, "rusher_player_id", "rusher_player_name", "yards_gained", "Rushing yards"),
        rolling_avg(rec, "receiver_player_id", "receiver_player_name", "yards_gained", "Receiving yards"),
        rolling_avg(rec, "receiver_player_id", "receiver_player_name", "catch", "Receptions"),
    ])
    return out.dropna(subset=["player"]).sort_values(["stat", "player"])


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


def injury_players(injuries, week, season):
    """Every player with a report this week, one row per player, for the detail view and search tab."""
    if injuries.empty:
        return pd.DataFrame()
    wk = injuries[(injuries["season"] == season) & (injuries["week"] == week)].copy()
    if wk.empty:
        return pd.DataFrame()
    name_col = "full_name" if "full_name" in wk.columns else "player_name"
    keep = wk[[name_col, "team", "position", "report_status", "practice_status"]].rename(
        columns={name_col: "player", "report_status": "status", "practice_status": "practice"})
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    keep["_ord"] = keep["status"].map(order).fillna(3)
    return keep.sort_values(["team", "_ord", "player"]).drop(columns="_ord")


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
inj_players = injury_players(injuries, week, SEASON)

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Games", "Matchups", "Teams", "Track record", "Injuries", "Player props"])

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

            if not inj_players.empty:
                game_players = inj_players[inj_players["team"].isin([a, h])]
                if not game_players.empty:
                    with st.expander("Player-by-player injury report ({} listed)".format(len(game_players))):
                        st.dataframe(game_players, hide_index=True, use_container_width=True)

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

with tab5:
    st.caption("This week's full injury report, every team. Context to read alongside the games above — "
               "it isn't used by the model.")
    if inj_players.empty:
        st.info("No injury report published for this week yet.")
    else:
        c1, c2 = st.columns(2)
        team_pick = c1.selectbox("Team", ["All"] + sorted(inj_players["team"].unique()))
        status_pick = c2.multiselect("Status", ["Out", "Doubtful", "Questionable"],
                                      default=["Out", "Doubtful", "Questionable"])
        view = inj_players[inj_players["status"].isin(status_pick)]
        if team_pick != "All":
            view = view[view["team"] == team_pick]
        st.dataframe(view, hide_index=True, use_container_width=True)

with tab6:
    st.caption("Each player's rolling 4-game average for the stat you pick, based on their most recent games "
               "(includes last season early on, weighted equally — no opponent adjustment, since testing showed "
               "it didn't improve on a plain average). This is an estimate to compare against a bookmaker's "
               "prop line yourself, not a tip, and single-game player stats are naturally noisy.")
    props = load_player_stats(SEASON)
    if props.empty:
        st.info("No player stats available yet.")
    else:
        c1, c2 = st.columns(2)
        stat_pick = c1.selectbox("Stat", sorted(props["stat"].unique()))
        team_filter = c2.selectbox("Team", ["All"] + sorted(props["team"].dropna().unique()), key="props_team")
        rows = props[props["stat"] == stat_pick]
        if team_filter != "All":
            rows = rows[rows["team"] == team_filter]
        search = st.text_input("Search player")
        if search:
            rows = rows[rows["player"].str.contains(search, case=False, na=False)]
        rows = rows.sort_values("avg", ascending=False).rename(columns={"avg": "Last-4-game average"})
        st.dataframe(rows[["player", "team", "Last-4-game average"]].round(1),
                     hide_index=True, use_container_width=True)
        st.caption("Enter a bookmaker's line below to compare it against the highlighted player's average.")
        if not rows.empty:
            chosen = st.selectbox("Compare a player", rows["player"])
            line = st.number_input("Bookmaker's prop line", min_value=0.0, step=0.5)
            avg = rows.loc[rows["player"] == chosen, "Last-4-game average"].iloc[0]
            if line > 0:
                st.metric("Model average vs line", "{:+.1f}".format(avg - line),
                          delta="{} the line".format("Over" if avg > line else "Under"))
