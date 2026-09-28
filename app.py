import datetime as dt
import re
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
        "defteam", "success", "yards_gained", "down", "game_id",
        "ydstogo", "yardline_100", "touchdown", "first_down"]
LOG_PATH = Path("predictions_log.csv")
BETS_PATH = Path("bets_log.csv")
PROP_LINES_PATH = Path("prop_lines.csv")
BET_COLS = ["id", "date_added", "game_date", "matchup", "bet_type", "bet", "legs",
            "tag", "odds", "stake", "status", "settled_date"]

DIV_ADJ = 4.0  # points shaved off the model's margin for division games — backtested on 2022-2025,
               # beat a random-games control, average miss improved from 10.36 to 10.18


@st.cache_data(ttl=6 * 3600, show_spinner="Loading player stats...")
def load_player_stats(season):
    """Each player's rolling 4-game average, as of right now, for rec yards / rush yards / receptions
    (a plain rolling average — tested against an opponent-adjusted version and it did not help, so no
    defensive adjustment is folded into this number), plus each defence's rank against that player's
    position over its last 4 games, shown separately as context rather than baked into the projection.
    """
    raw = nfl.load_pbp([season - 1, season]).select(
        ["season", "week", "season_type", "play_type", "posteam", "defteam",
         "rusher_player_id", "rusher_player_name", "receiver_player_id", "receiver_player_name",
         "yards_gained", "complete_pass"]
    ).to_pandas()
    raw = raw[raw["season_type"] == "REG"].copy()
    raw["idx"] = raw["season"] * 18 + raw["week"]

    try:
        pos = nfl.load_players().to_pandas()[["gsis_id", "position"]].dropna().drop_duplicates("gsis_id")
        pos_map = pos.set_index("gsis_id")["position"]
    except Exception:
        pos_map = pd.Series(dtype=object)

    def rolling_avg(df, id_col, name_col, value_col, label):
        d = df.dropna(subset=[id_col]).groupby([id_col, name_col, "posteam", "idx"])[value_col].sum().reset_index()
        d = d.sort_values([id_col, "idx"])
        d["avg"] = d.groupby(id_col)[value_col].transform(lambda s: s.rolling(4, min_periods=1).mean())
        last = d.groupby(id_col).tail(1).copy()
        last["stat"] = label
        last["position"] = last[id_col].map(pos_map)
        return last.rename(columns={name_col: "player", "posteam": "team"})[["player", "team", "position", "stat", "avg"]]

    run = raw[raw["play_type"] == "run"]
    rec = raw[raw["play_type"] == "pass"].copy()
    rec["catch"] = rec["complete_pass"].fillna(0)

    out = pd.concat([
        rolling_avg(run, "rusher_player_id", "rusher_player_name", "yards_gained", "Rushing yards"),
        rolling_avg(rec, "receiver_player_id", "receiver_player_name", "yards_gained", "Receiving yards"),
        rolling_avg(rec, "receiver_player_id", "receiver_player_name", "catch", "Receptions"),
    ])
    out = out.dropna(subset=["player"]).sort_values(["stat", "player"])

    # Defence vs position group: yards allowed per game to WR/TE/RB, last 4 games, ranked 1 (stingiest) to 32
    rec_pos = rec.dropna(subset=["receiver_player_id"]).copy()
    rec_pos["position"] = rec_pos["receiver_player_id"].map(pos_map)
    run_pos = run.dropna(subset=["rusher_player_id"]).copy()
    run_pos["position"] = run_pos["rusher_player_id"].map(pos_map)
    run_pos["defteam"] = run.loc[run_pos.index, "defteam"]

    def def_rank(df, positions):
        d = df[df["position"].isin(positions)]
        per_game = d.groupby(["defteam", "idx"])["yards_gained"].sum().reset_index().sort_values(["defteam", "idx"])
        per_game["last4"] = per_game.groupby("defteam")["yards_gained"].transform(
            lambda s: s.rolling(4, min_periods=1).mean())
        latest = per_game.groupby("defteam").tail(1).set_index("defteam")["last4"]
        return latest.rank(ascending=True).astype(int)  # 1 = fewest yards allowed = toughest matchup

    def_ranks = {
        "RB": def_rank(run_pos, ["RB", "FB"]),
        "WR": def_rank(rec_pos, ["WR"]),
        "TE": def_rank(rec_pos, ["TE"]),
    }
    return out, def_ranks


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


def load_bets():
    if BETS_PATH.exists():
        b = pd.read_csv(BETS_PATH)
        for c in BET_COLS:
            if c not in b.columns:
                b[c] = "Single" if c == "bet_type" else ""
        text_cols = ["game_date", "matchup", "bet_type", "bet", "legs", "tag", "status", "settled_date", "date_added"]
        for c in text_cols:
            b[c] = b[c].fillna("").astype(str)
        return b[BET_COLS]
    return pd.DataFrame(columns=BET_COLS)


def load_prop_lines():
    if PROP_LINES_PATH.exists():
        return pd.read_csv(PROP_LINES_PATH)
    return pd.DataFrame(columns=["player", "stat", "line", "updated"])


def save_prop_line(player, stat, line):
    lines = load_prop_lines()
    lines = lines[~((lines["player"] == player) & (lines["stat"] == stat))]
    row = pd.DataFrame([{"player": player, "stat": stat, "line": line,
                          "updated": dt.date.today().isoformat()}])
    lines = pd.concat([lines, row], ignore_index=True)
    lines.to_csv(PROP_LINES_PATH, index=False)
    return lines


STAT_KEYWORDS = [
    ("receiving yards", "Receiving yards"),
    ("rushing yards", "Rushing yards"),
    ("passing yards", "Passing yards"),
    ("receptions", "Receptions"),
]


def parse_prop_list(text):
    """Parse a pasted list like:
    * Ja'Marr Chase (WR): Over/Under 82.5 Receiving Yards | Over/Under 6.5 Receptions
    * Dawson Knox (TE): Receptions Under 1.5 (-189)
    Returns a list of (player, stat, line) tuples. Lines that don't match a player/stat
    pattern (team headers, blank lines) are silently skipped.
    """
    found = []
    for raw in text.splitlines():
        m = re.match(r"^\*?\s*(.+?)\s*\([A-Za-z]+\)\s*:\s*(.+)$", raw.strip())
        if not m:
            continue
        player, rest = m.group(1).strip(), m.group(2)
        for segment in rest.split("|"):
            seg_lower = segment.lower()
            stat_label = next((label for kw, label in STAT_KEYWORDS if kw in seg_lower), None)
            if not stat_label:
                continue
            num = re.search(r"(\d+\.\d+)", segment)
            if num:
                found.append((player, stat_label, float(num.group(1))))
    return found


def add_bet(bets, game_date, matchup, bet_type, bet_desc, legs, tag, odds, stake):
    next_id = int(bets["id"].max()) + 1 if not bets.empty else 1
    row = pd.DataFrame([{
        "id": next_id, "date_added": dt.date.today().isoformat(), "game_date": str(game_date),
        "matchup": matchup, "bet_type": bet_type, "bet": bet_desc, "legs": legs,
        "tag": tag, "odds": odds, "stake": stake,
        "status": "Pending", "settled_date": "",
    }])
    bets = pd.concat([bets, row], ignore_index=True)
    bets.to_csv(BETS_PATH, index=False)
    return bets


def settle_bet(bets, bet_id, status):
    bets.loc[bets["id"] == bet_id, "status"] = status
    bets.loc[bets["id"] == bet_id, "settled_date"] = dt.date.today().isoformat()
    bets.to_csv(BETS_PATH, index=False)
    return bets


def add_leg_callback():
    """Runs before the page redraws, so it can safely append the leg and clear the text box."""
    leg = st.session_state.get("new_leg_text", "").strip()
    if leg:
        st.session_state.setdefault("acca_legs", []).append(leg)
        st.session_state.new_leg_text = ""


def bet_profit(row):
    """Profit/loss for one settled bet. Odds are entered decimal-style (e.g. 1.91, 2.50, 8.40 for an acca)."""
    if row["status"] == "Won":
        return row["stake"] * (row["odds"] - 1)
    if row["status"] == "Lost":
        return -row["stake"]
    return 0.0  # Pending or Push


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
model = Ridge(alpha=500).fit(X, p["epa"].values, sample_weight=p["w"].values)
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

# Red zone: share of plays inside the 20 that go for a touchdown (a play-level proxy, not per-trip)
rz = allp.dropna(subset=["yardline_100", "touchdown"])
rz = rz[rz["yardline_100"] <= 20]
rz_off = rz.groupby("posteam")["touchdown"].mean() * 100
rz_def = rz.groupby("defteam")["touchdown"].mean() * 100
ranks["rz_off"] = rz_off.rank(ascending=False).astype(int)
ranks["rz_def"] = rz_def.rank(ascending=True).astype(int)

# Third down: conversion rate for and against
td3 = allp.dropna(subset=["down", "first_down"])
td3 = td3[td3["down"] == 3]
td3_off = td3.groupby("posteam")["first_down"].mean() * 100
td3_def = td3.groupby("defteam")["first_down"].mean() * 100
ranks["td3_off"] = td3_off.rank(ascending=False).astype(int)
ranks["td3_def"] = td3_def.rank(ascending=True).astype(int)

# Keep the raw percentages too, for a plainer display alongside the ranks
raw_pct = pd.DataFrame({
    "Red zone TD% (off)": rz_off.round(0), "Red zone TD% allowed (def)": rz_def.round(0),
    "3rd down % (off)": td3_off.round(0), "3rd down % allowed (def)": td3_def.round(0),
})

inj = injury_counts(injuries, week, SEASON)
inj_players = injury_players(injuries, week, SEASON)

# This week's games
g = todo[todo["week"] == week].copy()
if "gametime" in g.columns:
    g["kickoff"] = g["gametime"].fillna("TBD")
else:
    g["kickoff"] = "TBD"
g = g.sort_values(["gameday", "kickoff"])
POINTS_PER_EPA = 47.57  # fitted on 2022 data, tested on 2023-2025 — replaces a guessed value of 63
HOME_FIELD = 1.97        # fitted the same way — replaces a guessed value of 1.5
g["base_model"] = (g["home_team"].map(net) - g["away_team"].map(net)) * POINTS_PER_EPA + HOME_FIELD

div_col = "div_game" if "div_game" in g.columns else None
g["div_adj"] = np.where(g[div_col] == 1, np.sign(-g["base_model"].fillna(0)) * DIV_ADJ, 0.0) if div_col else 0.0

g["model"] = g["base_model"] + g["div_adj"]
g["gap"] = g["model"] - g["spread_line"]
g_all = g.copy()  # kept unfiltered so each tab's kickoff-time filter works independently

log = log_predictions(g[["season", "week", "gameday", "away_team", "home_team", "spread_line", "model"]]
                       .rename(columns={"model": "model_margin"}), SEASON, week)
log = fill_results(log, sched)

st.title("NFL Week " + str(week))
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Games", "Matchups", "Teams", "Track record", "Injuries", "Player props", "My bets"])

with tab1:
    times_available = sorted(g_all["kickoff"].unique())
    c1, c2 = st.columns([2, 1])
    sort_by = c1.radio("Sort by", ["Kickoff time", "Points difference to spread"], horizontal=True)
    time_pick = c2.multiselect("Kickoff time", times_available, default=times_available, key="games_time")

    g1 = g_all[g_all["kickoff"].isin(time_pick)]
    if sort_by == "Kickoff time":
        g1 = g1.sort_values(["gameday", "kickoff"])
    else:
        g1 = g1.reindex(g1["gap"].abs().sort_values(ascending=False, na_position="last").index)

    for _, r in g1.iterrows():
        a, h = r["away_team"], r["home_team"]
        with st.container(border=True):
            st.subheader(a + " at " + h)
            tags = []
            if div_col and r.get(div_col) == 1:
                tags.append("Division game")
            when = str(r["gameday"]) + (" at " + r["kickoff"] if r["kickoff"] != "TBD" else "")
            st.caption(when + (" — " + ", ".join(tags) if tags else ""))

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
    time_pick2 = st.multiselect("Kickoff time", times_available, default=times_available, key="matchups_time")
    g2 = g_all[g_all["kickoff"].isin(time_pick2)].sort_values(["gameday", "kickoff"])
    m = pd.DataFrame({
        "Game": g2["away_team"] + " at " + g2["home_team"],
        "Kickoff": g2["gameday"].astype(str) + " " + g2["kickoff"],
        "Away pass": g2["home_team"].map(ranks["def_pass"]) - g2["away_team"].map(ranks["off_pass"]),
        "Away run": g2["home_team"].map(ranks["def_run"]) - g2["away_team"].map(ranks["off_run"]),
        "Home pass": g2["away_team"].map(ranks["def_pass"]) - g2["home_team"].map(ranks["off_pass"]),
        "Home run": g2["away_team"].map(ranks["def_run"]) - g2["home_team"].map(ranks["off_run"]),
        "Plays": (g2["away_team"].map(ranks["plays_pg"]) + g2["home_team"].map(ranks["plays_pg"])).round(0),
    })
    edge_cols = ["Away pass", "Away run", "Home pass", "Home run"]
    st.dataframe(m.style.background_gradient(cmap="RdYlGn", vmin=-30, vmax=30, subset=edge_cols)
                 .format("{:.0f}", subset=edge_cols + ["Plays"]),
                 hide_index=True, use_container_width=True)

with tab3:
    st.caption("Ranks from 1 (best) to 32 (worst). O is offence, D is defence, Plays is per game, "
               "Pass % is passing share when the game is close.")
    view = ranks.sort_index()
    view = view[["off_pass", "off_run", "off_succ", "off_expl", "def_pass", "def_run", "def_succ", "def_expl",
                 "plays_pg", "pass_pct"]]
    view.columns = ["O pass", "O run", "O succ", "O expl", "D pass", "D run", "D succ", "D expl", "Plays", "Pass %"]
    st.dataframe(view.style.background_gradient(cmap="RdYlGn_r", vmin=1, vmax=32, subset=list(view.columns[:8])),
                 use_container_width=True)

    st.subheader("Red zone and third down")
    st.caption("Ranks from 1 (best) to 32 (worst). Red zone offence/defence is the share of plays inside the "
               "20 that go for a touchdown, a play-level approximation rather than a per-trip conversion rate. "
               "Third down is the conversion rate for and against. Context only, not used by the model.")
    rz3 = ranks[["rz_off", "rz_def", "td3_off", "td3_def"]].sort_index()
    rz3.columns = ["Red zone O", "Red zone D", "3rd down O", "3rd down D"]
    st.dataframe(rz3.style.background_gradient(cmap="RdYlGn_r", vmin=1, vmax=32),
                 use_container_width=True)
    with st.expander("Show the underlying percentages"):
        st.dataframe(raw_pct.sort_index().round(0), use_container_width=True)

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
    st.caption("Each player's rolling 4-game average for the stat you pick (no opponent adjustment — testing "
               "showed it didn't improve on a plain average, so it isn't folded into this number). The "
               "opponent's rank against that position over its last 4 games is shown separately as context "
               "you can weigh yourself, the same way the Matchups tab works for the spread. Neither is a tip, "
               "and single-game player stats are naturally noisy.")
    props, def_ranks = load_player_stats(SEASON)
    prop_lines = load_prop_lines()
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

        st.subheader("Bulk-import bookmaker lines")
        st.caption("Paste a list in the format: Player Name (POS): Over/Under 82.5 Receiving Yards | "
                   "Over/Under 6.5 Receptions — one player per line, team headers and blank lines are "
                   "ignored automatically. Works with Receiving yards, Rushing yards, Passing yards, "
                   "and Receptions.")
        bulk_text = st.text_area("Paste your list here", height=180, key="bulk_props_text")
        if st.button("Import lines"):
            parsed = parse_prop_list(bulk_text)
            if not parsed:
                st.warning("Couldn't find any recognisable player/stat lines in that text.")
            else:
                for player, stat, val in parsed:
                    save_prop_line(player, stat, val)
                prop_lines = load_prop_lines()
                st.success("Imported {} lines.".format(len(parsed)))
                with st.expander("Show what was imported"):
                    st.dataframe(pd.DataFrame(parsed, columns=["player", "stat", "line"]),
                                 hide_index=True, use_container_width=True)
                st.rerun()

        st.subheader("Add or update a single bookmaker line")
        c3, c4 = st.columns([3, 1])
        line_player = c3.selectbox("Player", rows["player"], key="line_player") if not rows.empty else None
        line_value = c4.number_input("Line", min_value=0.0, step=0.5, key="line_value")
        if st.button("Save line") and line_player:
            prop_lines = save_prop_line(line_player, stat_pick, line_value)
            st.success("Saved.")
            st.rerun()

        this_stat_lines = prop_lines[prop_lines["stat"] == stat_pick][["player", "line"]]
        rows = rows.merge(this_stat_lines, on="player", how="left").rename(columns={"line": "Bookmaker line"})
        rows["Difference"] = rows["Last-4-game average"] - rows["Bookmaker line"]

        st.subheader("Projections vs bookmaker lines")
        st.caption("Difference is the average minus the line — positive means the average sits over the line, "
                   "negative means under. Blank means no line has been entered for that player yet.")
        display_cols = ["player", "team", "position", "Last-4-game average", "Bookmaker line", "Difference"]
        st.dataframe(
            rows[display_cols].round(1).style.background_gradient(
                cmap="RdYlGn", subset=["Difference"], vmin=-10, vmax=10),
            hide_index=True, use_container_width=True,
        )

        st.caption("Pick a player and their upcoming opponent to see the matchup context alongside their average.")
        if not rows.empty:
            chosen = st.selectbox("Player", rows["player"], key="detail_player")
            prow = rows[rows["player"] == chosen].iloc[0]
            opp = st.selectbox("This week's opponent", sorted(net.index))
            pos_key = prow["position"] if prow["position"] in def_ranks else None

            c1, c2 = st.columns(2)
            c1.metric(prow["stat"] + " (last 4)", "{:.1f}".format(prow["Last-4-game average"]))
            if pos_key and opp in def_ranks[pos_key].index:
                rank = int(def_ranks[pos_key][opp])
                c2.metric(opp + " vs " + pos_key + " (rank, last 4)", "{} of 32".format(rank),
                           help="1 = toughest matchup (fewest yards allowed to this position), "
                                "32 = easiest matchup.")
                if rank <= 8:
                    st.warning("Tough matchup — this defence has allowed the fewest yards to the position over its last 4 games.")
                elif rank >= 25:
                    st.success("Favourable matchup — this defence has allowed the most yards to the position over its last 4 games.")
                else:
                    st.caption("Middle-of-the-pack matchup — nothing unusual either way over the last 4 games.")
            else:
                c2.metric("Opponent matchup", "n/a", help="Not enough data yet for this position or team.")

            if pd.notna(prow["Bookmaker line"]):
                st.metric("Average vs saved line", "{:+.1f}".format(prow["Difference"]),
                          delta="{} the line".format("Over" if prow["Difference"] > 0 else "Under"))
            else:
                st.caption("No line saved yet for this player at this stat — add one above.")

with tab7:
    st.caption("Your own betting log — not the model's predictions. Saved on the app's own storage, which "
               "isn't guaranteed to survive every restart, so download a backup after adding bets.")
    bets = load_bets()
    TAGS = ["JW", "RN", "Bet Club"]
    if "acca_legs" not in st.session_state:
        st.session_state.acca_legs = []

    st.subheader("Add a bet")
    bet_type = st.radio("Bet type", ["Single", "Player prop acca"], horizontal=True, key="bet_type_pick")
    c1, c2 = st.columns(2)
    game_date = c1.date_input("Game date", key="bet_date")
    tag = c2.selectbox("Tag", TAGS, key="bet_tag")
    matchup = st.text_input("Matchup or slate (e.g. KC at MIA, or 'Sunday slate' for an acca)", key="bet_matchup")

    if bet_type == "Single":
        bet_desc = st.text_input("What did you bet? (e.g. KC -10.0)", key="bet_desc")
        c3, c4 = st.columns(2)
        odds = c3.number_input("Odds (decimal, e.g. 1.91)", min_value=1.01, value=1.91, step=0.01,
                                format="%.2f", key="single_odds")
        stake = c4.number_input("Stake ($)", min_value=0.0, value=10.0, step=5.0, key="single_stake")
        if st.button("Add bet"):
            if matchup and bet_desc:
                bets = add_bet(bets, game_date, matchup, "Single", bet_desc, "", tag, odds, stake)
                st.success("Added.")
                st.rerun()
            else:
                st.warning("Fill in the matchup and the bet before adding.")

    else:
        st.write("Build the acca one prop line at a time, then save the whole bet below.")
        c5, c6 = st.columns([3, 1])
        c5.text_input("Add a prop line (e.g. Mahomes 250+ passing yards)", key="new_leg_text")
        c6.button("Add leg", on_click=add_leg_callback)

        if st.session_state.acca_legs:
            st.write("**Current legs:**")
            for i, leg in enumerate(st.session_state.acca_legs):
                lc1, lc2 = st.columns([5, 1])
                lc1.write(str(i + 1) + ". " + leg)
                if lc2.button("Remove", key="remove_leg_" + str(i)):
                    st.session_state.acca_legs.pop(i)
                    st.rerun()
        else:
            st.caption("No legs added yet.")

        c3, c4 = st.columns(2)
        odds = c3.number_input("Combined odds (decimal, e.g. 8.40)", min_value=1.01, value=1.91, step=0.01,
                                format="%.2f", key="acca_odds")
        stake = c4.number_input("Stake ($)", min_value=0.0, value=10.0, step=5.0, key="acca_stake")

        if st.button("Add acca"):
            if matchup and st.session_state.acca_legs:
                bet_desc = "Acca ({} legs)".format(len(st.session_state.acca_legs))
                legs_str = "\n".join(st.session_state.acca_legs)
                bets = add_bet(bets, game_date, matchup, "Acca", bet_desc, legs_str, tag, odds, stake)
                st.session_state.acca_legs = []
                st.success("Added.")
                st.rerun()
            else:
                st.warning("Add the matchup/slate and at least one leg before saving.")

    st.divider()

    if bets.empty:
        st.info("No bets logged yet.")
    else:
        bets["stake"] = bets["stake"].astype(float)
        bets["odds"] = bets["odds"].astype(float)
        bets["profit"] = bets.apply(bet_profit, axis=1)

        st.subheader("Settle a pending bet")
        pending = bets[bets["status"] == "Pending"]
        if pending.empty:
            st.caption("Nothing pending.")
        else:
            options = pending["id"].astype(str) + " — " + pending["matchup"] + " (" + pending["bet"] + ")"
            pick = st.selectbox("Pick a bet", options)
            pick_id = int(pick.split(" — ")[0])
            picked_row = pending[pending["id"] == pick_id].iloc[0]
            if picked_row["bet_type"] == "Acca" and picked_row["legs"]:
                with st.expander("Show legs"):
                    st.text(picked_row["legs"])
            c1, c2, c3 = st.columns(3)
            if c1.button("Mark Won"):
                bets = settle_bet(bets, pick_id, "Won")
                st.rerun()
            if c2.button("Mark Lost"):
                bets = settle_bet(bets, pick_id, "Lost")
                st.rerun()
            if c3.button("Mark Push"):
                bets = settle_bet(bets, pick_id, "Push")
                st.rerun()

        st.subheader("Summary")
        for label in TAGS:
            sub = bets[bets["tag"] == label]
            if sub.empty:
                continue
            settled = sub[sub["status"] != "Pending"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(label + " — staked", "${:.2f}".format(sub["stake"].sum()))
            c2.metric("Profit/loss", "${:+.2f}".format(settled["profit"].sum()))
            win_rate = (settled["status"] == "Won").mean() * 100 if not settled.empty else 0
            c3.metric("Win rate", "{:.0f}%".format(win_rate))
            c4.metric("Pending", int((sub["status"] == "Pending").sum()))

        st.subheader("All bets")
        view_cols = ["date_added", "game_date", "matchup", "bet_type", "bet", "tag",
                     "odds", "stake", "status", "profit"]
        st.dataframe(bets[view_cols].sort_values("date_added", ascending=False),
                     hide_index=True, use_container_width=True)

        acca_rows = bets[bets["bet_type"] == "Acca"]
        if not acca_rows.empty:
            with st.expander("Acca legs"):
                for _, r in acca_rows.iterrows():
                    st.write("**#{} — {} ({})**".format(int(r["id"]), r["matchup"], r["status"]))
                    st.text(r["legs"])

        st.download_button("Download log as CSV", bets.to_csv(index=False), "bets_log.csv", "text/csv")
