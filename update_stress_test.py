#!/usr/bin/env python3
"""Build the Action Network NFL Roster Stress Test dataset.

Run locally/Colab:
    pip install nflreadpy pandas numpy
    python update_stress_test.py --season 2026 --output-dir data

The script intentionally separates data ingestion, replacement mapping, scoring,
and export so that individual position models can be improved without changing
the frontend.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "nflreadpy is required. Install with: pip install nflreadpy pandas numpy"
    ) from exc


# -----------------------------
# Model configuration
# -----------------------------

POSITION_LEVERAGE = {
    "QB": 1.35,
    "LT": 1.18, "RT": 1.16, "OT": 1.16, "T": 1.16,
    "EDGE": 1.16, "DE": 1.13, "OLB": 1.08,
    "CB": 1.14, "NB": 1.12, "NCB": 1.12,
    "WR": 1.08, "S": 1.06, "FS": 1.06, "SS": 1.06,
    "LG": 1.05, "RG": 1.05, "G": 1.05, "C": 1.07, "OL": 1.07,
    "TE": 1.00, "LB": 1.00, "ILB": 1.00,
    "DT": 0.98, "NT": 0.96, "DL": 0.98,
    "RB": 0.90, "FB": 0.82,
    "K": 0.90, "P": 0.76, "LS": 0.62,
}

SPECIALISTS = {"K", "P", "LS"}
OFFENSE_HINTS = {
    "QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT", "OT", "T", "G", "OL"
}
DEFENSE_HINTS = {
    "DE", "DT", "NT", "DL", "EDGE", "LB", "ILB", "OLB", "CB", "NB", "NCB", "S", "FS", "SS", "DB"
}
UNAVAILABLE_STATUS_TOKENS = {
    "IR", "RES", "PUP", "NFI", "SUS", "OUT", "INJURED RESERVE", "RESERVE/INJURED"
}

# For transparent team score: five biggest individual vulnerability scores.
TEAM_TOP5_WEIGHTS = [0.45, 0.25, 0.15, 0.10, 0.05]


@dataclass
class BuildConfig:
    season: int
    recent_games: int = 3
    history_seasons: int = 3
    roles_offense: int = 11
    roles_defense: int = 11
    roles_special: int = 3
    output_dir: Path = Path("data")
    overrides_dir: Path | None = None


def to_pandas(frame: Any) -> pd.DataFrame:
    """Convert nflreadpy Polars output to pandas without assuming package internals."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def num(s: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def pct_to_unit(s: pd.Series) -> pd.Series:
    x = num(s)
    if not x.empty and x.quantile(0.95) > 1.5:
        x = x / 100.0
    return x.clip(0, 1.2)


def first_existing(df: pd.DataFrame, names: Iterable[str], default: Any = None) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def safe_col(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    return num(first_existing(df, names, 0.0))


def normalize_team_abbr(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).upper().strip()
    return {"JAC": "JAX", "LA": "LAR", "WSH": "WAS"}.get(s, s)


def broad_group(pos_grp: Any, pos: Any) -> str:
    grp = str(pos_grp or "").lower()
    p = str(pos or "").upper()
    if "off" in grp or p in OFFENSE_HINTS:
        return "Offense"
    if "def" in grp or p in DEFENSE_HINTS:
        return "Defense"
    if "special" in grp or p in SPECIALISTS:
        return "Special Teams"
    return "Other"


def position_family(pos: str) -> str:
    p = (pos or "").upper()
    if p in {"LT", "RT", "OT", "T"}: return "OT"
    if p in {"LG", "RG", "G"}: return "G"
    if p in {"C"}: return "C"
    if p in {"OL"}: return "OL"
    if p in {"DE", "EDGE"}: return "EDGE"
    if p in {"DT", "NT", "DL"}: return "IDL"
    if p in {"LB", "ILB", "OLB"}: return "LB"
    if p in {"CB", "NB", "NCB", "DB"}: return "CB"
    if p in {"S", "FS", "SS"}: return "S"
    if p in {"WR"}: return "WR"
    if p in {"TE"}: return "TE"
    if p in {"RB", "FB"}: return "RB"
    return p or "UNK"


def load_sources(cfg: BuildConfig) -> dict[str, pd.DataFrame]:
    seasons = list(range(cfg.season - cfg.history_seasons + 1, cfg.season + 1))
    print(f"Loading nflverse data for {seasons}...")
    sources = {
        "depth": to_pandas(nfl.load_depth_charts(cfg.season)),
        "snaps": to_pandas(nfl.load_snap_counts(seasons)),
        "stats": to_pandas(nfl.load_player_stats(seasons, summary_level="week")),
        "rosters_weekly": to_pandas(nfl.load_rosters_weekly(cfg.season)),
        "players": to_pandas(nfl.load_players()),
        "teams": to_pandas(nfl.load_teams()),
    }
    return sources


def latest_depth_chart(depth: pd.DataFrame) -> pd.DataFrame:
    d = depth.copy()
    if "team" not in d.columns and "club_code" in d.columns:
        d["team"] = d["club_code"]
    d["team"] = d["team"].map(normalize_team_abbr)

    # 2025+ has timestamped snapshots. Older schemas may have week.
    if "dt" in d.columns:
        d["_dt"] = pd.to_datetime(d["dt"], errors="coerce", utc=True)
        latest = d.groupby("team")["_dt"].transform("max")
        d = d[d["_dt"].eq(latest)].copy()
    elif "week" in d.columns:
        latest = num(d["week"]).groupby(d["team"]).transform("max")
        d = d[num(d["week"]).eq(latest)].copy()

    rename_map = {
        "full_name": "player_name",
        "position": "pos_abb",
        "depth_position": "pos_abb",
        "depth_team": "pos_rank",
        "formation": "pos_grp",
    }
    for old, new in rename_map.items():
        if new not in d.columns and old in d.columns:
            d[new] = d[old]

    if "pos_slot" not in d.columns:
        # Legacy fallback: each team + group + position becomes a slot.
        d["pos_slot"] = (
            d["team"].astype(str) + "|" + d.get("pos_grp", "").astype(str) + "|" + d["pos_abb"].astype(str)
        )
    d["pos_rank"] = pd.to_numeric(d.get("pos_rank"), errors="coerce").fillna(99).astype(int)
    d["pos_abb"] = d["pos_abb"].fillna("").astype(str).str.upper()
    d["group"] = [broad_group(g, p) for g, p in zip(d.get("pos_grp", ""), d["pos_abb"])]
    return d


def prepare_players(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    rename = {}
    if "display_name" in p.columns: rename["display_name"] = "player_name"
    p = p.rename(columns=rename)
    if "gsis_id" not in p.columns:
        p["gsis_id"] = ""
    p["gsis_id"] = p["gsis_id"].fillna("").astype(str)
    return p


def prepare_snap_metrics(snaps: pd.DataFrame, players: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    s = snaps.copy()
    s["team"] = s["team"].map(normalize_team_abbr)
    if "gsis_id" not in s.columns:
        # Snap counts key on PFR player ID. Map to GSIS.
        pmap = players[[c for c in ["pfr_id", "gsis_id"] if c in players.columns]].dropna().drop_duplicates()
        if "pfr_player_id" in s.columns and "pfr_id" in pmap.columns:
            s = s.merge(pmap, left_on="pfr_player_id", right_on="pfr_id", how="left")
        else:
            s["gsis_id"] = ""

    s["gsis_id"] = s["gsis_id"].fillna("").astype(str)
    s["week"] = pd.to_numeric(s.get("week"), errors="coerce").fillna(0)
    s["season"] = pd.to_numeric(s.get("season"), errors="coerce").fillna(0)
    s["off_pct"] = pct_to_unit(first_existing(s, ["offense_pct"], 0))
    s["def_pct"] = pct_to_unit(first_existing(s, ["defense_pct"], 0))
    s["st_pct"] = pct_to_unit(first_existing(s, ["st_pct"], 0))
    s["off_snaps"] = safe_col(s, ["offense_snaps"])
    s["def_snaps"] = safe_col(s, ["defense_snaps"])
    s["st_snaps"] = safe_col(s, ["st_snaps"])

    # Recent current-season usage.
    cur = s[s["season"].eq(cfg.season)].copy()
    cur = cur.sort_values(["team", "gsis_id", "week"], ascending=[True, True, False])
    recent = cur.groupby(["team", "gsis_id"], as_index=False).head(cfg.recent_games)
    rec = recent.groupby(["team", "gsis_id"], as_index=False).agg(
        recent_off_pct=("off_pct", "mean"),
        recent_def_pct=("def_pct", "mean"),
        recent_st_pct=("st_pct", "mean"),
        recent_games=("week", "nunique"),
    )

    # Career/recent-history snap volume with time decay.
    weights = {cfg.season - i: (0.58 ** i) for i in range(cfg.history_seasons)}
    s["season_weight"] = s["season"].map(weights).fillna(0)
    s["weighted_snaps"] = (s["off_snaps"] + s["def_snaps"] + s["st_snaps"]) * s["season_weight"]
    hist = s.groupby("gsis_id", as_index=False).agg(
        weighted_recent_snaps=("weighted_snaps", "sum"),
        career_sample_snaps=("off_snaps", "sum"),
        career_def_snaps=("def_snaps", "sum"),
        career_st_snaps=("st_snaps", "sum"),
    )
    hist["career_sample_snaps"] += hist["career_def_snaps"] + hist["career_st_snaps"]
    hist = hist.drop(columns=["career_def_snaps", "career_st_snaps"])
    return rec.merge(hist, on="gsis_id", how="outer")


def aggregate_stats(stats: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    """Create weighted recent player stat totals. Handles evolving nflverse columns gracefully."""
    st = stats.copy()
    st["season"] = pd.to_numeric(st.get("season"), errors="coerce").fillna(0)
    st["player_id"] = st.get("player_id", "").fillna("").astype(str)
    weights = {cfg.season - i: (0.58 ** i) for i in range(cfg.history_seasons)}
    st["w"] = st["season"].map(weights).fillna(0)

    wanted = [
        # offense
        "attempts", "passing_yards", "passing_tds", "passing_interceptions", "passing_epa",
        "carries", "rushing_yards", "rushing_tds", "rushing_epa",
        "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_epa",
        "receiving_first_downs", "rushing_first_downs",
        # defense
        "def_tackles", "def_tackles_solo", "def_tackles_for_loss", "def_sacks", "def_qb_hits",
        "def_interceptions", "def_pass_defended", "def_fumbles_forced", "def_tds",
        # possible kicking / punting fields across schemas
        "fg_att", "fg_made", "field_goal_attempts", "field_goals_made",
        "pat_att", "pat_made", "extra_point_attempts", "extra_points_made",
        "punts", "punting_yards", "punt_yards", "punts_inside_20", "punt_touchbacks",
    ]
    cols = [c for c in wanted if c in st.columns]
    for c in cols:
        st[c] = num(st[c]) * st["w"]
    if not cols:
        return pd.DataFrame({"gsis_id": st["player_id"].unique()})
    agg = st.groupby("player_id", as_index=False)[cols].sum().rename(columns={"player_id": "gsis_id"})
    return agg


def performance_raw(df: pd.DataFrame) -> pd.Series:
    """Position-sensitive public-data performance proxy.

    This is deliberately a transparent prototype model. OL and LS do not have
    robust public player-level quality data in nflverse, so they use a stability
    proxy and are flagged as such in the output.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)

    def C(*names: str) -> pd.Series:
        return safe_col(df, names)

    pos = df["position_family"]
    snaps = C("weighted_recent_snaps").clip(lower=1)

    # QB
    m = pos.eq("QB")
    att = C("attempts").clip(lower=1)
    q = (
        0.45 * (C("passing_epa") / att) +
        0.20 * (C("passing_yards") / att) / 8.0 +
        0.25 * (C("passing_tds") / att) * 20.0 -
        0.20 * (C("passing_interceptions") / att) * 20.0 +
        0.08 * (C("rushing_yards") / (C("carries").clip(lower=1))) / 6.0
    )
    out.loc[m] = q.loc[m]

    # RB
    m = pos.eq("RB")
    touches = (C("carries") + C("targets")).clip(lower=1)
    rb = (
        0.45 * C("rushing_yards") / C("carries").clip(lower=1) / 5.0 +
        0.25 * C("receiving_yards") / C("targets").clip(lower=1) / 9.0 +
        0.20 * (C("rushing_tds") + C("receiving_tds")) / touches * 20.0 +
        0.10 * (C("rushing_first_downs") + C("receiving_first_downs")) / touches
    )
    out.loc[m] = rb.loc[m]

    # WR/TE
    m = pos.isin(["WR", "TE"])
    targets = C("targets").clip(lower=1)
    rec = (
        0.50 * C("receiving_yards") / targets / 10.0 +
        0.20 * C("receptions") / targets +
        0.20 * C("receiving_tds") / targets * 15.0 +
        0.10 * C("receiving_first_downs") / targets
    )
    out.loc[m] = rec.loc[m]

    # Edge / interior DL
    m = pos.isin(["EDGE", "IDL"])
    dl = (
        4.0 * C("def_sacks") + 1.4 * C("def_qb_hits") + 1.6 * C("def_tackles_for_loss") +
        2.5 * C("def_fumbles_forced") + 0.12 * C("def_tackles")
    ) / snaps * 100.0
    out.loc[m] = dl.loc[m]

    # LB
    m = pos.eq("LB")
    lb = (
        0.20 * C("def_tackles") + 1.2 * C("def_tackles_for_loss") + 2.7 * C("def_sacks") +
        1.0 * C("def_qb_hits") + 4.0 * C("def_interceptions") + 1.1 * C("def_pass_defended") +
        2.5 * C("def_fumbles_forced")
    ) / snaps * 100.0
    out.loc[m] = lb.loc[m]

    # Secondary
    m = pos.isin(["CB", "S"])
    db = (
        4.8 * C("def_interceptions") + 1.6 * C("def_pass_defended") + 0.15 * C("def_tackles") +
        2.2 * C("def_sacks") + 2.2 * C("def_fumbles_forced")
    ) / snaps * 100.0
    out.loc[m] = db.loc[m]

    # K: use kicking if present, otherwise stability proxy below.
    m = pos.eq("K")
    fg_att = (C("fg_att", "field_goal_attempts")).clip(lower=0)
    fg_made = C("fg_made", "field_goals_made")
    pat_att = C("pat_att", "extra_point_attempts").clip(lower=0)
    pat_made = C("pat_made", "extra_points_made")
    have_k = (fg_att + pat_att) > 0
    kicker = 0.72 * (fg_made / fg_att.replace(0, np.nan)).fillna(0) + 0.28 * (pat_made / pat_att.replace(0, np.nan)).fillna(0)
    out.loc[m & have_k] = kicker.loc[m & have_k]

    # P: use punting if present.
    m = pos.eq("P")
    punts = C("punts").clip(lower=0)
    punt_yards = C("punting_yards", "punt_yards")
    inside = C("punts_inside_20")
    touchbacks = C("punt_touchbacks")
    have_p = punts > 0
    punter = 0.70 * (punt_yards / punts.replace(0, np.nan)).fillna(0) / 50.0 + 0.35 * inside / punts.replace(0, np.nan).fillna(1) - 0.20 * touchbacks / punts.replace(0, np.nan).fillna(1)
    out.loc[m & have_p] = punter.loc[m & have_p]

    # OL / C / G / OT and LS fallback: stability/experience proxy.
    fallback = (
        0.55 * np.log1p(C("weighted_recent_snaps")) / np.log(1500) +
        0.30 * C("recent_role_pct") +
        0.15 * np.log1p(C("career_sample_snaps")) / np.log(8000)
    )
    out = out.fillna(fallback)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def percentile_by_family(df: pd.DataFrame, raw_col: str = "performance_raw") -> pd.Series:
    def pct_rank(g: pd.Series) -> pd.Series:
        if len(g) == 1:
            return pd.Series([50.0], index=g.index)
        return g.rank(method="average", pct=True) * 100.0
    return df.groupby("position_family")[raw_col].transform(pct_rank).clip(0, 100)


def latest_roster_status(rosters: pd.DataFrame) -> pd.DataFrame:
    r = rosters.copy()
    r["team"] = r["team"].map(normalize_team_abbr)
    if "week" in r.columns:
        r["week"] = pd.to_numeric(r["week"], errors="coerce").fillna(0)
        latest = r.groupby("team")["week"].transform("max")
        r = r[r["week"].eq(latest)].copy()
    r["gsis_id"] = r.get("gsis_id", "").fillna("").astype(str)
    r["status"] = r.get("status", "").fillna("").astype(str)
    r["position"] = r.get("position", "").fillna("").astype(str).str.upper()
    r["position_family"] = r["position"].map(position_family)
    return r


def apply_injury_overrides(roster: pd.DataFrame, overrides_dir: Path | None) -> pd.DataFrame:
    r = roster.copy()
    r["manual_unavailable"] = False
    if not overrides_dir:
        return r
    path = overrides_dir / "injury_overrides.csv"
    if not path.exists():
        return r
    ov = pd.read_csv(path)
    ov["team"] = ov.get("team", "").map(normalize_team_abbr)
    ov["gsis_id"] = ov.get("gsis_id", "").fillna("").astype(str)
    ov["manual_unavailable"] = ov.get("unavailable", True).astype(bool)
    cols = [c for c in ["team", "gsis_id", "manual_unavailable"] if c in ov.columns]
    if {"team", "gsis_id"}.issubset(cols):
        r = r.merge(ov[cols].drop_duplicates(["team", "gsis_id"]), on=["team", "gsis_id"], how="left", suffixes=("", "_ov"))
        r["manual_unavailable"] = r.get("manual_unavailable_ov", r["manual_unavailable"]).fillna(r["manual_unavailable"]).astype(bool)
        r = r.drop(columns=[c for c in ["manual_unavailable_ov"] if c in r.columns])
    return r


def is_unavailable(status: str, manual: bool = False) -> bool:
    if manual:
        return True
    s = (status or "").upper()
    return any(tok in s for tok in UNAVAILABLE_STATUS_TOKENS)


def choose_roles(depth: pd.DataFrame, snap_metrics: pd.DataFrame, cfg: BuildConfig, unavailable_by_team: dict[str, set[str]] | None = None) -> pd.DataFrame:
    d = depth.copy()
    sm = snap_metrics[[c for c in ["team", "gsis_id", "recent_off_pct", "recent_def_pct", "recent_st_pct"] if c in snap_metrics.columns]].copy()
    d = d.merge(sm, on=["team", "gsis_id"], how="left")
    for c in ["recent_off_pct", "recent_def_pct", "recent_st_pct"]:
        if c not in d.columns: d[c] = 0.0
        d[c] = num(d[c])

    # Current first-choice row per team/slot. If the nominal depth-chart No. 1 is
    # currently unavailable, promote the next available player in that slot.
    unavailable_by_team = unavailable_by_team or {}
    d["_unavailable"] = [str(g) in unavailable_by_team.get(t, set()) for t, g in zip(d["team"], d["gsis_id"])]
    available = d[~d["_unavailable"]].copy()
    available = available.sort_values(["team", "pos_slot", "pos_rank", "recent_off_pct", "recent_def_pct", "recent_st_pct"], ascending=[True, True, True, False, False, False])
    starters = available.groupby(["team", "pos_slot"], as_index=False).head(1).copy()
    starters["usage"] = np.where(
        starters["group"].eq("Offense"), starters["recent_off_pct"],
        np.where(starters["group"].eq("Defense"), starters["recent_def_pct"], starters["recent_st_pct"])
    )

    selected = []
    for team, td in starters.groupby("team"):
        for group, n in [("Offense", cfg.roles_offense), ("Defense", cfg.roles_defense)]:
            g = td[td["group"].eq(group)].drop_duplicates("pos_slot")
            g = g.sort_values(["usage", "pos_slot"], ascending=[False, True]).head(n)
            selected.append(g)
        sp = td[td["group"].eq("Special Teams")].copy()
        # Prefer K/P/LS explicitly; otherwise highest-use special roles.
        picks = []
        for p in ["K", "P", "LS"]:
            cand = sp[sp["pos_abb"].eq(p)].sort_values("usage", ascending=False).head(1)
            if not cand.empty: picks.append(cand)
        if picks:
            sp_pick = pd.concat(picks, ignore_index=False).drop_duplicates("pos_slot").head(cfg.roles_special)
        else:
            sp_pick = sp.sort_values("usage", ascending=False).drop_duplicates("pos_slot").head(cfg.roles_special)
        selected.append(sp_pick)

    out = pd.concat(selected, ignore_index=True) if selected else starters.head(0)
    return out


def role_label_series(df: pd.DataFrame) -> pd.Series:
    labels = []
    counters: dict[tuple[str, str], int] = {}
    for _, row in df.sort_values(["team", "group", "pos_slot"]).iterrows():
        key = (row["team"], row["pos_abb"])
        counters[key] = counters.get(key, 0) + 1
        base = row["pos_abb"] or position_family(row["pos_abb"])
        # Number only repeated generic positions.
        if base in {"WR", "RB", "TE", "EDGE", "DE", "DT", "LB", "CB", "S", "OLB", "ILB"}:
            label = f"{base}{counters[key]}"
        else:
            label = base
        labels.append((row.name, label))
    return pd.Series({idx: label for idx, label in labels})


def merge_player_pool(
    depth: pd.DataFrame,
    snap_metrics: pd.DataFrame,
    stat_agg: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    # Build one row per depth-chart player, then calculate a comparable value.
    cols = ["team", "gsis_id", "player_name", "pos_abb", "pos_slot", "pos_rank", "group"]
    pool = depth[cols].drop_duplicates(["team", "pos_slot", "gsis_id", "pos_rank"]).copy()
    pool["position_family"] = pool["pos_abb"].map(position_family)
    pool = pool.merge(snap_metrics, on=["team", "gsis_id"], how="left")
    pool = pool.merge(stat_agg, on="gsis_id", how="left")

    pcols = [c for c in ["gsis_id", "years_exp", "headshot", "headshot_url", "pfr_id"] if c in players.columns]
    if pcols:
        pool = pool.merge(players[pcols].drop_duplicates("gsis_id"), on="gsis_id", how="left")

    for c in ["recent_off_pct", "recent_def_pct", "recent_st_pct", "weighted_recent_snaps", "career_sample_snaps"]:
        if c not in pool.columns: pool[c] = 0.0
        pool[c] = num(pool[c])
    pool["recent_role_pct"] = np.where(
        pool["group"].eq("Offense"), pool["recent_off_pct"],
        np.where(pool["group"].eq("Defense"), pool["recent_def_pct"], pool["recent_st_pct"])
    )
    pool["performance_raw"] = performance_raw(pool)

    # Backups often have small samples. Shrink thin samples toward the position-family
    # median rather than treating a handful of snaps like a full-season record.
    evidence_snaps = num(pool["weighted_recent_snaps"]).clip(lower=0)
    pool["sample_reliability"] = (1.0 - np.exp(-evidence_snaps / 325.0)).clip(0.0, 0.98)
    family_median = pool.groupby("position_family")["performance_raw"].transform("median").fillna(0.0)
    pool["performance_adjusted"] = (
        pool["sample_reliability"] * pool["performance_raw"] +
        (1.0 - pool["sample_reliability"]) * family_median
    )
    pool["player_value"] = percentile_by_family(pool, raw_col="performance_adjusted")

    proxy_families = {"OT", "G", "C", "OL", "LS"}
    pool["metric_basis"] = np.where(pool["position_family"].isin(proxy_families), "stability proxy", "public performance stats")
    return pool


def replacement_candidates(
    starter: pd.Series,
    pool: pd.DataFrame,
    unavailable_ids: set[str],
    limit: int = 2,
) -> list[tuple[pd.Series, str]]:
    """Return the most plausible next players, in order, plus mapping basis.

    Primary rule: next available player(s) in the same current depth-chart slot.
    Fallback: same position family, prioritizing recent role use and player value.
    The frontend exposes the basis so every projected replacement is auditable.
    """
    results: list[tuple[pd.Series, str]] = []
    used = {str(starter.get("gsis_id", ""))}

    same = pool[(pool["team"].eq(starter["team"])) & (pool["pos_slot"].eq(starter["pos_slot"]))].copy()
    same = same[~same["gsis_id"].isin(unavailable_ids)]
    same = same[same["pos_rank"].gt(starter["pos_rank"])].sort_values(
        ["pos_rank", "recent_role_pct", "player_value"], ascending=[True, False, False]
    )
    for _, row in same.iterrows():
        gid = str(row.get("gsis_id", ""))
        if gid in used:
            continue
        results.append((row, "next available player in the same depth-chart slot"))
        used.add(gid)
        if len(results) >= limit:
            return results

    fam = pool[(pool["team"].eq(starter["team"])) & (pool["position_family"].eq(starter["position_family"]))].copy()
    fam = fam[(~fam["gsis_id"].isin(unavailable_ids)) & (~fam["gsis_id"].astype(str).isin(used))]
    fam = fam.sort_values(["recent_role_pct", "player_value", "pos_rank"], ascending=[False, False, True])
    for _, row in fam.iterrows():
        gid = str(row.get("gsis_id", ""))
        if gid in used:
            continue
        results.append((row, "same-position-family fallback, prioritized by recent usage"))
        used.add(gid)
        if len(results) >= limit:
            break
    return results


def replacement_evidence(repl: pd.Series | None, basis: str) -> str:
    if repl is None:
        return "No clear active replacement found in the current depth-chart pool."
    rank = int(repl.get("pos_rank", 99) or 99)
    pct = float(repl.get("recent_role_pct", 0) or 0) * 100
    rank_text = f"No. {rank} on the latest depth chart" if rank < 90 else "Depth-chart fallback"
    return f"{rank_text}; {pct:.0f}% recent unit snap share; {basis}."


def confidence_label(repl: pd.Series | None) -> str:
    if repl is None:
        return "Low"
    pct = float(repl.get("recent_role_pct", 0) or 0)
    snaps = float(repl.get("career_sample_snaps", 0) or 0)
    rank = int(repl.get("pos_rank", 99) or 99)
    if rank == 2 and pct >= 0.15:
        return "High"
    if rank <= 2 and (pct > 0 or snaps >= 100):
        return "Medium"
    if snaps >= 400:
        return "Medium"
    return "Low"


def unit_stress_count(team: str, family: str, roster: pd.DataFrame) -> int:
    rr = roster[(roster["team"].eq(team)) & (roster["position_family"].eq(family))]
    return int(sum(is_unavailable(st, bool(man)) for st, man in zip(rr["status"], rr["manual_unavailable"])))


def apply_value_overrides(pool: pd.DataFrame, overrides_dir: Path | None) -> pd.DataFrame:
    if not overrides_dir:
        return pool
    path = overrides_dir / "player_value_overrides.csv"
    if not path.exists():
        return pool
    ov = pd.read_csv(path)
    if "gsis_id" not in ov.columns or "player_value" not in ov.columns:
        return pool
    ov["gsis_id"] = ov["gsis_id"].astype(str)
    ov["player_value"] = pd.to_numeric(ov["player_value"], errors="coerce")
    m = pool.merge(ov[["gsis_id", "player_value"]].rename(columns={"player_value": "override_value"}), on="gsis_id", how="left")
    mask = m["override_value"].notna()
    m.loc[mask, "player_value"] = m.loc[mask, "override_value"].clip(0, 100)
    m.loc[mask, "metric_basis"] = "manual validated override"
    return m.drop(columns="override_value")


def team_metadata(teams: pd.DataFrame) -> dict[str, dict[str, str]]:
    t = teams.copy()
    abbr_col = next((c for c in ["team_abbr", "team", "abbr"] if c in t.columns), None)
    if not abbr_col:
        return {}
    out = {}
    for _, r in t.iterrows():
        ab = normalize_team_abbr(r.get(abbr_col, ""))
        if not ab:
            continue
        out[ab] = {
            "name": str(r.get("team_name", r.get("team_nick", ab))),
            "logo": str(r.get("team_logo_espn", r.get("team_logo_wikipedia", "")) or ""),
            "color": str(r.get("team_color", "") or ""),
            "color2": str(r.get("team_color2", "") or ""),
        }
    return out


def build_records(sources: dict[str, pd.DataFrame], cfg: BuildConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    players = prepare_players(sources["players"])
    depth = latest_depth_chart(sources["depth"])
    snaps = prepare_snap_metrics(sources["snaps"], players, cfg)
    stats = aggregate_stats(sources["stats"], cfg)
    roster = apply_injury_overrides(latest_roster_status(sources["rosters_weekly"]), cfg.overrides_dir)
    pool = merge_player_pool(depth, snaps, stats, players)
    pool = apply_value_overrides(pool, cfg.overrides_dir)

    # Unavailable set from roster statuses and optional override file.
    unavailable_by_team: dict[str, set[str]] = {}
    for team, rr in roster.groupby("team"):
        unavailable_by_team[team] = {
            str(gsis) for gsis, st, man in zip(rr["gsis_id"], rr["status"], rr["manual_unavailable"])
            if gsis and is_unavailable(st, bool(man))
        }

    selected = choose_roles(depth, snaps, cfg, unavailable_by_team)
    selected["role_label"] = role_label_series(selected)

    records = []
    validations = []
    for _, strow in selected.iterrows():
        team = strow["team"]
        st_match = pool[(pool["team"].eq(team)) & (pool["gsis_id"].eq(str(strow.get("gsis_id", "")))) & (pool["pos_slot"].eq(strow["pos_slot"]))]
        if st_match.empty:
            continue
        starter = st_match.sort_values("pos_rank").iloc[0]
        candidates = replacement_candidates(starter, pool, unavailable_by_team.get(team, set()), limit=2)
        repl, repl_basis = candidates[0] if candidates else (None, "unresolved")
        next_repl, next_basis = candidates[1] if len(candidates) > 1 else (None, "unresolved")

        starter_value = float(starter.get("player_value", 50) or 50)
        replacement_value = float(repl.get("player_value", 35) or 35) if repl is not None else 25.0
        next_replacement_value = float(next_repl.get("player_value", 25) or 25) if next_repl is not None else 20.0
        depth_cliff = max(0.0, starter_value - replacement_value)
        next_depth_cliff = max(0.0, replacement_value - next_replacement_value)
        dependency = float(starter.get("recent_role_pct", 0) or 0)
        # Early season / missing snap fallback: first-choice starters still get a floor.
        dependency = min(1.0, max(dependency, 0.55 if starter["group"] != "Special Teams" else 0.35))
        leverage = float(POSITION_LEVERAGE.get(starter["pos_abb"], POSITION_LEVERAGE.get(starter["position_family"], 1.0)))
        depleted = unit_stress_count(team, starter["position_family"], roster)
        stress_multiplier = min(1.32, 1.0 + 0.08 * depleted)
        raw_impact = depth_cliff * dependency * leverage * stress_multiplier
        # After the starter is simulated out, the first replacement becomes the new first choice.
        # The next cliff therefore measures replacement 1 -> replacement 2 (or an unresolved fallback).
        after_loss_stress_multiplier = min(1.40, stress_multiplier + 0.08)
        raw_after_loss_impact = next_depth_cliff * dependency * leverage * after_loss_stress_multiplier
        conf = confidence_label(repl)
        next_conf = confidence_label(next_repl)

        record = {
            "team": team,
            "group": starter["group"],
            "role": str(strow.get("role_label", starter["pos_abb"])),
            "position": starter["pos_abb"],
            "position_family": starter["position_family"],
            "slot": str(starter["pos_slot"]),
            "starter": {
                "gsis_id": starter["gsis_id"],
                "name": starter["player_name"],
                "value": round(starter_value, 1),
                "recent_snap_share": round(dependency * 100, 1),
                "sample_reliability": round(float(starter.get("sample_reliability", 0) or 0) * 100, 1),
                "metric_basis": starter.get("metric_basis", "public performance stats"),
                "headshot": str(starter.get("headshot_url", starter.get("headshot", "")) or ""),
            },
            "replacement": {
                "gsis_id": "" if repl is None else repl["gsis_id"],
                "name": "No clear replacement" if repl is None else repl["player_name"],
                "value": round(replacement_value, 1),
                "depth_rank": None if repl is None else int(repl.get("pos_rank", 99)),
                "recent_snap_share": 0.0 if repl is None else round(float(repl.get("recent_role_pct", 0) or 0) * 100, 1),
                "sample_reliability": 0.0 if repl is None else round(float(repl.get("sample_reliability", 0) or 0) * 100, 1),
                "confidence": conf,
                "selection_basis": repl_basis,
                "evidence": replacement_evidence(repl, repl_basis),
                "metric_basis": "unresolved" if repl is None else repl.get("metric_basis", "public performance stats"),
            },
            "next_replacement": {
                "gsis_id": "" if next_repl is None else next_repl["gsis_id"],
                "name": "No clear second replacement" if next_repl is None else next_repl["player_name"],
                "value": round(next_replacement_value, 1),
                "depth_rank": None if next_repl is None else int(next_repl.get("pos_rank", 99)),
                "recent_snap_share": 0.0 if next_repl is None else round(float(next_repl.get("recent_role_pct", 0) or 0) * 100, 1),
                "sample_reliability": 0.0 if next_repl is None else round(float(next_repl.get("sample_reliability", 0) or 0) * 100, 1),
                "confidence": next_conf,
                "selection_basis": next_basis,
                "evidence": replacement_evidence(next_repl, next_basis),
                "metric_basis": "unresolved" if next_repl is None else next_repl.get("metric_basis", "public performance stats"),
            },
            "depth_cliff": round(depth_cliff, 1),
            "next_depth_cliff": round(next_depth_cliff, 1),
            "dependency": round(dependency * 100, 1),
            "position_leverage": round(leverage, 2),
            "unit_unavailable_count": depleted,
            "unit_stress_multiplier": round(stress_multiplier, 2),
            "raw_impact": raw_impact,
            "raw_after_loss_impact": raw_after_loss_impact,
        }
        records.append(record)

        flags = []
        if repl is None: flags.append("missing replacement")
        if conf == "Low": flags.append("low replacement confidence")
        if starter.get("metric_basis") == "stability proxy": flags.append("starter uses proxy metric")
        if repl is not None and repl.get("metric_basis") == "stability proxy": flags.append("replacement uses proxy metric")
        validations.append({
            "team": team,
            "role": record["role"],
            "starter": record["starter"]["name"],
            "replacement": record["replacement"]["name"],
            "confidence": conf,
            "replacement_basis": repl_basis,
            "replacement_evidence": replacement_evidence(repl, repl_basis),
            "second_replacement": "No clear second replacement" if next_repl is None else next_repl["player_name"],
            "flags": "; ".join(flags),
        })

    # Convert raw individual impact into a league-relative 0-100 percentile index.
    # Scenario scores are mapped against the same baseline distribution so "before" and
    # "after" remain comparable in the frontend.
    if records:
        vals = np.asarray([float(r["raw_impact"]) for r in records], dtype=float)
        sorted_vals = np.sort(vals)

        def pct_against_baseline(x: float) -> float:
            if len(sorted_vals) == 0:
                return 0.0
            return 100.0 * float(np.searchsorted(sorted_vals, x, side="right")) / len(sorted_vals)

        for r in records:
            r["impact_score"] = round(pct_against_baseline(float(r["raw_impact"])), 1)
            r["after_loss_impact_score"] = round(pct_against_baseline(float(r["raw_after_loss_impact"])), 1)
            r.pop("raw_impact", None)
            r.pop("raw_after_loss_impact", None)

    return records, pd.DataFrame(validations)


def build_payload(records: list[dict[str, Any]], teams_df: pd.DataFrame, cfg: BuildConfig) -> dict[str, Any]:
    meta = team_metadata(teams_df)
    by_team: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_team.setdefault(r["team"], []).append(r)

    teams = []
    for abbr, roles in sorted(by_team.items()):
        impacts = sorted([float(x["impact_score"]) for x in roles], reverse=True)
        top5 = impacts[:5] + [0] * max(0, 5 - len(impacts))
        vulnerability = sum(w * s for w, s in zip(TEAM_TOP5_WEIGHTS, top5))
        group_scores = {}
        for group in ["Offense", "Defense", "Special Teams"]:
            xs = sorted([float(x["impact_score"]) for x in roles if x["group"] == group], reverse=True)
            if not xs:
                group_scores[group] = 0.0
            else:
                group_scores[group] = round(float(np.mean(xs[: min(3, len(xs))])), 1)
        worst = max(roles, key=lambda x: x["impact_score"]) if roles else None
        major_cliffs = int(sum(float(x["impact_score"]) >= 80 for x in roles))
        group_ranking = sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)
        weakest_unit = group_ranking[0][0] if group_ranking else None
        md = meta.get(abbr, {})
        teams.append({
            "abbr": abbr,
            "name": md.get("name", abbr),
            "logo": md.get("logo", ""),
            "color": md.get("color", ""),
            "color2": md.get("color2", ""),
            "vulnerability_score": round(vulnerability, 1),
            "group_scores": group_scores,
            "most_vulnerable_role": None if not worst else worst["role"],
            "most_vulnerable_player": None if not worst else worst["starter"]["name"],
            "most_vulnerable_impact": 0 if not worst else worst["impact_score"],
            "major_depth_cliffs": major_cliffs,
            "weakest_unit": weakest_unit,
            "roles": sorted(roles, key=lambda x: (x["group"], -x["impact_score"])),
        })

    teams = sorted(teams, key=lambda x: x["vulnerability_score"], reverse=True)
    for i, t in enumerate(teams, 1):
        t["rank"] = i

    payload = {
        "metadata": {
            "title": "NFL Backup Depth Stress Test",
            "season": cfg.season,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": "0.2-depth-cascade",
            "team_score_definition": "Depth Risk Index: weighted average of the five largest current starter-to-next-up impact scores (45%, 25%, 15%, 10%, 5%). Higher means thinner backup depth, not a higher chance of injury.",
            "individual_score_definition": "Depth Cliff Impact: league-relative percentile of starter-to-replacement value gap × recent role share × position leverage × current unit depletion multiplier. Small performance samples are shrunk toward the position-family median before player values are ranked.",
            "replacement_definition": "Projected next-up players come from the latest available depth-chart order, filtered for availability and supported by recent unit snap share. Same-position-family usage is the fallback when the listed slot has no clear active successor.",
            "sources": [
                "nflverse/nflreadpy depth charts",
                "nflverse PFR snap counts",
                "nflverse player statistics",
                "nflverse weekly rosters",
            ],
            "important_limitations": [
                "This is a backup-depth stress test, not an injury probability, game-outcome model or playoff forecast.",
                "Public player-level OL and long-snapper quality data are limited; those roles use a labeled stability/experience proxy unless overridden.",
                "nflverse's dedicated injury-report feed is unavailable after 2024, so current short-term injury designations should be supplied through injury_overrides.csv or another validated feed.",
                "Replacement mapping should be manually reviewed before publication, especially for offensive line position shuffles and rotational defensive roles.",
            ],
        },
        "teams": teams,
    }
    return payload


def validation_summary(validation: pd.DataFrame, payload: dict[str, Any], cfg: BuildConfig) -> pd.DataFrame:
    team_rows = []
    for t in payload["teams"]:
        v = validation[validation["team"].eq(t["abbr"])] if not validation.empty else validation
        team_rows.append({
            "team": t["abbr"],
            "roles_found": len(t["roles"]),
            "expected_roles": cfg.roles_offense + cfg.roles_defense + cfg.roles_special,
            "missing_replacements": int(v["flags"].str.contains("missing replacement", na=False).sum()) if not v.empty else 0,
            "low_confidence_replacements": int(v["confidence"].eq("Low").sum()) if not v.empty else 0,
            "proxy_metric_roles": int(v["flags"].str.contains("proxy metric", na=False).sum()) if not v.empty else 0,
            "publication_check": "REVIEW" if (len(t["roles"]) != 25 or (not v.empty and (v["flags"] != "").any())) else "PASS",
        })
    return pd.DataFrame(team_rows)


def write_outputs(payload: dict[str, Any], validation: pd.DataFrame, cfg: BuildConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = cfg.output_dir / "nfl_roster_stress_test.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    validation.to_csv(cfg.output_dir / "validation_player_level.csv", index=False)
    validation_summary(validation, payload, cfg).to_csv(cfg.output_dir / "validation_team_summary.csv", index=False)
    print(f"Wrote {out_json}")
    print(f"Teams: {len(payload['teams'])}; roles: {sum(len(t['roles']) for t in payload['teams'])}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=datetime.now().year)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--overrides-dir", default="overrides")
    p.add_argument("--recent-games", type=int, default=3)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = BuildConfig(
        season=a.season,
        recent_games=a.recent_games,
        output_dir=Path(a.output_dir),
        overrides_dir=Path(a.overrides_dir) if a.overrides_dir else None,
    )
    sources = load_sources(cfg)
    records, validation = build_records(sources, cfg)
    payload = build_payload(records, sources["teams"], cfg)
    write_outputs(payload, validation, cfg)


if __name__ == "__main__":
    main()
