"""
Feature extraction from OE match data for the v2 prediction model.

Computes rolling team-level features (recent form, gold efficiency,
kill/death stats, objective control) from the raw match CSVs.

These features are computed incrementally — for each match, only data
available BEFORE that match is used (no lookahead).
"""

import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

DB_PATH = _ROOT / "db" / "lol_model.db"
RAW_DIR = _ROOT / "data" / "raw" / "oracleselixir"

WINDOW = 15  # rolling window size (last N games)
MIN_GAMES = 3  # minimum games needed to produce features


class TeamStats:
    """Rolling window of a team's recent game stats."""

    def __init__(self, window: int = WINDOW) -> None:
        self.window = window
        self.results: deque = deque(maxlen=window)
        self.kills: deque = deque(maxlen=window)
        self.deaths: deque = deque(maxlen=window)
        self.gamelengths: deque = deque(maxlen=window)
        self.gd15: deque = deque(maxlen=window)
        self.gd10: deque = deque(maxlen=window)
        self.first_bloods: deque = deque(maxlen=window)
        self.first_towers: deque = deque(maxlen=window)
        self.total_games = 0

    def add_game(
        self,
        result: int,
        kills: float,
        deaths: float,
        gamelength: float,
        gd10: float,
        gd15: float,
        first_blood: float,
        first_tower: float,
    ) -> None:
        self.results.append(result)
        self.kills.append(kills)
        self.deaths.append(deaths)
        self.gamelengths.append(gamelength)
        self.gd10.append(gd10)
        self.gd15.append(gd15)
        self.first_bloods.append(first_blood)
        self.first_towers.append(first_tower)
        self.total_games += 1

    def has_enough(self) -> bool:
        return len(self.results) >= MIN_GAMES

    def features(self) -> Dict[str, float]:
        """Return current feature dict (call BEFORE adding the current game)."""
        if not self.has_enough():
            return {}
        n = len(self.results)
        r = list(self.results)
        return {
            "win_rate": sum(r) / n,
            "win_rate_last5": sum(r[-5:]) / min(n, 5),
            "streak": self._streak(),
            "avg_kills": sum(self.kills) / n,
            "avg_deaths": sum(self.deaths) / n,
            "kda": (sum(self.kills)) / max(sum(self.deaths), 1),
            "avg_gamelength": sum(self.gamelengths) / n,
            "avg_gd10": sum(self.gd10) / n,
            "avg_gd15": sum(self.gd15) / n,
            "fb_rate": sum(self.first_bloods) / n,
            "ft_rate": sum(self.first_towers) / n,
            "games_played": self.total_games,
        }

    def _streak(self) -> int:
        """Current win/loss streak. Positive = wins, negative = losses."""
        if not self.results:
            return 0
        streak = 0
        last = self.results[-1]
        for r in reversed(self.results):
            if r == last:
                streak += 1
            else:
                break
        return streak if last == 1 else -streak


def load_team_game_data() -> pd.DataFrame:
    """
    Load team-summary rows from OE CSVs with the columns needed for features.
    Returns one row per team per game, sorted by date.
    """
    from scrapers.oe_scraper import T2_LEAGUES, YEARS

    frames = []
    for year in YEARS:
        path = RAW_DIR / f"{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df = df[df["league"].isin(T2_LEAGUES)]
        team_rows = df[df["position"] == "team"].copy()
        frames.append(team_rows)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    return df


def build_feature_dataset() -> pd.DataFrame:
    """
    Walk through all T2 matches chronologically. For each match, compute
    features from each team's recent history (BEFORE the match), then
    record the match outcome.

    Returns a DataFrame with one row per match:
        blue_* features, red_* features, elo_diff, result
    """
    logger.info("Loading team game data from OE CSVs…")
    df = load_team_game_data()
    if df.empty:
        logger.error("No team game data found")
        return pd.DataFrame()

    logger.info(f"  {len(df)} team-game rows loaded")

    # Group by gameid to get blue/red side pairs
    trackers: Dict[str, TeamStats] = defaultdict(TeamStats)
    records = []

    games_processed = 0
    for gameid, grp in df.groupby("gameid", sort=False):
        blue = grp[grp["side"] == "Blue"]
        red = grp[grp["side"] == "Red"]
        if blue.empty or red.empty:
            continue

        b = blue.iloc[0]
        r = red.iloc[0]
        blue_team = str(b["teamname"])
        red_team = str(r["teamname"])
        date = str(b["date"])
        league = str(b["league"])

        bt = trackers[blue_team]
        rt = trackers[red_team]

        # Extract features BEFORE this game
        if bt.has_enough() and rt.has_enough():
            bf = bt.features()
            rf = rt.features()
            record = {"gameid": gameid, "date": date, "league": league}
            for k, v in bf.items():
                record[f"blue_{k}"] = v
            for k, v in rf.items():
                record[f"red_{k}"] = v
            # Differentials
            record["wr_diff"] = bf["win_rate"] - rf["win_rate"]
            record["kda_diff"] = bf["kda"] - rf["kda"]
            record["gd15_diff"] = bf["avg_gd15"] - rf["avg_gd15"]
            record["gd10_diff"] = bf["avg_gd10"] - rf["avg_gd10"]
            record["streak_diff"] = bf["streak"] - rf["streak"]
            record["fb_diff"] = bf["fb_rate"] - rf["fb_rate"]
            record["result"] = int(b.get("result", 0))
            records.append(record)

        # Update trackers with this game's results
        def _safe(val, default=0.0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        b_result = int(b.get("result", 0))
        bt.add_game(
            result=b_result,
            kills=_safe(b.get("teamkills")),
            deaths=_safe(b.get("teamdeaths")),
            gamelength=_safe(b.get("gamelength")),
            gd10=_safe(b.get("golddiffat10")),
            gd15=_safe(b.get("golddiffat15")),
            first_blood=_safe(b.get("firstblood")),
            first_tower=_safe(b.get("firsttower")),
        )
        rt.add_game(
            result=1 - b_result,
            kills=_safe(r.get("teamkills")),
            deaths=_safe(r.get("teamdeaths")),
            gamelength=_safe(r.get("gamelength")),
            gd10=_safe(r.get("golddiffat10")),
            gd15=_safe(r.get("golddiffat15")),
            first_blood=_safe(r.get("firstblood")),
            first_tower=_safe(r.get("firsttower")),
        )
        games_processed += 1

    result_df = pd.DataFrame(records)
    logger.info(f"  {len(result_df)} feature rows built from {games_processed} games")
    return result_df


if __name__ == "__main__":
    df = build_feature_dataset()
    if not df.empty:
        print(f"\nFeature dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Date range: {df['date'].min()} → {df['date'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample (first row):")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")
