"""
V3 feature extraction from OE match data.

Computes ~50 rolling team-level features per side from 165 OE columns:
- Original: win rate, KDA, gold diff, first blood/tower (12 features)
- NEW: objective control, laning progression, economy, vision, tempo (27 features)

All features computed incrementally — only data BEFORE each match is used.
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

WINDOW = 15
MIN_GAMES = 3


class TeamStats:
    """Rolling window of a team's recent game stats (~39 features)."""

    def __init__(self, window: int = WINDOW) -> None:
        self.window = window
        self.total_games = 0

        # Original
        self.results: deque = deque(maxlen=window)
        self.kills: deque = deque(maxlen=window)
        self.deaths: deque = deque(maxlen=window)
        self.gamelengths: deque = deque(maxlen=window)
        self.gd10: deque = deque(maxlen=window)
        self.gd15: deque = deque(maxlen=window)
        self.first_bloods: deque = deque(maxlen=window)
        self.first_towers: deque = deque(maxlen=window)

        # Laning progression
        self.gd20: deque = deque(maxlen=window)
        self.xpdiff10: deque = deque(maxlen=window)
        self.xpdiff15: deque = deque(maxlen=window)
        self.csdiff10: deque = deque(maxlen=window)

        # Objective control
        self.dragons: deque = deque(maxlen=window)
        self.opp_dragons: deque = deque(maxlen=window)
        self.barons: deque = deque(maxlen=window)
        self.heralds: deque = deque(maxlen=window)
        self.void_grubs: deque = deque(maxlen=window)
        self.first_dragons: deque = deque(maxlen=window)
        self.first_barons: deque = deque(maxlen=window)
        self.first_heralds: deque = deque(maxlen=window)

        # Economy & efficiency
        self.dpm: deque = deque(maxlen=window)
        self.cspm: deque = deque(maxlen=window)
        self.earned_gold: deque = deque(maxlen=window)
        self.towers: deque = deque(maxlen=window)
        self.opp_towers: deque = deque(maxlen=window)
        self.turret_plates: deque = deque(maxlen=window)
        self.inhibitors: deque = deque(maxlen=window)

        # Vision
        self.vision_scores: deque = deque(maxlen=window)
        self.wards_placed: deque = deque(maxlen=window)
        self.wards_killed: deque = deque(maxlen=window)

        # Tempo
        self.team_kpm: deque = deque(maxlen=window)
        self.killsat15: deque = deque(maxlen=window)

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
        gd20: float = 0.0,
        xpdiff10: float = 0.0,
        xpdiff15: float = 0.0,
        csdiff10: float = 0.0,
        dragons: float = 0.0,
        opp_dragons: float = 0.0,
        barons: float = 0.0,
        heralds: float = 0.0,
        void_grubs: float = 0.0,
        first_dragon: float = 0.0,
        first_baron: float = 0.0,
        first_herald: float = 0.0,
        dpm: float = 0.0,
        cspm: float = 0.0,
        earned_gold: float = 0.0,
        towers: float = 0.0,
        opp_towers: float = 0.0,
        turret_plates: float = 0.0,
        inhibitors: float = 0.0,
        vision_score: float = 0.0,
        wards_placed_val: float = 0.0,
        wards_killed_val: float = 0.0,
        team_kpm_val: float = 0.0,
        killsat15_val: float = 0.0,
    ) -> None:
        # Original
        self.results.append(result)
        self.kills.append(kills)
        self.deaths.append(deaths)
        self.gamelengths.append(gamelength)
        self.gd10.append(gd10)
        self.gd15.append(gd15)
        self.first_bloods.append(first_blood)
        self.first_towers.append(first_tower)

        # Laning
        self.gd20.append(gd20)
        self.xpdiff10.append(xpdiff10)
        self.xpdiff15.append(xpdiff15)
        self.csdiff10.append(csdiff10)

        # Objectives
        self.dragons.append(dragons)
        self.opp_dragons.append(opp_dragons)
        self.barons.append(barons)
        self.heralds.append(heralds)
        self.void_grubs.append(void_grubs)
        self.first_dragons.append(first_dragon)
        self.first_barons.append(first_baron)
        self.first_heralds.append(first_herald)

        # Economy
        self.dpm.append(dpm)
        self.cspm.append(cspm)
        self.earned_gold.append(earned_gold)
        self.towers.append(towers)
        self.opp_towers.append(opp_towers)
        self.turret_plates.append(turret_plates)
        self.inhibitors.append(inhibitors)

        # Vision
        self.vision_scores.append(vision_score)
        self.wards_placed.append(wards_placed_val)
        self.wards_killed.append(wards_killed_val)

        # Tempo
        self.team_kpm.append(team_kpm_val)
        self.killsat15.append(killsat15_val)

        self.total_games += 1

    def has_enough(self) -> bool:
        return len(self.results) >= MIN_GAMES

    def _avg(self, d: deque) -> float:
        return sum(d) / len(d) if d else 0.0

    def _streak(self) -> int:
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

    def features(self) -> Dict[str, float]:
        """Return ~39 features. Call BEFORE adding the current game."""
        if not self.has_enough():
            return {}
        n = len(self.results)
        r = list(self.results)

        total_drags = sum(self.dragons) + sum(self.opp_dragons)
        gl_list = list(self.gamelengths)
        avg_gl = sum(gl_list) / n if n else 1
        total_kills = sum(self.kills)

        return {
            # --- Original (12) ---
            "win_rate": sum(r) / n,
            "win_rate_last5": sum(r[-5:]) / min(n, 5),
            "streak": self._streak(),
            "avg_kills": sum(self.kills) / n,
            "avg_deaths": sum(self.deaths) / n,
            "kda": total_kills / max(sum(self.deaths), 1),
            "avg_gamelength": avg_gl,
            "avg_gd10": self._avg(self.gd10),
            "avg_gd15": self._avg(self.gd15),
            "fb_rate": sum(self.first_bloods) / n,
            "ft_rate": sum(self.first_towers) / n,
            "games_played": self.total_games,

            # --- Laning progression (6) ---
            "avg_gd20": self._avg(self.gd20),
            "avg_xpdiff10": self._avg(self.xpdiff10),
            "avg_xpdiff15": self._avg(self.xpdiff15),
            "avg_csdiff10": self._avg(self.csdiff10),
            "gd_slope_10_15": self._avg(self.gd15) - self._avg(self.gd10),
            "gd_slope_15_20": self._avg(self.gd20) - self._avg(self.gd15),

            # --- Objective control (8) ---
            "avg_dragons": self._avg(self.dragons),
            "avg_opp_dragons": self._avg(self.opp_dragons),
            "dragon_control": sum(self.dragons) / max(total_drags, 1),
            "baron_rate": self._avg(self.barons),
            "first_dragon_rate": sum(self.first_dragons) / n,
            "first_baron_rate": sum(self.first_barons) / n,
            "first_herald_rate": sum(self.first_heralds) / n,
            "void_grub_rate": self._avg(self.void_grubs),

            # --- Economy & efficiency (6) ---
            "avg_dpm": self._avg(self.dpm),
            "avg_cspm": self._avg(self.cspm),
            "avg_earned_gold": self._avg(self.earned_gold),
            "avg_towers": self._avg(self.towers),
            "avg_turret_plates": self._avg(self.turret_plates),
            "tower_diff": self._avg(self.towers) - self._avg(self.opp_towers),

            # --- Vision (4) ---
            "avg_vision_score": self._avg(self.vision_scores),
            "avg_wards_placed": self._avg(self.wards_placed),
            "avg_wards_killed": self._avg(self.wards_killed),
            "vision_dominance": sum(self.vision_scores) / max(sum(self.vision_scores) + n * 50, 1),

            # --- Tempo (3) ---
            "avg_team_kpm": self._avg(self.team_kpm),
            "early_kill_share": sum(self.killsat15) / max(total_kills, 1),
            "avg_towers_per_min": sum(self.towers) / max(sum(gl_list) / 60, 1),
        }


def load_team_game_data() -> pd.DataFrame:
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


def _safe(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_feature_dataset() -> pd.DataFrame:
    """
    Walk through all T2 matches chronologically, compute V3 features
    from each team's recent history BEFORE the match.
    """
    logger.info("Loading team game data from OE CSVs…")
    df = load_team_game_data()
    if df.empty:
        logger.error("No team game data found")
        return pd.DataFrame()

    logger.info(f"  {len(df)} team-game rows loaded")

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

            # Differentials (original)
            record["wr_diff"] = bf["win_rate"] - rf["win_rate"]
            record["kda_diff"] = bf["kda"] - rf["kda"]
            record["gd15_diff"] = bf["avg_gd15"] - rf["avg_gd15"]
            record["gd10_diff"] = bf["avg_gd10"] - rf["avg_gd10"]
            record["streak_diff"] = bf["streak"] - rf["streak"]
            record["fb_diff"] = bf["fb_rate"] - rf["fb_rate"]

            # New differentials
            record["gd20_diff"] = bf["avg_gd20"] - rf["avg_gd20"]
            record["xpdiff10_diff"] = bf["avg_xpdiff10"] - rf["avg_xpdiff10"]
            record["dragon_ctrl_diff"] = bf["dragon_control"] - rf["dragon_control"]
            record["baron_rate_diff"] = bf["baron_rate"] - rf["baron_rate"]
            record["first_dragon_diff"] = bf["first_dragon_rate"] - rf["first_dragon_rate"]
            record["dpm_diff"] = bf["avg_dpm"] - rf["avg_dpm"]
            record["cspm_diff"] = bf["avg_cspm"] - rf["avg_cspm"]
            record["tower_ctrl_diff"] = bf["tower_diff"] - rf["tower_diff"]
            record["vision_diff"] = bf["avg_vision_score"] - rf["avg_vision_score"]
            record["tempo_diff"] = bf["avg_team_kpm"] - rf["avg_team_kpm"]
            record["gd_slope_diff"] = bf["gd_slope_10_15"] - rf["gd_slope_10_15"]
            record["plate_diff"] = bf["avg_turret_plates"] - rf["avg_turret_plates"]

            record["result"] = int(b.get("result", 0))
            records.append(record)

        # Update trackers with this game's results
        b_result = int(b.get("result", 0))
        for side_row, tracker, res in [(b, bt, b_result), (r, rt, 1 - b_result)]:
            tracker.add_game(
                result=res,
                kills=_safe(side_row.get("teamkills")),
                deaths=_safe(side_row.get("teamdeaths")),
                gamelength=_safe(side_row.get("gamelength")),
                gd10=_safe(side_row.get("golddiffat10")),
                gd15=_safe(side_row.get("golddiffat15")),
                first_blood=_safe(side_row.get("firstblood")),
                first_tower=_safe(side_row.get("firsttower")),
                gd20=_safe(side_row.get("golddiffat20")),
                xpdiff10=_safe(side_row.get("xpdiffat10")),
                xpdiff15=_safe(side_row.get("xpdiffat15")),
                csdiff10=_safe(side_row.get("csdiffat10")),
                dragons=_safe(side_row.get("dragons")),
                opp_dragons=_safe(side_row.get("opp_dragons")),
                barons=_safe(side_row.get("barons")),
                heralds=_safe(side_row.get("heralds")),
                void_grubs=_safe(side_row.get("void_grubs")),
                first_dragon=_safe(side_row.get("firstdragon")),
                first_baron=_safe(side_row.get("firstbaron")),
                first_herald=_safe(side_row.get("firstherald")),
                dpm=_safe(side_row.get("dpm")),
                cspm=_safe(side_row.get("cspm")),
                earned_gold=_safe(side_row.get("earnedgold")),
                towers=_safe(side_row.get("towers")),
                opp_towers=_safe(side_row.get("opp_towers")),
                turret_plates=_safe(side_row.get("turretplates")),
                inhibitors=_safe(side_row.get("inhibitors")),
                vision_score=_safe(side_row.get("visionscore")),
                wards_placed_val=_safe(side_row.get("wardsplaced")),
                wards_killed_val=_safe(side_row.get("wardskilled")),
                team_kpm_val=_safe(side_row.get("team kpm")),
                killsat15_val=_safe(side_row.get("killsat15")),
            )
        games_processed += 1

    result_df = pd.DataFrame(records)
    logger.info(f"  {len(result_df)} feature rows built from {games_processed} games")
    logger.info(f"  {len(result_df.columns)} columns (was 30 in V2)")
    return result_df


if __name__ == "__main__":
    df = build_feature_dataset()
    if not df.empty:
        print(f"\nFeature dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Date range: {df['date'].min()} → {df['date'].max()}")
        feature_cols = [c for c in df.columns if c not in ("gameid", "date", "league", "result")]
        print(f"Feature columns ({len(feature_cols)}):")
        for c in sorted(feature_cols):
            print(f"  {c}")
