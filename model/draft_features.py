"""
Draft/champion features for V3 model.

Computes rolling champion win rates and player comfort scores from
player-level OE CSV rows. Outputs per-match draft quality features
that merge into the main feature dataset by gameid.

Walk-forward: only uses champion data from BEFORE each match.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

RAW_DIR = _ROOT / "data" / "raw" / "oracleselixir"

CHAMP_WINDOW = 300
COMFORT_WINDOW_DAYS = 30


class ChampionTracker:
    """Tracks rolling win rates per champion-role and player-champion comfort."""

    def __init__(self, window: int = CHAMP_WINDOW) -> None:
        self.window = window
        # (champion, role) → list of (result, date_str)
        self.champ_stats: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
        # (player, champion) → list of date_str
        self.player_champ_history: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    def update(self, champion: str, role: str, player: str, result: int, date: str) -> None:
        key = (champion, role)
        self.champ_stats[key].append((result, date))
        if len(self.champ_stats[key]) > self.window * 2:
            self.champ_stats[key] = self.champ_stats[key][-self.window:]

        pk = (player, champion)
        self.player_champ_history[pk].append(date)
        if len(self.player_champ_history[pk]) > 100:
            self.player_champ_history[pk] = self.player_champ_history[pk][-50:]

    def get_champ_wr(self, champion: str, role: str) -> Optional[float]:
        key = (champion, role)
        entries = self.champ_stats.get(key, [])
        recent = entries[-self.window:]
        if len(recent) < 5:
            return None
        return sum(r for r, _ in recent) / len(recent)

    def get_player_comfort(self, player: str, champion: str, current_date: str) -> int:
        pk = (player, champion)
        history = self.player_champ_history.get(pk, [])
        return sum(1 for d in history if d < current_date and d >= current_date[:8] + "01")

    def get_overall_wr_percentile(self, wr: float) -> float:
        all_wrs = []
        for entries in self.champ_stats.values():
            recent = entries[-self.window:]
            if len(recent) >= 5:
                all_wrs.append(sum(r for r, _ in recent) / len(recent))
        if not all_wrs:
            return 0.5
        below = sum(1 for w in all_wrs if w <= wr)
        return below / len(all_wrs)


def load_player_game_data() -> pd.DataFrame:
    from scrapers.oe_scraper import T2_LEAGUES, YEARS

    frames = []
    for year in YEARS:
        path = RAW_DIR / f"{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df = df[df["league"].isin(T2_LEAGUES)]
        player_rows = df[df["position"].isin(["top", "jng", "mid", "bot", "sup"])].copy()
        frames.append(player_rows)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    return df


def build_draft_features() -> pd.DataFrame:
    """
    Walk through all player-level rows chronologically.
    For each match, compute draft features BEFORE the game.
    Returns DataFrame with gameid + draft features.
    """
    logger.info("Loading player game data for draft features…")
    df = load_player_game_data()
    if df.empty:
        logger.error("No player data found")
        return pd.DataFrame()

    logger.info(f"  {len(df)} player-game rows loaded")

    tracker = ChampionTracker()
    records = []
    processed_games = set()

    for gameid, grp in df.groupby("gameid", sort=False):
        if gameid in processed_games:
            continue
        processed_games.add(gameid)

        blue = grp[grp["side"] == "Blue"]
        red = grp[grp["side"] == "Red"]
        if len(blue) < 5 or len(red) < 5:
            continue

        date = str(blue.iloc[0]["date"])

        # Compute draft features BEFORE updating tracker
        blue_features = _compute_side_draft(blue, tracker, date)
        red_features = _compute_side_draft(red, tracker, date)

        if blue_features and red_features:
            record = {"gameid": gameid}
            for k, v in blue_features.items():
                record[f"blue_{k}"] = v
            for k, v in red_features.items():
                record[f"red_{k}"] = v
            record["draft_wr_diff"] = blue_features["avg_champ_wr"] - red_features["avg_champ_wr"]
            record["draft_meta_diff"] = blue_features["meta_score"] - red_features["meta_score"]
            record["draft_comfort_diff"] = blue_features["avg_comfort"] - red_features["avg_comfort"]
            records.append(record)

        # Update tracker with results
        for _, row in grp.iterrows():
            champ = str(row.get("champion", ""))
            role = str(row.get("position", ""))
            player = str(row.get("playername", ""))
            result = int(row.get("result", 0))
            if champ and champ != "nan" and role in ("top", "jng", "mid", "bot", "sup"):
                tracker.update(champ, role, player, result, date)

    result_df = pd.DataFrame(records)
    logger.info(f"  {len(result_df)} draft feature rows built")
    return result_df


def _compute_side_draft(
    side_df: pd.DataFrame,
    tracker: ChampionTracker,
    date: str,
) -> Optional[Dict[str, float]]:
    wrs = []
    comforts = []
    meta_count = 0

    for _, row in side_df.iterrows():
        champ = str(row.get("champion", ""))
        role = str(row.get("position", ""))
        player = str(row.get("playername", ""))

        if not champ or champ == "nan":
            continue

        wr = tracker.get_champ_wr(champ, role)
        if wr is not None:
            wrs.append(wr)
            if tracker.get_overall_wr_percentile(wr) >= 0.80:
                meta_count += 1

        comfort = tracker.get_player_comfort(player, champ, date)
        comforts.append(comfort)

    if not wrs:
        return None

    return {
        "avg_champ_wr": sum(wrs) / len(wrs),
        "meta_score": meta_count / 5.0,
        "avg_comfort": sum(comforts) / max(len(comforts), 1),
        "min_champ_wr": min(wrs),
    }


if __name__ == "__main__":
    df = build_draft_features()
    if not df.empty:
        print(f"\nDraft features: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample (first row):")
        for col in df.columns:
            val = df[col].iloc[0]
            print(f"  {col}: {val}")
