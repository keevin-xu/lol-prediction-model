"""
Draft-derived ELO adjustments.

Computes walk-forward draft signals from newmetrics CSVs and exposes
them as additive ELO offsets for the backtester.

Signals:
  A — Champion-role win rate differential (patch-aware, Bayes-shrunk)
  B — Player-champion mastery (signature pick count differential)
  D — Champion pool depth differential
  E — Patch freshness (games-on-current-patch per team, for Kelly scaling)

All trackers use strict game-sequence ordering: within a date,
games are ordered by gameid to prevent intra-day leakage.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

_ROOT = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data" / "newmetrics"

SHRINKAGE_K = 20
MASTERY_MIN_GAMES = 5
MASTERY_WR_THRESHOLD = 0.55
POOL_WINDOW = 30


class DraftTracker:
    """
    Walk-forward draft feature tracker.

    Loads all draft data into memory, indexed by gameid.
    Maintains rolling state that updates strictly in game-sequence order.
    """

    def __init__(self) -> None:
        # Raw data indexed by gameid
        self.game_picks: Dict[str, List[Dict]] = defaultdict(list)
        self.game_pickbans: Dict[str, Dict[str, Dict]] = {}
        self.game_meta: Dict[str, Dict] = {}

        # Walk-forward state for Signal A (champion-role WR)
        # (champion, role, patch_major) -> [wins, total]
        self._champ_role_patch: Dict[Tuple[str, str, str], List[int]] = defaultdict(lambda: [0, 0])
        # (champion, role) -> [wins, total] (global prior)
        self._champ_role_global: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])

        # Walk-forward state for Signal B (player-champion mastery)
        # (player, champion) -> [wins, total]
        self._player_champ: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])

        # Walk-forward state for Signal D (champion pool depth)
        # player -> list of (gameid, champion) for rolling window
        self._player_history: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # Walk-forward state for Signal E (patch freshness)
        # team -> {patch_major: game_count}
        self._team_patch_games: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Ordered game list for walk-forward
        self._game_order: List[str] = []
        self._processed_up_to: int = 0

    def load(self) -> None:
        """Load CSVs into memory."""
        logger.info("Loading draft data from newmetrics/...")

        # Load draft picks (player-level)
        picks_path = DATA_DIR / "draft_picks.csv"
        with open(picks_path) as f:
            for row in csv.DictReader(f):
                self.game_picks[row["gameid"]].append(row)

        # Load team pickbans
        pickbans_path = DATA_DIR / "team_pickbans.csv"
        with open(pickbans_path) as f:
            for row in csv.DictReader(f):
                gid = row["gameid"]
                if gid not in self.game_pickbans:
                    self.game_pickbans[gid] = {}
                self.game_pickbans[gid][row["side"]] = row

        # Load game metadata (for patch info)
        games_path = DATA_DIR / "games.csv"
        with open(games_path) as f:
            for row in csv.DictReader(f):
                self.game_meta[row["gameid"]] = row

        # Build game order: sort by (date, gameid) for strict sequencing
        self._game_order = sorted(
            self.game_meta.keys(),
            key=lambda gid: (self.game_meta[gid]["date"], gid),
        )

        logger.info(
            "  %d games, %d pick rows, %d pickban rows loaded",
            len(self.game_meta),
            sum(len(v) for v in self.game_picks.values()),
            len(self.game_pickbans),
        )

    def _major_patch(self, patch: str) -> str:
        """Extract major patch version: '14.12' -> '14.12', '14.12b' -> '14.12'."""
        parts = patch.split(".")
        if len(parts) >= 2:
            return parts[0] + "." + "".join(c for c in parts[1] if c.isdigit())
        return patch

    def advance_to(self, target_gameid: str) -> None:
        """
        Process all games BEFORE target_gameid in sequence order.
        Updates walk-forward state with their results.
        """
        while self._processed_up_to < len(self._game_order):
            gid = self._game_order[self._processed_up_to]
            if gid == target_gameid:
                break

            meta = self.game_meta.get(gid)
            if meta and (meta["date"], gid) >= (self.game_meta.get(target_gameid, {}).get("date", ""), target_gameid):
                break

            self._update_state(gid)
            self._processed_up_to += 1

    def _update_state(self, gameid: str) -> None:
        """Update all trackers with one game's results."""
        meta = self.game_meta.get(gameid, {})
        patch_major = self._major_patch(meta.get("patch", ""))
        winner_side = meta.get("winner", "")

        picks = self.game_picks.get(gameid, [])
        for pick in picks:
            champ = pick["champion"]
            role = pick["position"]
            player = pick["playername"]
            team = pick["teamname"]
            won = int(pick["result"])

            # Signal A: champion-role WR (patch-aware + global)
            self._champ_role_patch[(champ, role, patch_major)][0] += won
            self._champ_role_patch[(champ, role, patch_major)][1] += 1
            self._champ_role_global[(champ, role)][0] += won
            self._champ_role_global[(champ, role)][1] += 1

            # Signal B: player-champion mastery
            self._player_champ[(player, champ)][0] += won
            self._player_champ[(player, champ)][1] += 1

            # Signal D: champion pool depth
            self._player_history[player].append((gameid, champ))
            if len(self._player_history[player]) > POOL_WINDOW * 2:
                self._player_history[player] = self._player_history[player][-POOL_WINDOW:]

        # Signal E: patch freshness (per team)
        blue_team = meta.get("blue_team", "")
        red_team = meta.get("red_team", "")
        if blue_team and patch_major:
            self._team_patch_games[blue_team][patch_major] += 1
        if red_team and patch_major:
            self._team_patch_games[red_team][patch_major] += 1

    def get_champ_wr_shrunk(self, champion: str, role: str, patch_major: str) -> float:
        """Signal A: Bayes-shrunk champion-role win rate on this patch."""
        patch_stats = self._champ_role_patch.get((champion, role, patch_major))
        global_stats = self._champ_role_global.get((champion, role))

        # Global prior
        if global_stats and global_stats[1] > 0:
            global_wr = global_stats[0] / global_stats[1]
        else:
            global_wr = 0.5

        # Shrink patch WR toward global prior
        if patch_stats and patch_stats[1] > 0:
            n = patch_stats[1]
            raw_wr = patch_stats[0] / n
            return (n * raw_wr + SHRINKAGE_K * global_wr) / (n + SHRINKAGE_K)

        # No patch data: use shrunk global toward 0.5
        if global_stats and global_stats[1] > 0:
            n = global_stats[1]
            raw_wr = global_stats[0] / n
            return (n * raw_wr + SHRINKAGE_K * 0.5) / (n + SHRINKAGE_K)

        return 0.5

    def signal_a(self, gameid: str) -> Optional[float]:
        """
        Champion-role WR differential.
        Returns (blue_avg_wr - red_avg_wr) or None if data missing.
        """
        meta = self.game_meta.get(gameid)
        if not meta:
            return None
        patch_major = self._major_patch(meta.get("patch", ""))
        picks = self.game_picks.get(gameid, [])
        if len(picks) != 10:
            return None

        blue_wrs = []
        red_wrs = []
        for pick in picks:
            wr = self.get_champ_wr_shrunk(pick["champion"], pick["position"], patch_major)
            if pick["side"] == "Blue":
                blue_wrs.append(wr)
            else:
                red_wrs.append(wr)

        if len(blue_wrs) != 5 or len(red_wrs) != 5:
            return None

        blue_avg = sum(blue_wrs) / 5.0
        red_avg = sum(red_wrs) / 5.0
        return blue_avg - red_avg

    def signal_b(self, gameid: str) -> Optional[float]:
        """
        Player-champion mastery differential.
        Returns (blue_signature_count - red_signature_count) or None.
        """
        picks = self.game_picks.get(gameid, [])
        if len(picks) != 10:
            return None

        blue_sig = 0
        red_sig = 0
        for pick in picks:
            stats = self._player_champ.get((pick["playername"], pick["champion"]))
            if stats and stats[1] >= MASTERY_MIN_GAMES:
                wr = stats[0] / stats[1]
                if wr >= MASTERY_WR_THRESHOLD:
                    if pick["side"] == "Blue":
                        blue_sig += 1
                    else:
                        red_sig += 1

        return float(blue_sig - red_sig)

    def signal_d(self, gameid: str) -> Optional[float]:
        """
        Champion pool depth differential.
        Returns (blue_avg_depth - red_avg_depth) or None.
        """
        picks = self.game_picks.get(gameid, [])
        if len(picks) != 10:
            return None

        blue_depths = []
        red_depths = []
        for pick in picks:
            player = pick["playername"]
            history = self._player_history.get(player, [])
            recent = history[-POOL_WINDOW:]
            unique_champs = len(set(c for _, c in recent))
            if pick["side"] == "Blue":
                blue_depths.append(unique_champs)
            else:
                red_depths.append(unique_champs)

        if len(blue_depths) != 5 or len(red_depths) != 5:
            return None

        blue_avg = sum(blue_depths) / 5.0
        red_avg = sum(red_depths) / 5.0
        return blue_avg - red_avg

    def signal_e(self, gameid: str) -> Tuple[int, int]:
        """
        Patch freshness: games on current patch for each team.
        Returns (blue_games_on_patch, red_games_on_patch).
        """
        meta = self.game_meta.get(gameid, {})
        patch_major = self._major_patch(meta.get("patch", ""))
        blue_team = meta.get("blue_team", "")
        red_team = meta.get("red_team", "")

        blue_count = self._team_patch_games.get(blue_team, {}).get(patch_major, 0)
        red_count = self._team_patch_games.get(red_team, {}).get(patch_major, 0)
        return blue_count, red_count

    def compute_signals(self, gameid: str) -> Dict[str, Optional[float]]:
        """Compute all signals for a game. Must call advance_to(gameid) first."""
        return {
            "a": self.signal_a(gameid),
            "b": self.signal_b(gameid),
            "d": self.signal_d(gameid),
        }

    def reset(self) -> None:
        """Reset walk-forward state (for re-running with different params)."""
        self._champ_role_patch.clear()
        self._champ_role_global.clear()
        self._player_champ.clear()
        self._player_history.clear()
        self._team_patch_games.clear()
        self._processed_up_to = 0


if __name__ == "__main__":
    tracker = DraftTracker()
    tracker.load()

    # Quick sanity check on a few games
    test_games = tracker._game_order[1000:1005]
    for gid in test_games:
        tracker.advance_to(gid)
        signals = tracker.compute_signals(gid)
        meta = tracker.game_meta[gid]
        print(
            "%s  %s vs %s  A=%.4f  B=%.1f  D=%.1f"
            % (
                meta["date"][:10],
                meta["blue_team"][:15],
                meta["red_team"][:15],
                signals["a"] or 0,
                signals["b"] or 0,
                signals["d"] or 0,
            )
        )
