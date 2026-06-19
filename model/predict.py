"""
Win probability predictor.

Given two teams, computes blended ratings and outputs P(Team A wins).

Run standalone:
  python model/predict.py "Team A" "Team B"
  python model/predict.py --list          # show all teams
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from model.blend import get_all_ratings, get_team_rating
from model.calibration import PlattCalibrator
from model.pro_elo import DEFAULT_ELO, LEAGUE_TO_REGION, compute_regional_offsets

DB_PATH = _ROOT / "db" / "lol_model.db"

REGION_TIERS: Dict[str, int] = {
    "KR": 1, "CN": 1, "EU": 1,
    "NA": 2, "VN": 2, "PCS": 2, "TR": 2,
    "BR": 3, "JP": 3, "OCE": 3, "LAS": 3, "LAN": 3,
}

REGIONAL_ADJUSTMENT_FACTOR = 0.80

# Blue side wins 53.2% across 10,372 T2 matches → +22.3 ELO equivalent
BLUE_SIDE_ELO_OFFSET = 22.3

# Platt calibration (loaded once at module level)
_calibrator = PlattCalibrator()
_calibrator.load()


# ---------------------------------------------------------------------------
# Team region lookup
# ---------------------------------------------------------------------------
def get_team_region(team_name: str) -> Optional[Tuple[str, str]]:
    """Return (league, region) for a team, or None if not found."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT league FROM teams WHERE team_name = ?", (team_name,)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    league = row[0]
    region = LEAGUE_TO_REGION.get(league)
    if not region:
        return None
    return (league, region)


def get_regional_adjustment(region_a: str, region_b: str) -> float:
    """
    Compute the ELO adjustment to apply to team A's rating for a cross-region
    matchup. Positive means region A is stronger.

    Based on soloq-derived regional offsets (z-scored), scaled by
    REGIONAL_ADJUSTMENT_FACTOR to avoid overshoot.
    """
    if region_a == region_b:
        return 0.0
    offsets = compute_regional_offsets()
    offset_a = offsets.get(region_a, 0.0)
    offset_b = offsets.get(region_b, 0.0)
    return (offset_a - offset_b) * REGIONAL_ADJUSTMENT_FACTOR


def check_cross_region(team_a: str, team_b: str) -> Dict:
    """
    Check if two teams are from different regions.
    Returns dict with cross_region flag, regions, tiers, and warnings.
    """
    info_a = get_team_region(team_a)
    info_b = get_team_region(team_b)

    result = {
        "cross_region": False,
        "league_a": info_a[0] if info_a else None,
        "league_b": info_b[0] if info_b else None,
        "region_a": info_a[1] if info_a else None,
        "region_b": info_b[1] if info_b else None,
        "tier_a": REGION_TIERS.get(info_a[1], 0) if info_a else 0,
        "tier_b": REGION_TIERS.get(info_b[1], 0) if info_b else 0,
        "warnings": [],
    }

    if not info_a or not info_b:
        return result

    if info_a[1] != info_b[1]:
        result["cross_region"] = True
        result["warnings"].append(
            f"Cross-region matchup ({info_a[1]}/{info_a[0]} vs {info_b[1]}/{info_b[0]}). "
            f"A soloq-derived regional adjustment has been applied, but these teams "
            f"never play each other in T2 leagues — treat this prediction with less "
            f"confidence than same-region matchups."
        )
        tier_a = result["tier_a"]
        tier_b = result["tier_b"]
        if tier_a != tier_b and tier_a > 0 and tier_b > 0:
            stronger = team_a if tier_a < tier_b else team_b
            result["warnings"].append(
                f"Regional tier gap: Tier {tier_a} vs Tier {tier_b}. "
                f"The regional adjustment favors {stronger}'s region based on soloq "
                f"strength, but international performance depends on factors (coaching, "
                f"meta adaptation, stage experience) that soloq cannot capture."
            )

    return result


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def win_probability(rating_a: float, rating_b: float, scale: float = 400.0) -> float:
    """P(A wins) given two ELO-scale ratings."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / scale))


def bo_series_probability(p: float, best_of: int) -> float:
    """
    Convert a single-game win probability into a best-of-N series probability.

    P(win Bo5) = P(win 3+) = sum of binomial terms where A wins 3 out of
    the first 3, 4, or 5 games (the series stops when one side hits 3).

    Supports Bo1, Bo3, Bo5, Bo7.
    """
    if best_of == 1:
        return p
    wins_needed = (best_of + 1) // 2
    q = 1.0 - p
    total = 0.0
    for games_played in range(wins_needed, best_of + 1):
        losses = games_played - wins_needed
        from math import comb
        # A must win the final game, and win (wins_needed-1) of the prior games
        ways = comb(games_played - 1, losses)
        total += ways * (p ** wins_needed) * (q ** losses)
    return total


def predict_match(
    team_a: str,
    team_b: str,
    blend_k: int = 5,
    scale: float = 400.0,
    best_of: int = 1,
    side_a: Optional[str] = None,
) -> Dict:
    """
    End-to-end prediction for a matchup.

    side_a: "blue" or "red" if team_a's side is known (applies side offset).
            None if unknown (no side adjustment).
    Applies Platt calibration to shrink overconfident predictions.
    For cross-region matches, applies regional strength adjustment.
    If best_of > 1, includes series win probabilities.
    """
    rating_a = get_team_rating(team_a, blend_k=blend_k)
    rating_b = get_team_rating(team_b, blend_k=blend_k)

    if rating_a == DEFAULT_ELO:
        logger.warning(f"'{team_a}' not found in teams table — using default {DEFAULT_ELO}")
    if rating_b == DEFAULT_ELO:
        logger.warning(f"'{team_b}' not found in teams table — using default {DEFAULT_ELO}")

    region_check = check_cross_region(team_a, team_b)

    adj_rating_a = rating_a
    adj_rating_b = rating_b
    regional_adj = 0.0
    if region_check["cross_region"] and region_check["region_a"] and region_check["region_b"]:
        regional_adj = get_regional_adjustment(region_check["region_a"], region_check["region_b"])
        adj_rating_a = rating_a + regional_adj
        logger.info(
            f"Regional adjustment: {region_check['region_a']} vs {region_check['region_b']} "
            f"→ {regional_adj:+.1f} ELO to {team_a}"
        )

    # Apply blue side offset if side is known
    side_offset = 0.0
    if side_a == "blue":
        side_offset = BLUE_SIDE_ELO_OFFSET
    elif side_a == "red":
        side_offset = -BLUE_SIDE_ELO_OFFSET

    p_a_raw = win_probability(adj_rating_a + side_offset, adj_rating_b, scale)

    # Apply Platt calibration
    p_a = _calibrator.calibrate(p_a_raw) if _calibrator.fitted else p_a_raw

    for w in region_check["warnings"]:
        logger.warning(w)

    result = {
        "team_a": team_a,
        "team_b": team_b,
        "rating_a": round(rating_a, 1),
        "rating_b": round(rating_b, 1),
        "p_a_raw": round(p_a_raw, 4),
        "p_a": round(p_a, 4),
        "p_b": round(1.0 - p_a, 4),
        "calibrated": _calibrator.fitted,
        "side_a": side_a,
        "best_of": best_of,
        "cross_region": region_check["cross_region"],
        "region_a": region_check["region_a"],
        "region_b": region_check["region_b"],
        "tier_a": region_check["tier_a"],
        "tier_b": region_check["tier_b"],
        "warnings": region_check["warnings"],
    }

    if region_check["cross_region"]:
        result["regional_adjustment"] = round(regional_adj, 1)
        result["adj_rating_a"] = round(adj_rating_a, 1)
        result["adj_rating_b"] = round(adj_rating_b, 1)

    if best_of > 1:
        series_a = bo_series_probability(p_a, best_of)
        result["series_p_a"] = round(series_a, 4)
        result["series_p_b"] = round(1.0 - series_a, 4)

    return result


def predict_from_ratings(
    rating_a: float,
    rating_b: float,
    scale: float = 400.0,
) -> float:
    """Direct rating-to-probability for use by backtest.py."""
    return win_probability(rating_a, rating_b, scale)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def list_teams() -> None:
    """Print all teams sorted by blended rating."""
    ratings = get_all_ratings()
    if not ratings:
        logger.error("No teams in DB — run pro_elo.py first")
        return

    sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'#':>4}  {'Team':30}  {'Rating':>8}")
    print("-" * 48)
    for i, (team, rating) in enumerate(sorted_teams, 1):
        print(f"{i:4}  {team:30}  {rating:8.1f}")
    print(f"\n{len(sorted_teams)} teams total")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LoL T2 win probability predictor")
    parser.add_argument("teams", nargs="*", help="Two team names to predict")
    parser.add_argument("--list", action="store_true", help="List all teams by rating")
    parser.add_argument("--blend-k", type=int, default=10, help="Blend denominator")
    parser.add_argument("--scale", type=float, default=400.0, help="ELO scale factor")
    parser.add_argument("--bo", type=int, default=1, choices=[1, 3, 5, 7], help="Best-of series length (1, 3, 5, or 7)")
    parser.add_argument("--side", type=str, choices=["blue", "red"], default=None, help="Side for first team (blue/red)")
    args = parser.parse_args()

    if args.list:
        list_teams()
        return

    if len(args.teams) != 2:
        parser.error("Provide exactly 2 team names, e.g.: python model/predict.py 'Solary' 'Karmine Corp'")

    result = predict_match(
        args.teams[0], args.teams[1],
        blend_k=args.blend_k, scale=args.scale, best_of=args.bo,
        side_a=args.side,
    )

    print(f"\n  {result['team_a']}  vs  {result['team_b']}")
    if result.get("region_a") and result.get("region_b"):
        print(f"  Region: {result['region_a']} (Tier {result['tier_a']})  vs  {result['region_b']} (Tier {result['tier_b']})")
    print(f"  Rating: {result['rating_a']:.1f}  vs  {result['rating_b']:.1f}")

    if result.get("cross_region") and result.get("regional_adjustment"):
        adj = result["regional_adjustment"]
        print(f"  Regional adj: {adj:+.1f} ELO to {result['team_a']}")
        print(f"  Adj Rating: {result['adj_rating_a']:.1f}  vs  {result['adj_rating_b']:.1f}")

    side_str = f" ({result['side_a']} side)" if result.get("side_a") else ""
    cal_str = " [calibrated]" if result.get("calibrated") else ""
    if result.get("p_a_raw") and abs(result["p_a_raw"] - result["p_a"]) > 0.001:
        print(f"  Raw:    {result['p_a_raw']*100:.1f}%  vs  {(1-result['p_a_raw'])*100:.1f}%")
    print(f"  Game:   {result['p_a']*100:.1f}%  vs  {result['p_b']*100:.1f}%{side_str}{cal_str}")

    if result.get("series_p_a") is not None:
        print(f"  Bo{result['best_of']}:   {result['series_p_a']*100:.1f}%  vs  {result['series_p_b']*100:.1f}%")

    if result.get("warnings"):
        print(f"\n  ⚠ WARNINGS:")
        for w in result["warnings"]:
            print(f"    • {w}")
    print()


if __name__ == "__main__":
    main()
