"""
Platt scaling calibration — learns to correct overconfident predictions
by fitting a logistic regression on the model's own outputs.

The v1 ELO model says 87% but teams only win 80%. This module shrinks
extreme predictions toward realistic probabilities.

Usage:
    from model.calibration import PlattCalibrator
    cal = PlattCalibrator()
    cal.fit(predictions, actuals)  # list of (p, 0/1) pairs
    calibrated_p = cal.calibrate(raw_p)
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

CALIBRATION_PATH = _ROOT / "model" / "calibration_params.json"


class PlattCalibrator:
    """
    Platt scaling: fits P(y=1 | f) = 1 / (1 + exp(A*f + B))
    where f = log(p / (1-p)) is the log-odds of the raw prediction.
    """

    def __init__(self) -> None:
        self.a = -1.0  # identity by default (no correction)
        self.b = 0.0
        self.fitted = False

    def fit(self, predictions: List[Tuple[float, float]]) -> None:
        """
        Fit calibration from (raw_probability, actual_outcome) pairs.
        actual_outcome is 1.0 for correct, 0.0 for incorrect.
        """
        if len(predictions) < 50:
            logger.warning("Too few predictions for calibration fitting")
            return

        eps = 1e-7
        X = []
        y = []
        for p, actual in predictions:
            p_clipped = max(eps, min(1.0 - eps, p))
            log_odds = math.log(p_clipped / (1.0 - p_clipped))
            X.append(log_odds)
            y.append(actual)

        X = np.array(X).reshape(-1, 1)
        y = np.array(y)

        # Fit logistic regression: P(y=1) = 1 / (1 + exp(-(a*x + b)))
        # Use sklearn for robustness
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        lr.fit(X, y)

        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        self.fitted = True

        logger.info(f"Platt calibration fitted: a={self.a:.4f}, b={self.b:.4f}")

    def calibrate(self, raw_p: float) -> float:
        """Apply Platt scaling to a raw probability."""
        if not self.fitted:
            return raw_p

        eps = 1e-7
        p_clipped = max(eps, min(1.0 - eps, raw_p))
        log_odds = math.log(p_clipped / (1.0 - p_clipped))
        calibrated_log_odds = self.a * log_odds + self.b
        return 1.0 / (1.0 + math.exp(-calibrated_log_odds))

    def save(self, path: Optional[Path] = None) -> None:
        path = path or CALIBRATION_PATH
        data = {"a": self.a, "b": self.b, "fitted": self.fitted}
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Calibration params saved → {path}")

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or CALIBRATION_PATH
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.a = data["a"]
        self.b = data["b"]
        self.fitted = data["fitted"]
        return True
