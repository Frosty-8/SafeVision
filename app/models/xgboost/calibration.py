"""
SafeVision AI

Probability Calibration

Calibrates XGBoost prediction probabilities.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from pathlib import Path

from app.schemas.xgboost import CalibrationMethod
import joblib
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from xgboost import XGBClassifier

from app.utils.logger import logger


class ProbabilityCalibrator:
    """
    Calibrates probability estimates
    produced by the XGBoost model.
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.SIGMOID,
    ) -> None:

        if method not in {
            "sigmoid",
            "isotonic",
        }:

            raise ValueError("Unsupported calibration method.")

        self.method = method

        self.model: CalibratedClassifierCV | None = None

        logger.info("ProbabilityCalibrator initialized.")

    def fit(
        self,
        classifier: XGBClassifier,
        x,
        y,
    ) -> None:
        """
        Fit the probability calibrator.
        """

        self.model = CalibratedClassifierCV(
            estimator=classifier,
            method=self.method,
            cv="prefit",
        )

        self.model.fit(
            x,
            y,
        )

        logger.info("Calibration model fitted.")

    def predict(
        self,
        x,
    ) -> np.ndarray:
        """
        Return calibrated probabilities.
        """

        self._check_fitted()

        return self.model.predict_proba(
            x,
        )[:, 1]

    def evaluate(
        self,
        x,
        y,
    ) -> float:
        """
        Evaluate calibration quality.
        """

        probabilities = self.predict(
            x,
        )

        return brier_score_loss(
            y,
            probabilities,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save calibration model.
        """

        self._check_fitted()

        joblib.dump(
            self.model,
            path,
        )

        logger.info("Calibration model saved.")

    def load(
        self,
        path: str | Path,
    ) -> None:
        """
        Load calibration model.
        """

        self.model = joblib.load(
            path,
        )

        logger.info("Calibration model loaded.")

    def _check_fitted(
        self,
    ) -> None:

        if self.model is None:

            raise RuntimeError("Calibration model has not been fitted.")

    @property
    def is_fitted(
        self,
    ) -> bool:
        """
        Whether calibration model
        has been fitted.
        """

        return self.model is not None
