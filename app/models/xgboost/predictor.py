from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app.schemas.xgboost import (
    RiskPrediction,
)
from app.utils.logger import logger


class XGBoostPredictor:
    """
    Performs inference using the trained
    XGBoost risk model.
    """

    def __init__(
        self,
    ) -> None:

        self.model = None

        logger.info("XGBoostPredictor initialized.")

    def load(
        self,
        path: str | Path,
    ) -> None:
        """
        Load trained model.
        """

        self.model = joblib.load(
            path,
        )

        logger.info(
            "Loaded XGBoost model from '{}'.",
            path,
        )

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> RiskPrediction:
        """
        Predict risk for one sample.
        """

        self._check_loaded()

        probability = float(
            self.model.predict_proba(
                dataframe,
            )[
                0
            ][1]
        )

        prediction = int(
            self.model.predict(
                dataframe,
            )[0]
        )

        return RiskPrediction(
            probability=probability,
            risk_score=probability,
            decision=("unsafe" if prediction else "safe"),
        )

    def predict_batch(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict multiple samples.
        """

        self._check_loaded()

        probabilities = self.model.predict_proba(
            dataframe,
        )[:, 1]

        predictions = self.model.predict(
            dataframe,
        )

        return pd.DataFrame(
            {
                "prediction": predictions,
                "probability": probabilities,
            }
        )

    def predict_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> float:
        """
        Return prediction probability.
        """

        self._check_loaded()

        return float(
            self.model.predict_proba(
                dataframe,
            )[
                0
            ][1]
        )

    def _check_loaded(
        self,
    ) -> None:
        """
        Ensure model is loaded.
        """

        if self.model is None:

            raise RuntimeError("Model has not been loaded.")

    @property
    def is_loaded(
        self,
    ) -> bool:
        """
        Whether the model has been loaded.
        """

        return self.model is not None
