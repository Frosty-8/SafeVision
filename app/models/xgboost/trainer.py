from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.schemas.xgboost import (
    XGBoostTrainingReport,
)
from app.utils.logger import logger


class XGBoostTrainer:
    """
    Trainer for the XGBoost
    risk estimation model.
    """

    def __init__(
        self,
        random_state: int = 42,
    ) -> None:

        self.random_state = random_state

        self.model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        )

        logger.info("XGBoostTrainer initialized.")

    def train(
        self,
        dataframe: pd.DataFrame,
        target_column: str = "false_negative",
    ) -> XGBoostTrainingReport:
        """
        Train the XGBoost model.
        """

        start = time.perf_counter()

        x = dataframe.drop(
            columns=[target_column],
        )

        y = dataframe[target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        self.model.fit(
            x_train,
            y_train,
        )

        elapsed = time.perf_counter() - start

        return self.evaluate(
            x_test,
            y_test,
            elapsed,
        )

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        training_time: float,
    ) -> XGBoostTrainingReport:
        """
        Evaluate the trained model.
        """

        predictions = self.model.predict(
            x_test,
        )

        probabilities = self.model.predict_proba(
            x_test,
        )[:, 1]

        return XGBoostTrainingReport(
            accuracy=accuracy_score(
                y_test,
                predictions,
            ),
            precision=precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            recall=recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            f1_score=f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            roc_auc=roc_auc_score(
                y_test,
                probabilities,
            ),
            training_time=training_time,
            num_samples=len(
                x_test,
            ),
            feature_importance=self.feature_importance(),
        )

    def feature_importance(
        self,
    ) -> dict[str, float]:
        """
        Return feature importance scores.
        """

        booster = self.model.get_booster()

        scores = booster.get_score(
            importance_type="gain",
        )

        return dict(
            sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save trained model.
        """

        joblib.dump(
            self.model,
            path,
        )

        logger.info(
            "Saved XGBoost model to '{}'.",
            path,
        )

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
    ) -> pd.DataFrame:
        """
        Predict probabilities.
        """

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
