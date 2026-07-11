"""
SafeVision AI

Training History

Stores metrics collected during training.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import json

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.schemas.history import EpochMetrics

from app.utils.logger import logger


class TrainingHistory:
    """
    Stores training history.
    """

    def __init__(
        self,
    ) -> None:

        self.history: list[EpochMetrics] = []

        logger.info("TrainingHistory initialized.")

    def update(
        self,
        metrics: EpochMetrics,
    ) -> None:
        """
        Add epoch metrics.
        """

        self.history.append(
            metrics,
        )

    def latest(
        self,
    ) -> EpochMetrics | None:
        """
        Return latest epoch.
        """

        if not self.history:

            return None

        return self.history[-1]

    def best(
        self,
        monitor: str = "validation_loss",
        mode: str = "min",
    ) -> EpochMetrics | None:
        """
        Return best epoch.
        """

        if not self.history:

            return None

        return (
            min(
                self.history,
                key=lambda metric: getattr(
                    metric,
                    monitor,
                ),
            )
            if mode == "min"
            else max(
                self.history,
                key=lambda metric: getattr(
                    metric,
                    monitor,
                ),
            )
        )

    def summary(
        self,
    ) -> dict[str, float]:

        best = self.best()

        latest = self.latest()

        if best is None or latest is None:

            return {}

        return {
            "epochs": len(
                self.history,
            ),
            "best_validation_loss": best.validation_loss,
            "latest_train_loss": latest.train_loss,
            "latest_validation_loss": latest.validation_loss,
        }

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save history.
        """

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                [asdict(item) for item in self.history],
                file,
                indent=4,
            )

        logger.info("Training history saved.")

    def load(
        self,
        path: str | Path,
    ) -> None:
        """
        Load history.
        """

        with open(
            path,
            encoding="utf-8",
        ) as file:

            data = json.load(
                file,
            )

        self.history = [
            EpochMetrics(
                **item,
            )
            for item in data
        ]

    def dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return history as DataFrame.
        """

        return pd.DataFrame([asdict(item) for item in self.history])

    def clear(
        self,
    ) -> None:

        self.history.clear()
