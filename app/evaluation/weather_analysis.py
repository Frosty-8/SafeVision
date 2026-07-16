from __future__ import annotations

from torch.utils.data import DataLoader

from app.evaluation.evaluator import Evaluator
from app.schemas.detection import EvaluationReport

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)


class WeatherAnalyzer:
    """
    Weather robustness evaluator.
    """

    def __init__(
        self,
        evaluator: Evaluator,
    ) -> None:

        self.evaluator = evaluator

        logger.info("WeatherAnalyzer initialized.")

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, EvaluationReport]:
        """
        Evaluate weather robustness.
        """

        title(
            "Weather Analysis",
        )

        reports = {
            "clear": self._evaluate_clear(
                dataloader,
            ),
            "rain": self._evaluate_rain(
                dataloader,
            ),
            "fog": self._evaluate_fog(
                dataloader,
            ),
            "snow": self._evaluate_snow(
                dataloader,
            ),
            "glare": self._evaluate_glare(
                dataloader,
            ),
            "motion_blur": self._evaluate_motion_blur(
                dataloader,
            ),
        }

        success("Weather analysis completed.")

        return reports

    def _evaluate_clear(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Baseline evaluation.
        """

        self.evaluator.dataloader = dataloader

        return self.evaluator.evaluate()

    def _evaluate_rain(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Rain evaluation.
        """

        #
        # TODO:
        # Apply rain transform
        #

        self.evaluator.dataloader = dataloader

        return self.evaluator.evaluate()

    def compare(
        self,
        reports: dict[
            str,
            EvaluationReport,
        ],
    ) -> dict[str, float]:
        """
        Compare weather conditions.
        """

        baseline = reports["clear"]

        comparison = {}

        for name, report in reports.items():

            comparison[name] = report.map - baseline.map

        return comparison

    def summary(
        self,
        reports: dict[
            str,
            EvaluationReport,
        ],
    ) -> None:
        """
        Print summary.
        """

        logger.info("Weather Analysis Summary")

        for weather, report in reports.items():

            logger.info(
                "%s : %.4f",
                weather,
                report.map,
            )
