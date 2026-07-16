from __future__ import annotations

from torch.utils.data import DataLoader

from app.evaluation.evaluator import Evaluator
from app.schemas.detection import EvaluationReport

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title
)

class NightAnalyzer:
    """
    Evaluates detector robustness under low-light conditions.
    """
    def __init__(
        self,
        evaluator: Evaluator,
    ) -> None:
    
        self.evaluator = evaluator
    
        logger.info(
            "NightAnalyzer initialized."
        )

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, EvaluationReport]:
        """
        Evaluate detector under
        multiple lighting conditions.
        """
    
        title(
            "Night Analysis",
        )
    
        reports = {
    
            "daylight": self._evaluate_daylight(
                dataloader,
            ),
    
            "dusk": self._evaluate_dusk(
                dataloader,
            ),
    
            "night": self._evaluate_night(
                dataloader,
            ),
    
            "headlights": self._evaluate_headlights(
                dataloader,
            ),
    
            "streetlights": self._evaluate_streetlights(
                dataloader,
            ),
    
        }
    
        success(
            "Night analysis completed."
        )
    
        return reports

    def _evaluate_daylight(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Baseline evaluation.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()

    def _evaluate_dusk(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate under dusk lighting.
    
        TODO:
        Apply dusk transform.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()

    def _evaluate_night(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate under nighttime.
    
        TODO:
        Apply night transform.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()

    def _evaluate_headlights(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate under headlight glare.
    
        TODO:
        Apply headlight transform.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()

    def _evaluate_streetlights(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate under street lighting.
    
        TODO:
        Apply streetlight transform.
        """
    
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
        Compare all lighting conditions.
        """
    
        baseline = reports["daylight"]
    
        comparison = {}
    
        for name, report in reports.items():
    
            comparison[name] = (
    
                report.map
    
                - baseline.map
    
            )
    
        return comparison

    def summary(
        self,
        reports: dict[
            str,
            EvaluationReport,
        ],
    ) -> None:
        """
        Display summary.
        """
    
        logger.info(
            "Night Analysis Summary"
        )
    
        for name, report in reports.items():
    
            logger.info(
    
                "%s : %.4f",
    
                name,
    
                report.map,
    
            )