from __future__ import annotations

from torch.utils.data import DataLoader

from app.evaluation.evaluator import Evaluator
from app.schemas.detection import EvaluationReport

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)

class SliceAnalyzer:
    """
    Evaluate detector on dataset slices.
    """
    def __init__(
        self,
        evaluator: Evaluator,
    ) -> None:
    
        self.evaluator = evaluator
    
        logger.info(
            "SliceAnalyzer initialized."
        )

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, EvaluationReport]:
        """
        Evaluate detector for different
        object scales.
        """
    
        title(
            "Slice Analysis",
        )
    
        reports = {
    
            "small": self._evaluate_small(
                dataloader,
            ),
    
            "medium": self._evaluate_medium(
                dataloader,
            ),
    
            "large": self._evaluate_large(
                dataloader,
            ),
    
        }
    
        success(
            "Slice analysis completed."
        )
    
        return reports


    def _evaluate_small(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate small objects.
    
        TODO:
        Filter dataset for
        small bounding boxes.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()

    def _evaluate_medium(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate medium objects.
    
        TODO:
        Filter dataset for
        medium bounding boxes.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()


    def _evaluate_large(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate large objects.
    
        TODO:
        Filter dataset for
        large bounding boxes.
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
        Compare object scales.
        """
    
        comparison = {}
    
        for size, report in reports.items():
    
            comparison[size] = report.map
    
        return comparison

    def summary(
        self,
        reports: dict[
            str,
            EvaluationReport,
        ],
    ) -> None:
        """
        Display slice summary.
        """
    
        logger.info(
            "Slice Analysis Summary"
        )
    
        for size, report in reports.items():
    
            logger.info(
    
                "%s : %.4f",
    
                size,
    
                report.map,
    
            )