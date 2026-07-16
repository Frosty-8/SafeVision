from __future__ import annotations

from torch.utils.data import DataLoader

from app.evaluation.evaluator import Evaluator
from app.schemas.detection import EvaluationReport

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)

class OcclusionAnalyzer:
    """
    Evaluates detector robustness
    under object occlusion.
    """
    def __init__(
        self,
        evaluator: Evaluator,
    ) -> None:
    
        self.evaluator = evaluator
    
        logger.info(
            "OcclusionAnalyzer initialized."
        )

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, EvaluationReport]:
        """
        Evaluate detector under different
        occlusion levels.
        """
    
        title(
            "Occlusion Analysis",
        )
    
        reports = {
    
            "none": self._evaluate_no_occlusion(
                dataloader,
            ),
    
            "light": self._evaluate_light_occlusion(
                dataloader,
            ),
    
            "medium": self._evaluate_medium_occlusion(
                dataloader,
            ),
    
            "heavy": self._evaluate_heavy_occlusion(
                dataloader,
            ),
    
        }
    
        success(
            "Occlusion analysis completed."
        )
    
        return reports

    def _evaluate_no_occlusion(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Baseline evaluation.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()

    
    
    def _evaluate_light_occlusion(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate light occlusion.
    
        TODO:
        Apply 20% object occlusion.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()


    def _evaluate_medium_occlusion(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate medium occlusion.
    
        TODO:
        Apply 50% object occlusion.
        """
    
        self.evaluator.dataloader = dataloader
    
        return self.evaluator.evaluate()


    def _evaluate_heavy_occlusion(
        self,
        dataloader: DataLoader,
    ) -> EvaluationReport:
        """
        Evaluate heavy occlusion.
    
        TODO:
        Apply 80% object occlusion.
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
        Compare detector performance.
        """
    
        baseline = reports["none"]
    
        comparison = {}
    
        for level, report in reports.items():
    
            comparison[level] = (
    
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
        Display evaluation summary.
        """
    
        logger.info(
            "Occlusion Analysis Summary"
        )
    
        for level, report in reports.items():
    
            logger.info(
    
                "%s : %.4f",
    
                level,
    
                report.map,
    
            )