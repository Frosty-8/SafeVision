from __future__ import annotations

from app.metrics.confusion_matrix import (
    ConfusionMatrix
)

from app.utils.logger import logger

class FalseNegativeRate:
    """
    Computes False Negative rate.
    """
    def __init__(
        self,
        confusion_matrix: ConfusionMatrix,
    ) -> None:
    
        self.confusion_matrix = confusion_matrix
    
        logger.info(
            "FalseNegativeRate initialized."
        )

    def compute(
        self,
    ) -> float:
        """
        Compute False Negative Rate.
        """
    
        tp = self.confusion_matrix.true_positive
    
        fn = self.confusion_matrix.false_negative
    
        denominator = tp + fn
    
        if denominator == 0:
    
            return 0.0
    
        return fn / denominator

    def average(
        self,
    ) -> float:
        """
        Average False Negative Rate.
        """
    
        return self.compute()

    def reset(
        self,
    ) -> None:
        """
        Reset underlying statistics.
        """
    
        self.confusion_matrix.reset()

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Metric summary.
        """
    
        return {
    
            "false_negative_rate": self.compute(),
    
        }