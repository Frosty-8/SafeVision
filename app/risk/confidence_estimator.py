from __future__ import annotations

import torch

from app.schemas.detection import DetectionOutput

from app.utils.logger import logger

class ConfidenceEstimator:
    """
    Estimates calibrated confidence statistics for detection
    """

    def __init__(
        self,
        minimum_confidence: float = 0.5,
    ) -> None:
    
        if not 0.0 <= minimum_confidence <= 1.0:
    
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )
    
        self.minimum_confidence = minimum_confidence
    
        logger.info(
            "ConfidenceEstimator initialized."
        )

    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, float]:
        """
        Callable interface.
        """
    
        return self.estimate(
            outputs,
        )

    def estimate(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, float]:
        """
        Estimate confidence statistics.
        """
    
        confidence = self.normalize(
            outputs.confidence,
        )
    
        mean = confidence.mean().item()
    
        minimum = confidence.min().item()
    
        maximum = confidence.max().item()
    
        std = confidence.std().item()
    
        below_threshold = int(
    
            (
                confidence
                < self.minimum_confidence
            ).sum().item()
    
        )
    
        return {
    
            "mean": mean,
    
            "minimum": minimum,
    
            "maximum": maximum,
    
            "std": std,
    
            "below_threshold": below_threshold,
    
        }

    def normalize(
        self,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clamp confidence values.
        """
    
        return confidence.clamp(
            min=0.0,
            max=1.0,
        )

    def confidence_level(
        self,
        confidence: float,
    ) -> str:
        """
        Convert confidence score into
        a qualitative level.
        """
    
        if confidence >= 0.90:
    
            return "very_high"
    
        if confidence >= 0.75:
    
            return "high"
    
        if confidence >= 0.50:
    
            return "medium"
    
        if confidence >= 0.25:
    
            return "low"
    
        return "very_low"

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Configuration summary.
        """
    
        return {
    
            "minimum_confidence": self.minimum_confidence,
    
        }