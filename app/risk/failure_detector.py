from __future__ import annotations

from app.risk.confidence_estimator import (
    ConfidenceEstimator,
)

from app.risk.uncertainty import (
    UncertaintyEstimator,
)

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger


class FailureDetector:
    """
    Detect potentially unsafe predictions.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.50,
        uncertainty_threshold: float = 0.40,
    ) -> None:
    
        self.confidence_threshold = confidence_threshold
    
        self.uncertainty_threshold = uncertainty_threshold
    
        self.confidence_estimator = ConfidenceEstimator(
            minimum_confidence=confidence_threshold,
        )
    
        self.uncertainty_estimator = UncertaintyEstimator()
    
        logger.info(
            "FailureDetector initialized."
        )

    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, object]:
        """
        Callable interface.
        """
    
        return self.detect(
            outputs,
        )

    def detect(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, object]:
        """
        Detect prediction failures.
        """
    
        confidence = self.confidence_estimator(
            outputs,
        )
    
        uncertainty = self.uncertainty_estimator(
            outputs,
        )
    
        confidence_failures = self.confidence_failures(
            confidence,
        )
    
        uncertainty_failures = self.uncertainty_failures(
            uncertainty,
        )
    
        localization_failures = self.localization_failures(
            outputs,
        )
    
        total = (
    
            confidence_failures
    
            +
    
            uncertainty_failures
    
            +
    
            localization_failures
    
        )
    
        return {
    
            "confidence_failures": confidence_failures,
    
            "uncertainty_failures": uncertainty_failures,
    
            "localization_failures": localization_failures,
    
            "total_failures": total,
    
            "status": self.classify(
                total,
            ),
    
        }

    def confidence_failures(
        self,
        confidence: dict[str, float],
    ) -> int:
        """
        Count detections below
        confidence threshold.
        """
    
        return int(
            confidence["below_threshold"]
        )

    def uncertainty_failures(
        self,
        uncertainty: dict[str, float],
    ) -> int:
        """
        Detect high uncertainty.
        """
    
        return int(
    
            uncertainty["total"]
    
            >
    
            self.uncertainty_threshold
    
        )

    def localization_failures(
        self,
        outputs: DetectionOutput,
    ) -> int:
        """
        Detect invalid bounding boxes.
        """
    
        widths = (
    
            outputs.boxes[:, 2]
    
            -
    
            outputs.boxes[:, 0]
    
        )
    
        heights = (
    
            outputs.boxes[:, 3]
    
            -
    
            outputs.boxes[:, 1]
    
        )
    
        invalid = (
    
            (widths <= 0)
    
            |
    
            (heights <= 0)
    
        )
    
        return int(
            invalid.sum().item()
        )

    def classify(
        self,
        failures: int,
    ) -> str:
        """
        Classify system state.
        """
    
        if failures == 0:
    
            return "safe"
    
        if failures <= 3:
    
            return "warning"
    
        return "unsafe"

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Configuration summary.
        """
    
        return {
    
            "confidence_threshold": self.confidence_threshold,
    
            "uncertainty_threshold": self.uncertainty_threshold,
    
        }