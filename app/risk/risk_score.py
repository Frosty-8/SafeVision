from __future__ import annotations

from app.risk.confidence_estimator import (
    ConfidenceEstimator,
)

from app.risk.failure_detector import (
    FailureDetector,
)

from app.risk.uncertainty import (
    UncertaintyEstimator,
)

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger
from typing import cast

class RiskScore:
    """
    Computes ab overall perception risk score.
    """
    def __init__(
        self,
        confidence_weight: float = 0.40,
        uncertainty_weight: float = 0.35,
        failure_weight: float = 0.25,
    ) -> None:
    
        total = (
    
            confidence_weight
    
            +
    
            uncertainty_weight
    
            +
    
            failure_weight
    
        )
    
        if abs(total - 1.0) > 1e-6:
    
            raise ValueError(
                "Risk weights must sum to 1."
            )
    
        self.confidence_weight = confidence_weight
    
        self.uncertainty_weight = uncertainty_weight
    
        self.failure_weight = failure_weight
    
        self.confidence_estimator = (
            ConfidenceEstimator()
        )
    
        self.uncertainty_estimator = (
            UncertaintyEstimator()
        )
    
        self.failure_detector = (
            FailureDetector()
        )
    
        logger.info(
            "RiskScore initialized."
        )


    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, float | str]:
        """
        Callable interface.
        """
    
        return self.compute(
            outputs,
        )

    def compute(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, float | str]:
        """
        Compute overall risk.
        """
    
        confidence = (
            self.confidence_estimator(
                outputs,
            )
        )
    
        uncertainty = (
            self.uncertainty_estimator(
                outputs,
            )
        )
    
        failures = (
            self.failure_detector(
                outputs,
            )
        )
    
        confidence_score = (
            self.confidence_component(
                confidence,
            )
        )
    
        uncertainty_score = (
            self.uncertainty_component(
                uncertainty,
            )
        )
    
        failure_score = (
            self.failure_component(
                failures,
            )
        )
    
        score = (
    
            self.confidence_weight
    
            * confidence_score
    
            +
    
            self.uncertainty_weight
    
            * uncertainty_score
    
            +
    
            self.failure_weight
    
            * failure_score
    
        )
    
        score = self.normalize(
            score,
        )
    
        return {
    
            "score": score,
    
            "level": self.level(
                score,
            ),
    
        }


    def confidence_component(
        self,
        confidence: dict[str, float],
    ) -> float:
        """
        Confidence contribution.
        """
    
        return 1.0 - confidence["mean"]


    def uncertainty_component(
        self,
        uncertainty: dict[str, float],
    ) -> float:
        """
        Uncertainty contribution.
        """
    
        return uncertainty["total"]


    def failure_component(
        self,
        failures: dict[str, object],
    ) -> float:
        """
        Failure contribution.
        """
        total_failures = cast(int, failures["total_failures"])
        return min(total_failures / 10.0,1.0,
    
        )

    def normalize(
        self,
        score: float,
    ) -> float:
        """
        Clamp score.
        """
    
        return max(
    
            0.0,
    
            min(
    
                score,
    
                1.0,
    
            ),
    
        )

    def level(
        self,
        score: float,
    ) -> str:
        """
        Risk category.
        """
    
        if score < 0.30:
    
            return "low"
    
        if score < 0.60:
    
            return "medium"
    
        if score < 0.80:
    
            return "high"
    
        return "critical"


    def summary(
        self,
    ) -> dict[str, float]:
        """
        Configuration.
        """
    
        return {
    
            "confidence_weight": self.confidence_weight,
    
            "uncertainty_weight": self.uncertainty_weight,
    
            "failure_weight": self.failure_weight,
    
        }