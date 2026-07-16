from __future__ import annotations

from app.risk.confidence_estimator import (
    ConfidenceEstimator,
)

from app.risk.decision_engine import (
    DecisionEngine,
)

from app.risk.failure_detector import (
    FailureDetector,
)

from app.risk.risk_score import (
    RiskScore,
)

from app.risk.uncertainty import (
    UncertaintyEstimator,
)

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)


class RiskAnalyzer:
    """
    Complete perception risk analysis pipeline.
    """
    def __init__(
        self,
        confidence_estimator: ConfidenceEstimator | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        failure_detector: FailureDetector | None = None,
        risk_score: RiskScore | None = None,
        decision_engine: DecisionEngine | None = None,
    ) -> None:
    
        self.confidence_estimator = (
    
            confidence_estimator
    
            if confidence_estimator is not None
    
            else ConfidenceEstimator()
    
        )
    
        self.uncertainty_estimator = (
    
            uncertainty_estimator
    
            if uncertainty_estimator is not None
    
            else UncertaintyEstimator()
    
        )
    
        self.failure_detector = (
    
            failure_detector
    
            if failure_detector is not None
    
            else FailureDetector()
    
        )
    
        self.risk_score = (
    
            risk_score
    
            if risk_score is not None
    
            else RiskScore()
    
        )
    
        self.decision_engine = (
    
            decision_engine
    
            if decision_engine is not None
    
            else DecisionEngine(
                risk_model=self.risk_score,
            )
    
        )
    
        logger.info(
            "RiskAnalyzer initialized."
        )


    def analyze(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, object]:
        """
        Run complete risk analysis.
        """
    
        confidence = self.confidence_estimator(
            outputs,
        )
    
        uncertainty = self.uncertainty_estimator(
            outputs,
        )
    
        failures = self.failure_detector(
            outputs,
        )
    
        risk = self.risk_score(
            outputs,
        )
    
        decision = self.decision_engine(
            outputs,
        )
    
        return {
    
            "confidence": confidence,
    
            "uncertainty": uncertainty,
    
            "failures": failures,
    
            "risk": risk,
    
            "decision": decision,
    
        }

    def analyze_batch(
        self,
        outputs: list[
            DetectionOutput,
        ],
    ) -> list[
        dict[str, object],
    ]:
        """
        Analyze multiple predictions.
        """
    
        return [
    
            self.analyze(
                output,
            )
    
            for output in outputs
    
        ]

    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, object]:
        """
        Callable interface.
        """
    
        return self.analyze(
            outputs,
        )


    def summary(
        self,
    ) -> None:
        """
        Display pipeline information.
        """
    
        title(
            "Risk Analysis Pipeline",
        )
    
        logger.info(
    
            "ConfidenceEstimator : %s",
    
            type(
                self.confidence_estimator,
            ).__name__,
    
        )
    
        logger.info(
    
            "UncertaintyEstimator : %s",
    
            type(
                self.uncertainty_estimator,
            ).__name__,
    
        )
    
        logger.info(
    
            "FailureDetector : %s",
    
            type(
                self.failure_detector,
            ).__name__,
    
        )
    
        logger.info(
    
            "RiskScore : %s",
    
            type(
                self.risk_score,
            ).__name__,
    
        )
    
        logger.info(
    
            "DecisionEngine : %s",
    
            type(
                self.decision_engine,
            ).__name__,
    
        )
    
        success(
            "Risk pipeline ready."
        )