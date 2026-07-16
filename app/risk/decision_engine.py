from __future__ import annotations

from app.risk.risk_score import RiskScore

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger

class DecisionEngine:
    """
    Makes safety decision based on overall perception risk.
    """
    def __init__(
        self,
        risk_model: RiskScore | None = None,
    ) -> None:
    
        self.risk_model = (
    
            risk_model
    
            if risk_model is not None
    
            else RiskScore()
    
        )
    
        logger.info(
            "DecisionEngine initialized."
        )

    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, object]:
    
        return self.decide(
            outputs,
        )

    def decide(
        self,
        outputs: DetectionOutput,
    ) -> dict[str, object]:
        """
        Compute operational decision.
        """
    
        assessment = self.risk_model(
            outputs,
        )
    
        score = assessment["score"]
    
        level = assessment["level"]
    
        return {
    
            "risk_score": score,
    
            "risk_level": level,
    
            "decision": self.safety_action(
                level,
            ),
    
            "recommended_speed": self.recommended_speed(
                level,
            ),
    
            "warning": self.warning_message(
                level,
            ),
    
            "recommendation": self.recommendation(
                level,
            ),
    
        }


    def safety_action(
        self,
        level: str,
    ) -> str:
        """
        Primary system action.
        """
    
        actions = {
    
            "low": "CONTINUE",
    
            "medium": "PROCEED_WITH_CAUTION",
    
            "high": "REDUCE_SPEED",
    
            "critical": "EMERGENCY_STOP",
    
        }
    
        return actions.get(
    
            level,
    
            "UNKNOWN",
    
        )

    def warning_message(
        self,
        level: str,
    ) -> str:
        """
        Human-readable warning.
        """
    
        messages = {
    
            "low":
    
                "Environment considered safe.",
    
            "medium":
    
                "Proceed carefully.",
    
            "high":
    
                "High perception risk detected.",
    
            "critical":
    
                "Unsafe environment detected.",
    
        }
    
        return messages.get(
    
            level,
    
            "Unknown state.",
    
        )


    def recommended_speed(
        self,
        level: str,
    ) -> int:
        """
        Recommended operating speed.
        """
    
        speeds = {
    
            "low": 100,
    
            "medium": 60,
    
            "high": 30,
    
            "critical": 0,
    
        }
    
        return speeds.get(
    
            level,
    
            0,
    
        )

    def recommendation(
        self,
        level: str,
    ) -> str:
        """
        Additional recommendation.
        """
    
        recommendations = {
    
            "low":
    
                "Continue monitoring.",
    
            "medium":
    
                "Increase sensor awareness.",
    
            "high":
    
                "Reduce speed and verify detections.",
    
            "critical":
    
                "Stop vehicle and request operator intervention.",
    
        }
    
        return recommendations.get(
    
            level,
    
            "No recommendation.",
    
        )

    def summary(
        self,
    ) -> dict[str, str]:
        """
        Decision engine summary.
        """
    
        return {
    
            "risk_model":
    
                type(
    
                    self.risk_model,
    
                ).__name__,
    
        }