from __future__ import annotations

import torch

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger

class UncertaintyEstimator:
    """
    Estimates prediction uncertainty
    """
    def __init__(
        self,
        epsilon: float = 1e-6,
    ) -> None:
    
        self.epsilon = epsilon
    
        logger.info(
            "UncertaintyEstimator initialized."
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
        Estimate uncertainty statistics.
        """
    
        confidence = outputs.confidence
    
        if confidence.numel() == 0:
    
            return {
    
                "epistemic": 0.0,
    
                "aleatoric": 0.0,
    
                "entropy": 0.0,
    
                "total": 0.0,
    
            }
    
        epistemic = self.epistemic(
            confidence,
        )
    
        aleatoric = self.aleatoric(
            confidence,
        )
    
        entropy = self.predictive_entropy(
            confidence,
        )
    
        total = self.total_uncertainty(
    
            epistemic,
    
            aleatoric,
    
            entropy,
    
        )
    
        return {
    
            "epistemic": epistemic,
    
            "aleatoric": aleatoric,
    
            "entropy": entropy,
    
            "total": total,
    
        }

    def epistemic(
        self,
        confidence: torch.Tensor,
    ) -> float:
        """
        Approximate model uncertainty.
        """
    
        return float(
            confidence.var().item()
        )

    def aleatoric(
        self,
        confidence: torch.Tensor,
    ) -> float:
        """
        Estimate data uncertainty.
        """
    
        return float(
    
            (
    
                confidence
    
                * (1.0 - confidence)
    
            )
    
            .mean()
    
            .item()
    
        )


    def predictive_entropy(
        self,
        confidence: torch.Tensor,
    ) -> float:
        """
        Compute predictive entropy.
        """
    
        confidence = confidence.clamp(
    
            self.epsilon,
    
            1.0 - self.epsilon,
    
        )
    
        entropy = -(
    
            confidence
    
            * torch.log(confidence)
    
            +
    
            (1.0 - confidence)
    
            * torch.log(
    
                1.0 - confidence,
    
            )
    
        )
    
        return float(
    
            entropy.mean().item()
    
        )

    def total_uncertainty(
        self,
        epistemic: float,
        aleatoric: float,
        entropy: float,
    ) -> float:
        """
        Aggregate uncertainty.
        """
    
        return (
    
            0.4 * epistemic
    
            +
    
            0.3 * aleatoric
    
            +
    
            0.3 * entropy
    
        )

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Configuration.
        """
    
        return {
    
            "epsilon": self.epsilon,
    
        }