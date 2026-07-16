from __future__ import annotations

import torch

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger

class ConfidenceCalibrator:
    """
    Confidence calibration using temperature scaling.
    """
    def __init__(
        self,
        temperature: float = 1.0,
    ) -> None:
    
        if temperature <= 0:
    
            raise ValueError(
                "Temperature must be positive."
            )
    
        self.temperature = temperature
    
        logger.info(
            "ConfidenceCalibrator initialized."
        )


    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Calibrate prediction confidence.
        """
    
        return self.calibrate(
            outputs,
        )


    def calibrate(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Apply confidence calibration.
        """
    
        confidence = self.temperature_scaling(
    
            outputs.confidence,
    
        )
    
        confidence = self.normalize(
            confidence,
        )
    
        return DetectionOutput(
    
            boxes=outputs.boxes,
    
            class_logits=outputs.class_logits,
    
            confidence=confidence,
    
            labels=outputs.labels,
    
            queries=outputs.queries,
    
        )

    def temperature_scaling(
        self,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply temperature scaling.
        """
    
        confidence = confidence.clamp(
    
            min=1e-6,
    
            max=1.0 - 1e-6,
    
        )
    
        logits = torch.log(
    
            confidence
    
            /
    
            (1.0 - confidence)
    
        )
    
        logits = logits / self.temperature
    
        return torch.sigmoid(
            logits,
        )

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

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Return calibration settings.
        """
    
        return {
    
            "temperature": self.temperature,
    
        }