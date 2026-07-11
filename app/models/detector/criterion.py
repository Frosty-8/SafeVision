from __future__ import annotations

import torch
import torch.nn as nn

from app.models.detector.losses import (
    BoundingBoxLoss,
    ClassificationLoss,
    ConfidenceLoss,
    GIoULoss,
)

from app.models.detector.matcher import (
    HungarianMatcher,
)

from app.schemas.detection import (
    DetectionOutput,
    DetectionTarget,
    LossOutput,
)

from app.utils.logger import logger


class SetCriterion(nn.Module):
    """
    Computes the complete detector loss.
    """

    def __init__(
        self,
        class_weight: float = 2.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        confidence_weight: float = 1.0,
    ) -> None:

        super().__init__()

        self.matcher = HungarianMatcher()

        self.classification_loss = ClassificationLoss()

        self.bbox_loss = BoundingBoxLoss()

        self.giou_loss = GIoULoss()

        self.confidence_loss = ConfidenceLoss()

        self.class_weight = class_weight

        self.bbox_weight = bbox_weight

        self.giou_weight = giou_weight

        self.confidence_weight = confidence_weight

        logger.info("SetCriterion initialized.")

    def forward(
        self,
        predictions: DetectionOutput,
        targets: list[DetectionTarget],
    ) -> LossOutput:
        matches = self.matcher(
            predictions,
            targets,
        )

        classification = self.classification_loss(
            predictions,
            targets,
            matches,
        )

        bbox = self.bbox_loss(
            predictions,
            targets,
            matches,
        )

        giou = self.giou_loss(
            predictions,
            targets,
            matches,
        )

        confidence = self.confidence_loss(
            predictions,
            targets,
            matches,
        )

        total = (
            self.class_weight * classification
            + self.bbox_weight * bbox
            + self.giou_weight * giou
            + self.confidence_weight * confidence
        )

        return LossOutput(
            total=total,
            classification=classification,
            bbox=bbox,
            giou=giou,
            confidence=confidence,
        )

    @property
    def weights(
        self,
    ) -> dict[str, float]:
        """
        Return configured loss weights.
        """

        return {
            "classification": self.class_weight,
            "bbox": self.bbox_weight,
            "giou": self.giou_weight,
            "confidence": self.confidence_weight,
        }
