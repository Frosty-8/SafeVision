"""
SafeVision AI

Detection Head

Combines all prediction heads into
a single detector output.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.models.detector.bbox_head import (
    BoundingBoxHead,
)

from app.models.detector.classification_head import (
    ClassificationHead,
)

from app.models.detector.confidence_head import (
    ConfidenceHead,
)

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger


class DetectionHead(nn.Module):
    """
    Combines all prediction heads.
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 91,
    ) -> None:
    
        super().__init__()
    
        self.classification = ClassificationHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )
    
        self.box_regression = BoundingBoxHead(
            embedding_dim=embedding_dim,
        )
    
        self.confidence = ConfidenceHead(
            embedding_dim=embedding_dim,
        )
    
        logger.info(
            "DetectionHead initialized."
        )

    def forward(
        self,
        queries: torch.Tensor,
    ) -> DetectionOutput:
        """
        Predict object detections.
        """
    
        class_logits = self.classification(
            queries,
        )
    
        boxes = self.box_regression(
            queries,
        )
    
        confidence = self.confidence(
            queries,
        )
    
        return DetectionOutput(
            class_logits=class_logits,
            boxes=boxes,
            confidence=confidence,
        )

    @property
    def num_classes(
        self,
    ) -> int:
        """
        Number of predicted classes.
        """
    
        return self.classification.output_dim

    @property
    def box_dimension(
        self,
    ) -> int:
        """
        Bounding box dimension.
        """
    
        return self.box_regression.output_dim