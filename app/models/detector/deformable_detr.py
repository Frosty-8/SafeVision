"""
SafeVision AI

Deformable DETR

Complete detector model.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.models.detector.criterion import (
    SetCriterion,
)

from app.models.detector.detection_head import (
    DetectionHead,
)

from app.models.detector.transformer import (
    DeformableTransformer,
)

from app.schemas.detection import (
    DetectionOutput,
    DetectionTarget,
    LossOutput,
)

from app.utils.logger import logger


class DeformableDETR(nn.Module):
    """
    SafeVision object detector.
    """

    def __init__(
        self,
        input_channels: int = 256,
        embedding_dim: int = 256,
        num_classes: int = 91,
        num_queries: int = 300,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        criterion: SetCriterion | None = None,
    ) -> None:

        super().__init__()

        self.transformer = DeformableTransformer(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_points=num_points,
            num_queries=num_queries,
        )

        self.detection_head = DetectionHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

        self.criterion = criterion if criterion is not None else SetCriterion()

        logger.info("DeformableDETR initialized.")

    def forward(
        self,
        features: torch.Tensor,
        targets: list[DetectionTarget] | None = None,
    ) -> DetectionOutput | LossOutput:
        queries = self.transformer(features)

        detections = self.detection_head(queries)

        if self.training and targets is not None:

            return self.criterion(
                detections,
                targets,
            )

        return detections

    @property
    def num_queries(
        self,
    ) -> int:

        return self.transformer.num_queries

    @property
    def embedding_dim(
        self,
    ) -> int:

        return self.transformer.output_dim

    def freeze_backbone(
        self,
    ) -> None:
        """
        Freeze backbone parameters.
        """

        for parameter in self.transformer.backbone.parameters():

            parameter.requires_grad = False

    def unfreeze_backbone(
        self,
    ) -> None:
        """
        Unfreeze backbone parameters.
        """

        for parameter in self.transformer.backbone.parameters():

            parameter.requires_grad = True

    @property
    def num_parameters(
        self,
    ) -> int:

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameters(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def summary(
        self,
    ) -> dict[str, int]:

        return {
            "parameters": self.num_parameters,
            "trainable": self.trainable_parameters,
            "queries": self.num_queries,
            "embedding_dim": self.embedding_dim,
        }
