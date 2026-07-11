"""
SafeVision AI

Hungarian Matcher

Matches predicted objects to ground-truth
objects using the Hungarian algorithm.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scipy.optimize import linear_sum_assignment

from app.schemas.detection import DetectionOutput, DetectionTarget

from app.utils.logger import logger
from app.models.detector.losses import (
    generalized_box_iou,
)


class HungarianMatcher(nn.Module):
    """
    Hungarian matching used during training.
    """

    def __init__(
        self,
        class_cost: float = 2.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
    ) -> None:

        super().__init__()

        self.class_cost = class_cost

        self.bbox_cost = bbox_cost

        self.giou_cost = giou_cost

        logger.info("HungarianMatcher initialized.")

    def _classification_cost(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute classification cost.
        """

        probabilities = logits.softmax(
            dim=-1,
        )

        return -probabilities[
            :,
            labels,
        ]

    def _bbox_cost(
        self,
        predicted_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        L1 box distance.
        """

        return torch.cdist(
            predicted_boxes,
            target_boxes,
            p=1,
        )

    def forward(
        self,
        predictions: DetectionOutput,
        targets: list[DetectionTarget],
    ):
        matches = []

        batch_size = predictions.boxes.shape[0]

        for batch_index in range(batch_size):
            pred_logits = predictions.class_logits[batch_index]
            pred_boxes = predictions.boxes[batch_index]

            target = targets[batch_index]

            labels = target.labels
            boxes = target.boxes

            class_cost = self._classification_cost(
                pred_logits,
                labels,
            )

            bbox_cost = self._bbox_cost(
                pred_boxes,
                boxes,
            )

            giou_cost = self._giou_cost(
                pred_boxes,
                boxes,
            )

            cost = (
                self.class_cost * class_cost
                + self.bbox_cost * bbox_cost
                + self.giou_cost * giou_cost
            )

            row_indices, column_indices = linear_sum_assignment(
                cost.detach().cpu(),
            )

            matches.append(
                (
                    torch.as_tensor(row_indices, dtype=torch.int64),
                    torch.as_tensor(column_indices, dtype=torch.int64),
                )
            )

        return matches

    def _giou_cost(
        self,
        predicted_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GIoU cost.
        """

        return -generalized_box_iou(
            predicted_boxes,
            target_boxes,
        )
