from __future__ import annotations

import torch

from app.metrics.iou import IoUMetric

from app.utils.logger import logger


class ConfusionMatrix:
    """
    Detection confusion matrix.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
    ) -> None:

        self.iou_threshold = iou_threshold

        self.iou = IoUMetric()

        self.reset()

        logger.info("ConfusionMatrix initialized.")

    def reset(
        self,
    ) -> None:
        """
        Reset statistics.
        """

        self.true_positive = 0

        self.false_positive = 0

        self.false_negative = 0

        self.true_negative = 0

    def update(
        self,
        predicted_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> None:
        """
        Update confusion statistics.
        """
        ious = self.iou.compute(
            predicted_boxes,
            target_boxes,
        )

        true_positive = int((ious >= self.iou_threshold).sum().item())

        false_positive = max(
            len(predicted_boxes) - true_positive,
            0,
        )

        false_negative = max(
            len(target_boxes) - true_positive,
            0,
        )

        self.true_positive += true_positive

        self.false_positive += false_positive

        self.false_negative += false_negative

    def matrix(
        self,
    ) -> torch.Tensor:
        """
        Return confusion matrix.
        """

        return torch.tensor(
            [
                [
                    self.true_positive,
                    self.false_positive,
                ],
                [
                    self.false_negative,
                    self.true_negative,
                ],
            ],
            dtype=torch.int64,
        )

    def summary(
        self,
    ) -> dict[str, int]:
        """
        Confusion statistics.
        """

        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "tn": self.true_negative,
        }
