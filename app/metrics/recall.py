from __future__ import annotations

import torch

from app.metrics.iou import IoUMetric

from app.utils.logger import logger


class RecallMetric:
    """
    Computes detector recall.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
    ) -> None:

        self.iou_threshold = iou_threshold

        self.iou = IoUMetric()

        self.reset()

        logger.info("RecallMetric initialized.")

    def reset(
        self,
    ) -> None:
        """
        Reset metric statistics.
        """

        self.true_positives = 0

        self.false_negatives = 0

    def compute(
        self,
        predicted_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> float:
        """
        Compute recall for one batch.
        """
        ious = self.iou.compute(
            predicted_boxes,
            target_boxes,
        )

        true_positive = int((ious >= self.iou_threshold).sum().item())

        false_negative = max(
            len(target_boxes) - true_positive,
            0,
        )

        self.true_positives += true_positive

        self.false_negatives += false_negative

        denominator = self.true_positives + self.false_negatives

        if denominator == 0:

            return 0.0

        return self.true_positives / denominator

    def batch_compute(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> float:
        """
        Compute recall across batches.
        """

        self.reset()

        for prediction, target in zip(
            predictions,
            targets,
            strict=True,
        ):

            self.compute(
                prediction,
                target,
            )

        return self.average()

    def average(
        self,
    ) -> float:
        """
        Average recall.
        """

        denominator = self.true_positives + self.false_negatives

        if denominator == 0:

            return 0.0

        return self.true_positives / denominator

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Recall summary.
        """

        return {
            "recall": self.average(),
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
        }
