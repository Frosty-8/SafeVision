from __future__ import annotations

import torch

from app.metrics.iou import IoUMetric

from app.utils.logger import logger


class PrecisionMetric:
    """
    Computes detection precision.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
    ) -> None:

        self.iou_threshold = iou_threshold

        self.iou = IoUMetric()

        self.reset()

        logger.info("PrecisionMetric initialized.")

    def reset(
        self,
    ) -> None:
        """
        Reset statistics.
        """

        self.true_positives = 0

        self.false_positives = 0

    def compute(
        self,
        predicted_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> float:
        """
        Compute precision for one batch.
        """

        def compute(
            self,
            predicted_boxes: torch.Tensor,
            target_boxes: torch.Tensor,
        ) -> float:
            """
            Compute precision for one batch.
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

        self.true_positives += true_positive

        self.false_positives += false_positive

        denominator = self.true_positives + self.false_positives

        if denominator == 0:

            return 0.0

        return self.true_positives / denominator

    def batch_compute(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> float:
        """
        Compute precision across batches.
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
        Average precision.
        """

        denominator = self.true_positives + self.false_positives

        if denominator == 0:

            return 0.0

        return self.true_positives / denominator

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Precision summary.
        """

        return {
            "precision": self.average(),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
        }
