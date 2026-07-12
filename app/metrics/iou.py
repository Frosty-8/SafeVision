from __future__ import annotations

import torch

from app.models.detector.losses import (
    box_cxcywh_to_xyxy,
)

from app.utils.logger import logger

class IoUMetric:
    """
    Computes IoU statistics.
    """
    def __init__(self) -> None:
        self.reset()

        logger.info(
            "IoUMetric initialized."
        )

    def reset(
        self,
    ) -> None:
        """
        Reset accumulated statistics.
        """
    
        self.total_iou = 0.0
    
        self.num_boxes = 0

    def compute(
        self,
        predicted_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU for matched boxes.
    
        Parameters
        ----------
        predicted_boxes
            (N,4) in cxcywh format.
    
        target_boxes
            (N,4) in cxcywh format.
        """
        predicted_boxes = box_cxcywh_to_xyxy(
            predicted_boxes,
        )
        
        target_boxes = box_cxcywh_to_xyxy(
            target_boxes,
        )

        top_left = torch.maximum(
            predicted_boxes[:, :2],
            target_boxes[:, :2],
        )
        
        bottom_right = torch.minimum(
            predicted_boxes[:, 2:],
            target_boxes[:, 2:],
        )
        
        wh = (
            bottom_right
            - top_left
        ).clamp(
            min=0,
        )
        
        intersection = wh[:, 0] * wh[:, 1]

        pred_area = (
            predicted_boxes[:, 2]
            - predicted_boxes[:, 0]
        ) * (
            predicted_boxes[:, 3]
            - predicted_boxes[:, 1]
        )
        
        target_area = (
            target_boxes[:, 2]
            - target_boxes[:, 0]
        ) * (
            target_boxes[:, 3]
            - target_boxes[:, 1]
        )

        union = (
            pred_area
            + target_area
            - intersection
        ).clamp(
            min=1e-6,
        )

        iou = intersection / union
        
        self.total_iou += float(
            iou.sum().item()
        )
        
        self.num_boxes += len(
            iou,
        )
        
        return iou

    def batch_compute(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute IoU over multiple batches.
        """
    
        values = []
    
        for prediction, target in zip(
            predictions,
            targets,
            strict=True,
        ):
    
            values.append(
                self.compute(
                    prediction,
                    target,
                )
            )
    
        return torch.cat(
            values,
        )

    def average(
        self,
    ) -> float:
        """
        Average IoU.
        """
    
        if self.num_boxes == 0:
    
            return 0.0
    
        return (
            self.total_iou
            / self.num_boxes
        )

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Metric summary.
        """
    
        return {
    
            "average_iou": self.average(),
    
            "num_boxes": self.num_boxes,
    
        }