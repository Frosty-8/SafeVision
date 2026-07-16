from __future__ import annotations

import torch

from torchvision.ops import (
    batched_nms,
    nms,
)

from app.schemas.detection import (
    DetectionOutput,
)
from app.utils.logger import logger

class NMS:
    """
    Performs Non-Maximum Suppression.
    """
    def __init__(
        self,
        iou_threshold: float = 0.5,
    ) -> None:
    
        self.iou_threshold = iou_threshold
    
        logger.info(
            "NMS initialized."
        )

    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Default class-wise NMS.
        """
    
        return self.class_wise_nms(
            outputs,
        )

    def standard_nms(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Standard torchvision NMS.
        """
        if outputs.labels is None:
            raise ValueError(
                "DetectionOutput.labels must be populated before NMS."
            )
        
        labels = outputs.labels
    
        keep = nms(
    
            outputs.boxes,
    
            outputs.confidence,
    
            self.iou_threshold,
    
        )
    
        return DetectionOutput(
    
            boxes=outputs.boxes[keep],
    
            class_logits=outputs.class_logits[keep],
    
            confidence=outputs.confidence[keep],
    
            labels=labels[keep],
    
        )


    def class_wise_nms(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Perform NMS independently
        for every class.
        """
        if outputs.labels is None:
            raise ValueError(
                "DetectionOutput.labels must be populated before NMS."
            )
        
        labels = outputs.labels
    
        keep = batched_nms(
    
            outputs.boxes,
    
            outputs.confidence,
    
            labels,
    
            self.iou_threshold,
    
        )
    
        return DetectionOutput(
    
            boxes=outputs.boxes[keep],
    
            class_logits=outputs.class_logits[keep],
    
            confidence=outputs.confidence[keep],
    
            labels=labels[keep],
    
        )

    def batched(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Alias for batched NMS.
        """
    
        return self.class_wise_nms(
            outputs,
        )

    def soft_nms(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Soft-NMS.
    
        TODO:
        Implement Gaussian Soft-NMS.
        """
    
        logger.warning(
            "Soft-NMS not implemented. Falling back to standard NMS."
        )
    
        return self.standard_nms(
            outputs,
        )

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Return configuration.
        """
    
        return {
    
            "iou_threshold": self.iou_threshold,
    
        }