from __future__ import annotations

import torch

from app.schemas.detection import DetectionOutput

from app.utils.logger import logger

class PostProcessor:
    """
    Cleans detector predictions.
    """
    def __init__(
        self,
        score_threshold: float = 0.25,
    ) -> None:
    
        self.score_threshold = score_threshold
    
        logger.info(
            "PostProcessor initialized."
        )


    def __call__(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Complete post-processing pipeline.
        """
    
        outputs = self.filter_scores(
            outputs,
        )
    
        outputs = self.convert_boxes(
            outputs,
        )
    
        outputs = self.clip_boxes(
            outputs,
        )
    
        outputs = self.remove_invalid_boxes(
            outputs,
        )
    
        outputs = self.prepare_for_nms(
            outputs,
        )
    
        return self.finalize(
            outputs,
        )

    def filter_scores(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Remove detections below the confidence threshold.
        """
    
        keep = outputs.confidence >= self.score_threshold
    
        return DetectionOutput(
            class_logits=outputs.class_logits[keep],
            boxes=outputs.boxes[keep],
            confidence=outputs.confidence[keep],
            labels=(
                outputs.labels[keep]
                if outputs.labels is not None
                else None
            ),
            queries=(
                outputs.queries[keep]
                if outputs.queries is not None
                else None
            ),
        )

    def convert_boxes(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Convert boxes from cxcywh to xyxy.
        """
    
        cx, cy, w, h = outputs.boxes.unbind(dim=-1)
    
        boxes = torch.stack(
            (
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2,
            ),
            dim=-1,
        )
    
        return DetectionOutput(
            class_logits=outputs.class_logits,
            boxes=boxes,
            confidence=outputs.confidence,
            labels=outputs.labels,
            queries=outputs.queries,
        )

    def clip_boxes(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Clip boxes to normalized coordinates.
        """
    
        return DetectionOutput(
            class_logits=outputs.class_logits,
            boxes=outputs.boxes.clamp(0.0, 1.0),
            confidence=outputs.confidence,
            labels=outputs.labels,
            queries=outputs.queries,
        )


    def remove_invalid_boxes(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Remove zero-area boxes.
        """
    
        boxes = outputs.boxes
    
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    
        keep = (widths > 0) & (heights > 0)
    
        return DetectionOutput(
            class_logits=outputs.class_logits[keep],
            boxes=boxes[keep],
            confidence=outputs.confidence[keep],
            labels=(
                outputs.labels[keep]
                if outputs.labels is not None
                else None
            ),
            queries=(
                outputs.queries[keep]
                if outputs.queries is not None
                else None
            ),
        )

    def prepare_for_nms(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Decode predicted labels.
        """
    
        labels = outputs.class_logits.argmax(dim=-1)
    
        return DetectionOutput(
            class_logits=outputs.class_logits,
            boxes=outputs.boxes,
            confidence=outputs.confidence,
            labels=labels,
            queries=outputs.queries,
        )

    def finalize(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Final validation before NMS.
        """
    
        return DetectionOutput(
            class_logits=outputs.class_logits,
            boxes=outputs.boxes,
            confidence=outputs.confidence,
            labels=outputs.labels,
            queries=outputs.queries,
        )

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Configuration summary.
        """
    
        return {
    
            "score_threshold": self.score_threshold,
    
        }