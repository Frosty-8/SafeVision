from __future__ import annotations

import json

from dataclasses import asdict
from pathlib import Path

from app.schemas.detection import (
    DetectionOutput,
    DetectionTarget,
)
import numpy as np
from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)
from scipy.optimize import linear_sum_assignment

class FailureAnalyzer:
    """
    Analyze detector failures.
    """
    def __init__(
        self,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> None:
    
        self.iou_threshold = iou_threshold
    
        self.confidence_threshold = confidence_threshold
    
        self.reset()
    
        logger.info(
            "FailureAnalyzer initialized."
        )

    def reset(
        self,
    ) -> None:
        """
        Reset statistics.
        """
    
        self.false_positives = 0
    
        self.false_negatives = 0
    
        self.localization_errors = 0
    
        self.classification_errors = 0
    
        self.low_confidence = 0
    
        self.duplicate_detections = 0

    def analyze(
        self,
        predictions: DetectionOutput,
        targets: DetectionTarget,
    ) -> None:
        """
        Analyze prediction failures.
        """
    
        self.reset()
    
        self._find_false_positives(
            predictions,
            targets,
        )
    
        self._find_false_negatives(
            predictions,
            targets,
        )
    
        self._find_localization_errors(
            predictions,
            targets,
        )
    
        self._find_classification_errors(
            predictions,
            targets,
        )
    
        self._find_low_confidence(
            predictions,
        )

        self._find_duplicate_detections(
            predictions,
            targets,
        )

    
    def _match(
        self,
        predictions: DetectionOutput,
        targets: DetectionTarget,
    ) -> tuple[
        list[tuple[int, int, float]],
        set[int],
        set[int],
    ]:
        """
        Match predictions with targets using the Hungarian algorithm.
        """
    
        num_predictions = predictions.boxes.shape[0]
        num_targets = targets.boxes.shape[0]
    
        if num_predictions == 0:
            return [], set(), set(range(num_targets))
    
        if num_targets == 0:
            return [], set(range(num_predictions)), set()
    
        iou_matrix = np.zeros(
            (num_predictions, num_targets),
            dtype=np.float32,
        )
    
        for i in range(num_predictions):
            for j in range(num_targets):
                iou_matrix[i, j] = self._compute_iou(
                    predictions.boxes[i],
                    targets.boxes[j],
                )
    
        cost_matrix = 1.0 - iou_matrix
    
        pred_indices, target_indices = linear_sum_assignment(
            cost_matrix
        )
    
        matches = []
        matched_predictions = set()
        matched_targets = set()
    
        for pred_idx, target_idx in zip(
            pred_indices,
            target_indices,
        ):
            iou = float(iou_matrix[pred_idx, target_idx])
    
            if iou >= self.iou_threshold:
                matches.append(
                    (
                        pred_idx,
                        target_idx,
                        iou,
                    )
                )
    
                matched_predictions.add(pred_idx)
                matched_targets.add(target_idx)
    
        unmatched_predictions = (
            set(range(num_predictions))
            - matched_predictions
        )
    
        unmatched_targets = (
            set(range(num_targets))
            - matched_targets
        )
    
        return (
            matches,
            unmatched_predictions,
            unmatched_targets,
        )

    def _find_low_confidence(
        self,
        predictions: DetectionOutput,
    ) -> None:
        """
        Count detections below confidence threshold.
        """
    
        self.low_confidence = int(
            (
                predictions.confidence
                <
                self.confidence_threshold
            ).sum().item()
        )

    def _find_false_positives(
        self,
        predictions: DetectionOutput,
        targets: DetectionTarget,
    ) -> None:
        """
        Count unmatched predictions.
        """
    
        _, unmatched_predictions, _ = self._match(
            predictions,
            targets,
        )
    
        self.false_positives = len(
            unmatched_predictions
        )


    def _find_false_negatives(
        self,
        predictions: DetectionOutput,
        targets: DetectionTarget,
    ) -> None:
        """
        Count unmatched targets.
        """
    
        _, _, unmatched_targets = self._match(
            predictions,
            targets,
        )
    
        self.false_negatives = len(
            unmatched_targets
        )

    def _find_localization_errors(
        self,
        predictions: DetectionOutput,
        targets: DetectionTarget,
    ) -> None:
        """
        Correct class but IoU below threshold.
        """
    
        pred_labels = predictions.class_logits.argmax(dim=1)
    
        errors = 0
    
        for pred_idx in range(predictions.boxes.shape[0]):
    
            best_iou = 0.0
            best_target = -1
    
            for target_idx in range(targets.boxes.shape[0]):
    
                iou = self._compute_iou(
                    predictions.boxes[pred_idx],
                    targets.boxes[target_idx],
                )
    
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_idx
    
            if (
                best_target >= 0
                and
                pred_labels[pred_idx].item()
                ==
                targets.labels[best_target].item()
                and
                0.0 < best_iou < self.iou_threshold
            ):
                errors += 1
    
        self.localization_errors = errors

    def _find_classification_errors(
        self,
        predictions: DetectionOutput,
        targets: DetectionTarget,
    ) -> None:
        """
        Correct localization but wrong class.
        """
    
        pred_labels = predictions.class_logits.argmax(dim=1)
    
        matches, _, _ = self._match(
            predictions,
            targets,
        )
    
        errors = 0
    
        for pred_idx, target_idx, _ in matches:
    
            if (
                pred_labels[pred_idx].item()
                !=
                targets.labels[target_idx].item()
            ):
                errors += 1
    
        self.classification_errors = errors

    def _find_duplicate_detections(
        self,
        predictions: DetectionOutput,
        targets: DetectionTarget,
    ) -> None:
        """
        Count duplicate detections.
        """
    
        duplicates = 0
    
        for target_idx in range(targets.boxes.shape[0]):
    
            matched = 0
    
            for pred_idx in range(predictions.boxes.shape[0]):
    
                iou = self._compute_iou(
                    predictions.boxes[pred_idx],
                    targets.boxes[target_idx],
                )
    
                if iou >= self.iou_threshold:
                    matched += 1
    
            if matched > 1:
                duplicates += matched - 1
    
        self.duplicate_detections = duplicates

    def _compute_iou(
        self,
        box1,
        box2,
    ) -> float:
        """
        Compute IoU between two boxes.
        """
    
        box1 = box1.tolist()
        box2 = box2.tolist()
    
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
    
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
    
        intersection = inter_w * inter_h
    
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        union = area1 + area2 - intersection
    
        if union <= 0:
            return 0.0
    
        return intersection / union