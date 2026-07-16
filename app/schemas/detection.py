"""
SafeVision AI

Detection Schemas

Schemas used throughout the object
detection pipeline.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DetectionOutput:
    """
    Output produced by the detector.
    """

    class_logits: torch.Tensor

    boxes: torch.Tensor

    confidence: torch.Tensor

    queries: torch.Tensor | None = None

    labels: torch.Tensor | None = None


@dataclass(slots=True)
class DetectionResult:
    """
    Single decoded detection.
    """

    class_id: int

    class_name: str

    confidence: float

    bbox: tuple[
        float,
        float,
        float,
        float,
    ]


@dataclass(slots=True)
class DetectionTarget:

    labels: torch.Tensor

    boxes: torch.Tensor


@dataclass(slots=True)
class LossOutput:

    total: torch.Tensor

    classification: torch.Tensor

    bbox: torch.Tensor

    giou: torch.Tensor

    confidence: torch.Tensor


@dataclass(slots=True)
class MatchedBatch:

    logits: torch.Tensor

    boxes: torch.Tensor

    labels: torch.Tensor

    target_boxes: torch.Tensor


@dataclass(slots=True)
class EvaluationReport:
    """
    Complete detector evaluation.
    """

    loss: float

    map: float

    precision: float

    recall: float

    f1_score: float

    average_iou: float

    fps: float

    latency_ms: float

    num_images: int

    num_detections: int
