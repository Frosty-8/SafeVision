"""
SafeVision AI

XGBoost Schemas

Schemas used by the XGBoost risk prediction pipeline.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum


class CalibrationMethod(str, Enum):
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


class RiskDecision(str, Enum):

    SAFE = "safe"

    UNSAFE = "unsafe"


@dataclass(slots=True)
class DetectionFeatures:
    """
    Features extracted from a single detection
    for XGBoost risk prediction.
    """

    confidence: float

    box_width: float

    box_height: float

    box_area: float

    aspect_ratio: float

    object_class: int

    image_brightness: float

    occlusion_score: float

    attention_score: float

    entropy: float

    distance: float | None = None

    velocity: float | None = None

    weather_score: float | None = None

    false_negative: bool = False


@dataclass(slots=True)
class XGBoostTrainingReport:
    """
    Stores training statistics.
    """

    accuracy: float

    precision: float

    recall: float

    f1_score: float

    roc_auc: float

    training_time: float

    num_samples: int

    feature_importance: dict[
        str,
        float,
    ] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class RiskPrediction:
    """
    Prediction produced by the
    XGBoost model.
    """

    probability: float

    risk_score: float

    decision: str
