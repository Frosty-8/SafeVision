from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float

    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map5095: float | None = None
    false_negative_rate: float | None = None

    extras: dict[str, float] = field(default_factory=dict)