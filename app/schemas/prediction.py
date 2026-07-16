from dataclasses import dataclass
import torch

@dataclass(slots=True)
class PredictionResult:

    boxes: torch.Tensor

    labels: torch.Tensor

    scores: torch.Tensor

    inference_time: float

    image_size: tuple[int, int]