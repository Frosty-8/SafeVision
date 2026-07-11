from __future__ import annotations

from typing import Any

import torch

Target = dict[str, Any]
Batch = list[tuple[torch.Tensor, Target]]


def detection_collate_fn(
    batch: Batch,
) -> tuple[
    torch.Tensor,
    list[Target],
]:
    """
    Collate function for object detection.

    Parameters
    ----------
    batch:
        List of dataset samples.

    Returns
    -------
    images:
        Tensor[B,C,H,W]

    targets:
        List of dictionaries
    """

    images = []

    targets = []

    for image, target in batch:

        images.append(image)

        targets.append(target)

    images = torch.stack(
        images,
        dim=0,
    )

    return images, targets
