<!--from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.schemas.detection import (
    DetectionOutput,
    DetectionTarget,
)


def box_cxcywh_to_xyxy(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Convert normalized boxes.

    (cx,cy,w,h)

    →

    (x1,y1,x2,y2)
    """

    cx, cy, w, h = boxes.unbind(
        dim=-1,
    )

    return torch.stack(

        (

            cx - w / 2,

            cy - h / 2,

            cx + w / 2,

            cy + h / 2,

        ),

        dim=-1,

    )

def box_area(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute box area.
    """

    return (

        boxes[..., 2]
        - boxes[..., 0]

    ).clamp(
        min=0,
    ) * (

        boxes[..., 3]
        - boxes[..., 1]

    ).clamp(
        min=0,
    )


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise IoU matrix.
    """
    area1 = box_area(
        boxes1,
    )
    
    area2 = box_area(
        boxes2,
    )
    
    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[:, :2],
    )
    
    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[:, 2:],
    )
    
    wh = (
    
        rb
        - lt
    
    ).clamp(
        min=0,
    )
    
    intersection = wh[..., 0] * wh[..., 1]
    
    union = (
    
        area1[:, None]
    
        +
    
        area2
    
        -
    
        intersection
    
    )
    
    return intersection / (
        union + 1e-6
    )


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise Generalized IoU.
    """
    boxes1 = box_cxcywh_to_xyxy(
        boxes1,
    )
    
    boxes2 = box_cxcywh_to_xyxy(
        boxes2,
    )

    iou = box_iou(
        boxes1,
        boxes2,
    )


    lt = torch.minimum(
        boxes1[:, None, :2],
        boxes2[:, :2],
    )
    
    rb = torch.maximum(
        boxes1[:, None, 2:],
        boxes2[:, 2:],
    )
    
    wh = (
    
        rb
        - lt
    
    ).clamp(
        min=0,
    )
    
    area = wh[..., 0] * wh[..., 1]

    area1 = box_area(
        boxes1,
    )
    
    area2 = box_area(
        boxes2,
    )
    
    lt_i = torch.maximum(
        boxes1[:, None, :2],
        boxes2[:, :2],
    )
    
    rb_i = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[:, 2:],
    )
    
    wh_i = (
    
        rb_i
        - lt_i
    
    ).clamp(
        min=0,
    )
    
    intersection = wh_i[..., 0] * wh_i[..., 1]
    
    union = (
    
        area1[:, None]
    
        +
    
        area2
    
        -
    
        intersection
    
    )

    return iou - (
    
        area - union
    
    ) / (
    
        area + 1e-6
    
    )



class ClassificationLoss(
    nn.Module,
):

    def forward(
        self,
        predictions: DetectionOutput,
        targets: list[DetectionTarget],
        matches,
    ) -> torch.Tensor:

        #
        # Implement Hungarian matched
        # classification loss.
        #

        return F.cross_entropy(
            ...,
        )


class BoundingBoxLoss(
    nn.Module,
):

    def forward(
        self,
        predictions: DetectionOutput,
        targets: list[DetectionTarget],
        matches,
    ) -> torch.Tensor:

        return F.l1_loss(
            ...,
        )


class GIoULoss(
    nn.Module,
):

    def forward(
        self,
        predictions: DetectionOutput,
        targets: list[DetectionTarget],
        matches,
    ) -> torch.Tensor:

        giou = generalized_box_iou(
            ...,
        )

        return 1 - giou.mean()


class ConfidenceLoss(
    nn.Module,
):

    def forward(
        self,
        predictions: DetectionOutput,
        targets: list[DetectionTarget],
        matches,
    ) -> torch.Tensor:

        #
        # Binary confidence
        #

        return F.binary_cross_entropy(
            ...,
        )-->