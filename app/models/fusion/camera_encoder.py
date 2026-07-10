from __future__ import annotations

import torch
import torch.nn as nn

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from app.utils.logger import logger


class CameraEncoder(nn.Module):
    """
    Camera feature encoder.

    Produces a fixed-dimensional feature map
    suitable for sensor fusion.
    """

    def __init__(self, 
        embedding_dim: int = 256,
       pretrained: bool = True
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.backbone = self._build_backbone(
            pretrained,
        )

        self.projection = self._build_projection()

        logger.info(
            "CameraEncoder Initialized."
        )


    def _build_backbone(
        self,
        pretrained: bool,
    ) -> nn.Module:
        """
        Build CNN backbone.
        """
    
        model = resnet18(
    
            weights=(
                ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )
    
        )
    
        backbone = nn.Sequential(
            *list(model.children())[:-2]
        )
    
        return backbone


    def _build_projection(
        self,
    ) -> nn.Module:
        """
        Project backbone features into the
        common fusion embedding dimension.
        """
    
        return nn.Sequential(
    
            nn.Conv2d(
    
                512,
    
                self.embedding_dim,
    
                kernel_size=1,
    
            ),
    
            nn.BatchNorm2d(
    
                self.embedding_dim,
    
            ),
    
            nn.ReLU(
                inplace=True,
            ),
    
        )


    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode RGB images.
    
        Parameters
        ----------
        images
    
            Tensor
    
            Shape
    
            (B,3,H,W)
    
        Returns
        -------
        Tensor
    
            Shape
    
            (B,256,H',W')
        """
    
        features = self.backbone(
            images,
        )
    
        embeddings = self.projection(
            features,
        )
    
        return embeddings

    @property
    def output_channels(
        self,
    ) -> int:
        """
        Output embedding dimension.
        """
    
        return self.embedding_dim