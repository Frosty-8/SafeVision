"""
SafeVision AI

LiDAR Feature Encoder

Encodes BEV LiDAR features into the
shared embedding space.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger

class LiDAREncoder(nn.Module):
    """
    Encodes LiDAR BEV features into the common embedding space.
    """
    def __init__(
        self,
        input_channels: int = 4,
        embedding_dim: int = 256,
    ) -> None:
    
        super().__init__()
    
        self.embedding_dim = embedding_dim
    
        self.encoder = self._build_encoder()
    
        self.projection = self._build_projection()
    
        logger.info(
            "LiDAREncoder initialized."
        )

    def _build_encoder(
        self,
    ) -> nn.Module:
        """
        Build LiDAR encoder.
        """
    
        return nn.Sequential(
    
            nn.Conv2d(
                4,
                64,
                kernel_size=3,
                padding=1,
            ),
    
            nn.BatchNorm2d(64),
    
            nn.ReLU(inplace=True),
    
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
    
            nn.BatchNorm2d(128),
    
            nn.ReLU(inplace=True),
    
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
    
            nn.BatchNorm2d(256),
    
            nn.ReLU(inplace=True),
        )

    def _build_projection(
        self,
    ) -> nn.Module:
        """
        Project features into the common
        embedding dimension.
        """
    
        return nn.Sequential(
    
            nn.Conv2d(
                256,
                self.embedding_dim,
                kernel_size=1,
            ),
    
            nn.BatchNorm2d(
                self.embedding_dim,
            ),
    
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        lidar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode LiDAR features.
    
        Parameters
        ----------
        lidar
    
            (B,C,H,W)
    
        Returns
        -------
        Tensor
    
            (B,256,H,W)
        """
    
        features = self.encoder(
            lidar,
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