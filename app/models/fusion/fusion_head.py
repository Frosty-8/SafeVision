"""
SafeVision AI

Fusion Head

Aligns camera, LiDAR and radar feature maps
into a common embedding space.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger


class FusionHead(nn.Module):
    """
    Feature alignment module for
    multi-modal sensor fusion.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim

        self.camera_projection = self._build_projection()

        self.lidar_projection = self._build_projection()

        self.radar_projection = self._build_projection()

        logger.info("FusionHead initialized.")

    def _build_projection(
        self,
    ) -> nn.Sequential:
        """
        Build a projection layer.
        """

        return nn.Sequential(
            nn.Conv2d(
                self.embedding_dim,
                self.embedding_dim,
                kernel_size=1,
                bias=False,
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
        camera_features: torch.Tensor,
        lidar_features: torch.Tensor,
        radar_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Align feature maps before fusion.
        """

        camera_features = self.camera_projection(
            camera_features,
        )

        lidar_features = self.lidar_projection(
            lidar_features,
        )

        radar_features = self.radar_projection(
            radar_features,
        )

        return (
            camera_features,
            lidar_features,
            radar_features,
        )
