"""
SafeVision AI

Sensor Fusion Pipeline

Combines camera, LiDAR and radar
into one unified feature representation.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.models.fusion.camera_encoder import CameraEncoder
from app.models.fusion.fusion_head import FusionHead
from app.models.fusion.fusion_transformer import FusionTransformer
from app.models.fusion.lidar_encoder import LiDAREncoder
from app.models.fusion.radar_encoder import RadarEncoder

from app.utils.logger import logger


class SensorFusion(nn.Module):
    """
    Multi-modal sensor fusion pipeline.

    Produces a unified feature map for
    downstream object detection.
    """


    def __init__(
        self,
        embedding_dim: int = 256,
    ) -> None:
    
        super().__init__()
    
        self.camera_encoder = CameraEncoder(
            embedding_dim=embedding_dim,
        )
    
        self.lidar_encoder = LiDAREncoder(
            embedding_dim=embedding_dim,
        )
    
        self.radar_encoder = RadarEncoder(
            embedding_dim=embedding_dim,
        )
    
        self.fusion_head = FusionHead(
            embedding_dim=embedding_dim,
        )
    
        self.transformer = FusionTransformer(
            embedding_dim=embedding_dim,
        )
    
        logger.info(
            "SensorFusion initialized."
        )


    def forward(
        self,
        camera: torch.Tensor,
        lidar: torch.Tensor,
        radar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse all available sensor inputs.
    
        Parameters
        ----------
        camera
            RGB images.
    
        lidar
            LiDAR feature tensor.
    
        radar
            Radar feature tensor.
    
        Returns
        -------
        torch.Tensor
            Unified feature map.
        """
    
        camera_features = self.camera_encoder(
            camera,
        )
    
        lidar_features = self.lidar_encoder(
            lidar,
        )
    
        radar_features = self.radar_encoder(
            radar,
        )
    
        (
            camera_features,
            lidar_features,
            radar_features,
        ) = self.fusion_head(
            camera_features,
            lidar_features,
            radar_features,
        )
    
        fused = self.transformer(
            camera_features,
            lidar_features,
            radar_features,
        )
    
        return fused


    @property
    def output_channels(
        self,
    ) -> int:
        """
        Output embedding dimension.
        """
    
        return (
            self.camera_encoder.output_channels
        )