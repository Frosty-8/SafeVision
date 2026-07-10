"""
SafeVision AI

Fusion Transformer

Attention-based sensor fusion.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger


class FusionTransformer(nn.Module):
    """
    Multi-modal transformer encoder.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=embedding_dim,

            nhead=num_heads,

            dropout=dropout,

            batch_first=True,

            activation="gelu",

        )

        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers,

        )

        logger.info(
            "FusionTransformer initialized."
        )


    def _flatten(
        self,
        features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[int, int],
    ]:
        """
        Convert feature map into tokens.
        """
    
        batch, channels, height, width = features.shape
    
        tokens = (
    
            features
    
            .flatten(2)
    
            .transpose(1, 2)
    
        )
    
        return tokens, (height, width)


    def _unflatten(
        self,
        tokens: torch.Tensor,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Restore feature map.
        """
    
        height, width = shape
    
        batch, _, channels = tokens.shape
    
        return (
    
            tokens
    
            .transpose(1,2)
    
            .reshape(
    
                batch,
    
                channels,
    
                height,
    
                width,
    
            )
    
        )


    def forward(
        self,
        camera_features: torch.Tensor,
        lidar_features: torch.Tensor,
        radar_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse three sensor modalities.
        """
    
        camera,spatial = self._flatten(
            camera_features,
        )
    
        lidar,_ = self._flatten(
            lidar_features,
        )
    
        radar,_ = self._flatten(
            radar_features,
        )
    
        tokens = torch.cat(
    
            [
    
                camera,
    
                lidar,
    
                radar,
    
            ],
    
            dim=1,
    
        )
    
        fused = self.encoder(
            tokens,
        )
    
        fused = fused[:, : camera.shape[1]]
    
        return self._unflatten(
    
            fused,
    
            spatial,
    
        )