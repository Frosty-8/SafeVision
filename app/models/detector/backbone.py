from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger

class ResidualBlock(nn.Module):
    """
    Lightweight residual refinement block.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                channels,
            ),

            nn.ReLU(
                inplace=True,
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                channels,
            ),
        )

        self.activation = nn.ReLU(
            inplace=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        out = self.block(
            x,
        )

        out += identity

        return self.activation(
            out,
        )

class Backbone(nn.Module):
    """
    Feature adapter for fused sensor
    representations.
    """
    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels: int = 256,
    ) -> None:
    
        super().__init__()
    
        self.input_projection = nn.Sequential(
    
            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
    
            nn.BatchNorm2d(
                hidden_channels,
            ),
    
            nn.ReLU(
                inplace=True,
            ),
        )
    
        self.refinement = nn.Sequential(
    
            ResidualBlock(
                hidden_channels,
            ),
    
            ResidualBlock(
                hidden_channels,
            ),
    
        )
    
        self.output_projection = nn.Sequential(
    
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
    
            nn.BatchNorm2d(
                hidden_channels,
            ),
        )

        self._output_channels = hidden_channels
        
        logger.info(
            "Detector backbone initialized."
        )

    @property
    def output_channels(
        self,
    ) -> int:
        """
        Output feature dimension.
        """
        return self._output_channels

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine fused features before the
        transformer encoder.
        """
    
        features = self.input_projection(
            features,
        )
    
        features = self.refinement(
            features,
        )
    
        features = self.output_projection(
            features,
        )
    
        return features


