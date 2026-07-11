from __future__ import annotations

import torch
import torch.nn as nn

from app.models.detector.deformable_attention import (
    DeformableAttention,
    SpatialShape
)
from app.models.detector.positional_encoding import (
    PositionalEncoding
)

from app.utils.logger import logger


class EncoderLayer(nn.Module):
    """
    Single deformable transformer
    encoder layer.
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
    
        super().__init__()
    
        self.attention = DeformableAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_points=num_points,
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim,
        )
        
        self.norm2 = nn.LayerNorm(
            embedding_dim,
        )

        hidden_dim = embedding_dim * mlp_ratio
        
        self.ffn = nn.Sequential(
        
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
        
            nn.GELU(),
        
            nn.Dropout(
                dropout,
            ),
        
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        
            nn.Dropout(
                dropout,
            ),
        )



    def forward(
        self,
        tokens: torch.Tensor,
        spatial_shape: SpatialShape,
    ) -> torch.Tensor:
        attention = self.attention(
            query=tokens,
            value=tokens,
            spatial_shape=spatial_shape,
        )

        tokens = tokens + attention
        
        tokens = self.norm1(
            tokens,
        )

        output = self.ffn(
            tokens,
        )

        tokens = tokens + output
        
        tokens = self.norm2(
            tokens,
        )
        
        return tokens


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
    ) -> None:
    
        super().__init__()
    
        self.position = PositionalEncoding(
            embedding_dim,
        )
    
        self.layers = nn.ModuleList(
    
            [
    
                EncoderLayer(
    
                    embedding_dim,
    
                    num_heads,
    
                    num_points,
    
                )
    
                for _ in range(
                    num_layers,
                )
    
            ]
    
        )
    
        logger.info(
            "TransformerEncoder initialized."
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        features = self.position(
            features,
        )


        batch_size, channels, height, width = features.shape
        
        tokens = (
        
            features
        
            .flatten(2)
        
            .transpose(1, 2)
        
        )

        spatial_shape = SpatialShape(
            height=height,
            width=width,
        )


        for layer in self.layers:
        
            tokens = layer(
                tokens,
                spatial_shape,
            )

        tokens = (
        
            tokens
        
            .transpose(1, 2)
        
            .reshape(
        
                batch_size,
        
                channels,
        
                height,
        
                width,
        
            )
        
        )

        return tokens