from __future__ import annotations

import torch
import torch.nn as nn

from app.models.detector.backbone import Backbone
from app.models.detector.encoder import (
    TransformerEncoder
)
from app.models.detector.decoder import (
    TransformerDecoder
)
from app.utils.logger import logger


class DeformableTransformer(nn.Module):
    """
    Complete transformer used by SafeVision DETR.
    """
    def __init__(
        self,
        input_channels: int = 256,
        embedding_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        num_queries: int = 300,
    ) -> None:
    
        super().__init__()

        self.backbone = Backbone(
            input_channels=input_channels,
            hidden_channels=embedding_dim,
        )

        self.encoder = TransformerEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            num_points=num_points,
        )

        self.decoder = TransformerDecoder(
            embedding_dim=embedding_dim,
            num_queries=num_queries,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            num_points=num_points,
        )

        logger.info(
            "DeformableTransformer initialized."
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the complete transformer.
    
        Parameters
        ----------
        features
            Fused feature map
    
            (B,C,H,W)
    
        Returns
        -------
        torch.Tensor
    
            Detection queries
    
            (B,Q,C)
        """
        features = self.backbone(
            features,
        )

        memory = self.encoder(
            features,
        )

        queries = self.decoder(
            memory,
        )

        return queries

    @property
    def output_dim(
        self,
    ) -> int:
        """
        Output embedding dimension.
        """
    
        return self.decoder.embedding_dim


    @property
    def num_queries(
        self,
    ) -> int:
        """
        Number of object queries.
        """
    
        return self.decoder.num_queries