from __future__ import annotations

import torch
import torch.nn as nn

from app.models.detector.deformable_attention import (
    DeformableAttention,
    SpatialShape
)
from app.utils.logger import logger

class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer.
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
    
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
    
        self.cross_attention = DeformableAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_points=num_points,
        )
    
        hidden_dim = embedding_dim * mlp_ratio
    
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )
    
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)


    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        spatial_shape: SpatialShape,
    ) -> torch.Tensor:
        self_attention, _ = self.self_attention(
            queries,
            queries,
            queries,
        )

        queries = self.norm1(
            queries + self_attention,
        )

        cross_attention = self.cross_attention(
            query=queries,
            value=memory,
            spatial_shape=spatial_shape,
        )

        queries = self.norm2(
            queries + cross_attention,
        )

        ffn_output = self.ffn(
            queries,
        )
        
        queries = self.norm3(
            queries + ffn_output,
        )
        
        return queries

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_queries: int = 300,
        num_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
    ) -> None:
    
        super().__init__()
    
        self.query_embeddings = nn.Embedding(
            num_queries,
            embedding_dim,
        )
    
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    embedding_dim,
                    num_heads,
                    num_points,
                )
                for _ in range(num_layers)
            ]
        )
    
        self.num_queries = num_queries
    
        self.embedding_dim = embedding_dim
    
        logger.info(
            "TransformerDecoder initialized."
        )

    def forward(
        self,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = memory.shape
        
        memory = (
            memory
            .flatten(2)
            .transpose(1, 2)
        )


        spatial_shape = SpatialShape(
            height=height,
            width=width,
        )

        queries = self.query_embeddings.weight.unsqueeze(0)
        
        queries = queries.expand(
            batch_size,
            -1,
            -1,
        )

        for layer in self.layers:
        
            queries = layer(
                queries,
                memory,
                spatial_shape,
            )


        return queries