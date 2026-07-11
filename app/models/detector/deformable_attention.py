from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.utils.logger import logger
from dataclasses import dataclass


@dataclass(slots=True)
class SpatialShape:
    """
    Spatial dimensions of a feature map.
    """

    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError("height must be positive.")

        if self.width <= 0:
            raise ValueError("width must be positive.")


class DeformableAttention(nn.Module):
    """
    Multi-scale deformable attention.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
    ) -> None:

        super().__init__()

        if embedding_dim % num_heads != 0:

            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        self.embedding_dim = embedding_dim

        self.num_heads = num_heads

        self.num_points = num_points

        self.head_dim = embedding_dim // num_heads


        self.offset_projection = nn.Linear(
        
            embedding_dim,
        
            num_heads * num_points * 2,
        
        )

        self.attention_projection = nn.Linear(
        
            embedding_dim,
        
            num_heads * num_points,
        
        )

        self.value_projection = nn.Linear(
        
            embedding_dim,
        
            embedding_dim,
        
        )

        self.output_projection = nn.Linear(
        
            embedding_dim,
        
            embedding_dim,
        
        )

        logger.info(
            "DeformableAttention initialized."
        )

    def _reference_points(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate normalized reference
        points for every spatial location.
        """
        y = torch.linspace(
            0.5,
            height - 0.5,
            height,
            device=device,
        )
        
        x = torch.linspace(
            0.5,
            width - 0.5,
            width,
            device=device,
        )
        
        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )
        
        reference = torch.stack(
        
            (
        
                xx / width,
        
                yy / height,
        
            ),
        
            dim=-1,
        
        )
        
        return reference.reshape(
            -1,
            2,
        )

    def _sampling_grid(
        self,
        offsets: torch.Tensor,
        spatial_shape: SpatialShape,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute normalized sampling grid for
        deformable attention.
    
        Returns
        -------
        torch.Tensor
            Sampling grid of shape
    
            (
                B,
                H * W,
                Heads,
                Points,
                2,
            )
    
            with coordinates in [-1, 1].
        """
    
        reference_points = self._reference_points(
            spatial_shape.height,
            spatial_shape.width,
            device,
        )
    
        #
        # (1, H * W, 2)
        #
        reference_points = reference_points.unsqueeze(
            0,
        )
    
        #
        # (B, H * W, 2)
        #
        reference_points = reference_points.expand(
            offsets.shape[0],
            -1,
            -1,
        )
    
        #
        # (B, H * W, 1, 2)
        #
        reference_points = reference_points.unsqueeze(
            2,
        )
    
        #
        # (B, H * W, Heads, 2)
        #
        reference_points = reference_points.expand(
            -1,
            -1,
            self.num_heads,
            -1,
        )
    
        #
        # (B, H * W, Heads, 1, 2)
        #
        reference_points = reference_points.unsqueeze(
            3,
        )
    
        #
        # (B, H * W, Heads, Points, 2)
        #
        reference_points = reference_points.expand(
            -1,
            -1,
            -1,
            self.num_points,
            -1,
        )
    
        sampling_locations = (
            reference_points
            + offsets
        )
    
        sampling_grid = (
            sampling_locations * 2.0
        ) - 1.0
    
        return sampling_grid


    def _prepare_value_map(
        self,
        value: torch.Tensor,
        spatial_shape: SpatialShape,
    ) -> torch.Tensor:
        """
        Restore the projected value tensor to its
        spatial layout for sampling.
    
        Parameters
        ----------
        value
            Projected value tensor of shape
    
            (
                B,
                H * W,
                Heads,
                HeadDim,
            )
    
        spatial_shape
            Spatial dimensions of the feature map.
    
        Returns
        -------
        torch.Tensor
            Value tensor of shape
    
            (
                B,
                Heads,
                H,
                W,
                HeadDim,
            )
        """
    
        batch_size = value.shape[0]
    
        value = value.permute(
            0,
            2,
            1,
            3,
        )
    
        value = value.reshape(
            batch_size,
            self.num_heads,
            spatial_shape.height,
            spatial_shape.width,
            self.head_dim,
        )
    
        return value


    def _sample_features(
        self,
        value: torch.Tensor,
        sampling_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample feature values from the projected
        value map using bilinear interpolation.
    
        Parameters
        ----------
        value
            Tensor of shape
    
            (
                B,
                Heads,
                H,
                W,
                HeadDim,
            )
    
        sampling_grid
            Tensor of shape
    
            (
                B,
                H * W,
                Heads,
                Points,
                2,
            )
    
        Returns
        -------
        torch.Tensor
    
            (
                B,
                H * W,
                Heads,
                Points,
                HeadDim,
            )
        """
    
        batch_size = value.shape[0]
    
        sampled = []
    
        for head in range(self.num_heads):
    
            #
            # (B,H,W,D)
            #
            head_value = value[:, head]
    
            #
            # (B,D,H,W)
            #
            head_value = head_value.permute(
                0,
                3,
                1,
                2,
            )
    
            #
            # (B,N,P,2)
            #
            grid = sampling_grid[:, :, head]
    
            #
            # Bilinear interpolation
            #
            sampled_head = F.grid_sample(
                head_value,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
    
            #
            # (B,D,N,P)
            #
            sampled_head = sampled_head.permute(
                0,
                2,
                3,
                1,
            )
    
            sampled.append(
                sampled_head,
            )
    
        return torch.stack(
            sampled,
            dim=2,
        )


    def _aggregate(
        self,
        sampled: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate sampled features using
        learned attention weights.
    
        Parameters
        ----------
        sampled
    
            (
                B,
                N,
                Heads,
                Points,
                HeadDim,
            )
    
        attention
    
            (
                B,
                N,
                Heads,
                Points,
            )
    
        Returns
        -------
        torch.Tensor
    
            (
                B,
                N,
                EmbeddingDim,
            )
        """
    
        attention = attention.unsqueeze(
            -1,
        )
    
        output = (
            sampled
            * attention
        ).sum(
            dim=3,
        )
    
        batch_size = output.shape[0]
    
        sequence_length = output.shape[1]
    
        output = output.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )
    
        return output


        

    def _reshape_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Split embedding dimension into
        attention heads.
        """
        batch,length,channels = tensor.shape
        
        return tensor.view(
        
            batch,
        
            length,
        
            self.num_heads,
        
            self.head_dim,
        
        )

    def _validate_input(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Validate attention inputs.
        """
    
        if query.ndim != 3:
            raise ValueError(
                "Query tensor must have shape (B, N, C)."
            )
    
        if value.ndim != 3:
            raise ValueError(
                "Value tensor must have shape (B, N, C)."
            )
    
        if query.shape[0] != value.shape[0]:
            raise ValueError(
                "Batch size mismatch between query and value."
            )
    
        if query.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected query embedding dimension "
                f"{self.embedding_dim}, "
                f"got {query.shape[-1]}."
            )
    
        if value.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected value embedding dimension "
                f"{self.embedding_dim}, "
                f"got {value.shape[-1]}."
            )


    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        spatial_shape: SpatialShape,
    ) -> torch.Tensor:
        """
        Perform deformable attention.
    
        Parameters
        ----------
        query
            Query tensor of shape (B, N, C).
    
        value
            Value tensor of shape (B, N, C).
    
        spatial_shape
            Spatial dimensions of the original feature map.
    
        Returns
        -------
        torch.Tensor
            Refined feature embeddings.
        """
        self._validate_input(
            query,
            value,
        )
    
        batch_size, sequence_length, _ = query.shape
    
        #
        # Project values.
        #
        value = self.value_projection(
            value,
        )
    
        value = self._reshape_heads(
            value,
        )
    
        #
        # Predict sampling offsets.
        #
        offsets = self.offset_projection(
            query,
        )
    
        offsets = offsets.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_points,
            2,
        )
    
        #
        # Predict attention weights.
        #
        attention = self.attention_projection(
            query,
        )
    
        attention = attention.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.num_points,
        )
    
        attention = torch.softmax(
            attention,
            dim=-1,
        )
    
        #
        # Compute normalized sampling grid.
        #
        sampling_grid = self._sampling_grid(
            offsets=offsets,
            spatial_shape=spatial_shape,
            device=query.device,
        )
    
        #
        # Restore projected values to spatial layout.
        #
        value = self._prepare_value_map(
            value=value,
            spatial_shape=spatial_shape,
        )
    
        #
        # Remaining stages:
        #
        sampled = self._sample_features(
            value=value,
            sampling_grid=sampling_grid,
        )
        
        output = self._aggregate(
            sampled=sampled,
            attention=attention,
        )
        
        output = self.output_projection(
            output,
        )
        
        return output