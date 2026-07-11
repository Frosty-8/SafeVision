from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger


class ConfidenceHead(nn.Module):
    """
    Predict object confidence scores.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim

        self.confidence = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(
                inplace=True,
            ),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(
                inplace=True,
            ),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        logger.info("ConfidenceHead initialized.")

    def forward(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict confidence scores.

        Parameters
        ----------
        queries

            (B,Q,C)

        Returns
        -------
        torch.Tensor

            (B,Q,1)
        """

        scores = self.confidence(
            queries,
        )

        return torch.sigmoid(
            scores,
        )

    @property
    def output_dim(
        self,
    ) -> int:
        """
        Number of confidence outputs.
        """

        return 1
