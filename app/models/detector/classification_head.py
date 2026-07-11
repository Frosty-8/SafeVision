from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger


class ClassificationHead(nn.Module):
    """
    Predict class logits for every object query.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 91,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

        logger.info("ClassificationHead initialized.")

    def forward(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        queries

            (B,Q,C)

        Returns
        -------
        torch.Tensor

            Class logits

            (B,Q,num_classes)
        """

        return self.classifier(
            queries,
        )

    @property
    def output_dim(
        self,
    ) -> int:
        """
        Number of predicted classes.
        """

        return self.num_classes
