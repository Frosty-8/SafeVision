"""
SafeVision AI

Validation Engine

Evaluates models on validation datasets.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.utils.logger import logger
from rich.table import Table

from app.utils.console import console

class Validator:
    """
    Validation engine for SafeVision.
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        losses: dict[str, nn.Module],
        device: torch.device,
    ) -> None:
    
        self.model = model
    
        self.dataloader = dataloader
    
        self.losses = losses
    
        self.device = device
    
        logger.info(
            "Validator initialized."
        )

    def validate(
        self,
    ) -> dict[str, float]:
        """
        Run one validation epoch.
        """
    
        logger.info(
            "Starting validation..."
        )
    
        self.model.eval()
    
        metrics = self._validate_epoch()
    
        self.print_summary(
            metrics,
        )
    
        return metrics





    def _validate_epoch(
        self,
    ) -> dict[str, float]:
    
        total_loss = 0.0
    
        batches = 0
    
        with torch.no_grad():
    
            for images, targets in self.dataloader:
    
                images = images.to(
                    self.device,
                )
    
                outputs = self.model(
                    images,
                )
    
                loss = self._compute_loss(
                    outputs,
                    targets,
                )
    
                total_loss += float(
                    loss.item(),
                )
    
                batches += 1
    
        return {
    
            "loss":
                total_loss / max(
                    batches,
                    1,
                )
    
        }


    def _compute_loss(
        self,
        outputs: Any,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Compute validation loss.
    
        Placeholder until detector criterion
        is implemented.
        """
    
        if isinstance(
            outputs,
            torch.Tensor,
        ):
    
            return outputs.mean()
    
        raise NotImplementedError(
            "Detection loss not implemented."
        )


    def print_summary(
        self,
        metrics: dict[str, float],
    ) -> None:
        """
        Print validation metrics.
        """
    
        table = Table(
            title="Validation Summary",
        )
    
        table.add_column("Metric")
    
        table.add_column(
            "Value",
            justify="right",
        )
    
        for key, value in metrics.items():
    
            table.add_row(
                key,
                f"{value:.4f}",
            )
    
        console.print(
            table,
        )