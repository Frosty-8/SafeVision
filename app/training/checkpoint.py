"""
SafeVision AI

Checkpoint Manager

Handles saving and loading training checkpoints.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from app.utils.logger import logger


class CheckpointManager:
    """
    Saves and restores training checkpoints.
    """

    def __init__(
        self,
        directory: str | Path = "models/checkpoints",
        metric_name: str = "loss",
        mode: str = "min",
    ) -> None:

        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metric_name = metric_name

        self.mode = mode.lower()

        self.best_metric = float("inf") if self.mode == "min" else float("-inf")

        logger.info(
            "Checkpoint directory: {}",
            self.directory,
        )

    def save(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        scaler: Any | None,
        metrics: dict[str, float],
        filename: str,
    ) -> Path:
        """
        Save checkpoint.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        path = self.directory / filename

        torch.save(
            checkpoint,
            path,
        )

        logger.info(
            "Checkpoint saved: {}",
            path.name,
        )

        return path

    def load(
        self,
        path: str | Path,
    ) -> dict[str, Any]:
        """
        Load checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        logger.info(
            "Checkpoint loaded: {}",
            Path(path).name,
        )

        return checkpoint

    def save_best(
        self,
        *,
        metric: float,
        **kwargs: Any,
    ) -> bool:
        """
        Save best checkpoint.
        """

        improved = (
            metric < self.best_metric
            if self.mode == "min"
            else metric > self.best_metric
        )

        if not improved:

            return False

        self.best_metric = metric

        self.save(
            filename="best_model.pt",
            **kwargs,
        )

        logger.info("New best checkpoint saved.")

        return True

    def save_last(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Save latest checkpoint.
        """

        self.save(
            filename="last_checkpoint.pt",
            **kwargs,
        )

    def exists(
        self,
        filename: str = "last_checkpoint.pt",
    ) -> bool:
        """
        Check if checkpoint exists.
        """

        return (self.directory / filename).exists()

    def latest_checkpoint(
        self,
    ) -> Path:

        return self.directory / "last_checkpoint.pt"

    def cleanup(
        self,
        keep: int = 5,
    ) -> None:
        """
        Remove old checkpoints.
        """

        checkpoints = sorted(
            self.directory.glob("*.pt"),
            key=lambda file: file.stat().st_mtime,
            reverse=True,
        )

        for checkpoint in checkpoints[keep:]:

            checkpoint.unlink()

            logger.info(
                "Deleted {}",
                checkpoint.name,
            )
