from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch
from torch.amp import GradScaler
from torch.amp import autocast

from app.utils.logger import logger


class MixedPrecisionTrainer:
    """
    Automatic Mixed Precision manager.
    """

    def __init__(self, enabled: bool = True, device: str = "cuda") -> None:
        self.enabled = (
            enabled and torch.cuda.is_available() and device.startswith("cuda")
        )

        self.device = device

        self.scaler = GradScaler(device=device, enabled=self.enabled)

        logger.info("Mixed Precision Enabled: {}", self.enabled)

    def autocast(
        self,
    ) -> ContextManager:
        """
        Return autocast context.
        """

        if not self.enabled:

            return nullcontext()

        return autocast(
            device_type=self.device,
        )

    def backward(
        self,
        loss: torch.Tensor,
    ) -> None:
        """
        Backpropagation.
        """

        if self.enabled:

            self.scaler.scale(loss).backward()

        else:

            loss.backward()

    def step(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Optimizer step.
        """

        if self.enabled:

            self.scaler.step(
                optimizer,
            )

        else:

            optimizer.step()

    def update(
        self,
    ) -> None:
        """
        Update GradScaler.
        """

        if self.enabled:

            self.scaler.update()

    def state_dict(
        self,
    ) -> dict:
        """
        Return scaler state.
        """

        return self.scaler.state_dict()

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        """
        Restore scaler state.
        """

        self.scaler.load_state_dict(
            state,
        )
