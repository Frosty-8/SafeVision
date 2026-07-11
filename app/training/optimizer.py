"""
SafeVision AI

Optimizer Factory

Creates optimizers for model training.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from torch import nn
from torch.optim import AdamW
from torch.optim import SGD
from torch.optim import Optimizer

from app.utils.logger import logger


class OptimizerFactory:
    """
    Factory for creating optimizers.
    """

    def __init__(
        self,
        optimizer: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
    ) -> None:

        self.optimizer = optimizer.lower()

        self.learning_rate = learning_rate

        self.weight_decay = weight_decay

        self.momentum = momentum

    def create(
        self,
        model: nn.Module,
    ) -> Optimizer:
        """
        Create optimizer.
        """

        logger.info(
            "Creating {} optimizer.",
            self.optimizer,
        )

        parameters = self._group_parameters(
            model,
        )

        if self.optimizer == "adamw":

            return self._create_adamw(
                parameters,
            )

        if self.optimizer == "sgd":

            return self._create_sgd(
                parameters,
            )

        raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def _group_parameters(
        self,
        model: nn.Module,
    ) -> list[dict]:
        """
        Create optimizer parameter groups.
        """

        decay = []

        no_decay = []

        for name, parameter in model.named_parameters():

            if not parameter.requires_grad:
                continue

            if name.endswith("bias") or parameter.ndim == 1:

                no_decay.append(parameter)

            else:

                decay.append(parameter)

        logger.info(
            "Weight decay parameters: {}",
            len(decay),
        )

        logger.info(
            "No weight decay parameters: {}",
            len(no_decay),
        )

        return [
            {
                "params": decay,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ]

    def _create_adamw(
        self,
        parameters: list[dict],
    ) -> Optimizer:
        """
        Create AdamW optimizer.
        """

        return AdamW(
            parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _create_sgd(
        self,
        parameters: list[dict],
    ) -> Optimizer:
        """
        Create SGD optimizer.
        """

        return SGD(
            parameters,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )

    @property
    def supported_optimizers(
        self,
    ) -> tuple[str, ...]:
        """
        Supported optimizers.
        """

        return (
            "adamw",
            "sgd",
        )
