from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
from app.utils.logger import logger


class BaseDetector(
    nn.Module,
    ABC,
):
    """
    Base class for every detector.

    Every detector must implement
    forward().
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        logger.info(
            "{} initialized.",
            self.__class__.__name__,
        )

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
    ):
        """
        Run detector inference.
        """

        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
    ):
        """
        Perform inference.
        """

        self.eval()

        return self.forward(
            features,
        )

    def freeze(
        self,
    ) -> None:
        """
        Freeze every parameter.
        """

        for parameter in self.parameters():

            parameter.requires_grad = False

        logger.info("Detector frozen.")

    def unfreeze(
        self,
    ) -> None:
        """
        Unfreeze every parameter.
        """

        for parameter in self.parameters():

            parameter.requires_grad = True

        logger.info("Detector unfrozen.")

    def save_weights(
        self,
        path: str | Path,
    ) -> None:
        """
        Save model weights.
        """

        torch.save(
            self.state_dict(),
            path,
        )

        logger.info(
            "Weights saved to '{}'.",
            path,
        )

    def load_weights(
        self,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> None:
        """
        Load model weights.
        """

        state = torch.load(
            path,
            map_location=map_location,
        )

        self.load_state_dict(
            state,
        )

        logger.info(
            "Weights loaded from '{}'.",
            path,
        )

    @property
    def num_parameters(
        self,
    ) -> int:
        """
        Total parameters.
        """

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameters(
        self,
    ) -> int:
        """
        Trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def summary(
        self,
    ) -> None:
        """
        Print detector summary.
        """

        logger.info(
            "Detector: {}",
            self.__class__.__name__,
        )

        logger.info(
            "Parameters: {:,}",
            self.num_parameters,
        )

        logger.info(
            "Trainable: {:,}",
            self.trainable_parameters,
        )

    @property
    def device(
        self,
    ) -> torch.device:
        """
        Return the device where the model
        currently resides.
        """

        return next(self.parameters()).device
