from __future__ import annotations

from torch import nn

from app.utils.logger import logger

class LossFactory:
    """
    Factory for creating training losses.
    """

    def __init__(
        self,
        classification_loss: str = "cross_entropy",
        regression_loss: str = "smooth_l1",
    ) -> None:
        self.classification_loss = classification_loss.lower()
        
        self.regression_loss = regression_loss.lower()


    def create(
        self,
    ) -> dict[str, nn.Module]:
        """
        Create training losses.
        """
    
        logger.info(
            "Creating training losses."
        )
    
        return {
    
            "classification": self._classification_loss(),
    
            "regression": self._regression_loss(),
    
        }


    def _classification_loss(
        self,
    ) -> nn.Module:
        """
        Create classification loss.
        """
    
        if self.classification_loss == "cross_entropy":
    
            return nn.CrossEntropyLoss()
    
        if self.classification_loss == "focal":
    
            return self._focal_loss()
    
        raise ValueError(
            f"Unsupported classification loss: "
            f"{self.classification_loss}"
        )


    def _regression_loss(
        self,
    ) -> nn.Module:
        """
        Create regression loss.
        """
    
        if self.regression_loss == "l1":
    
            return nn.L1Loss()
    
        if self.regression_loss == "smooth_l1":
    
            return nn.SmoothL1Loss()
    
        raise ValueError(
            f"Unsupported regression loss: "
            f"{self.regression_loss}"
        )

    def _focal_loss(
        self,
    ) -> nn.Module:
        """
        Create focal loss.
        """
    
        return FocalLoss()



class FocalLoss(nn.Module):
    """
    Binary focal loss.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.loss = nn.BCEWithLogitsLoss(
            reduction="none",
        )

    def forward(
        self,
        inputs,
        targets,
    ):
    
        bce = self.loss(
            inputs,
            targets,
        )
    
        probability = (-bce).exp()
    
        focal = (
            self.alpha
            * (1 - probability) ** self.gamma
            * bce
        )
    
        return focal.mean()




    @property
    def supported_losses(
        self,
    ) -> dict[str, tuple[str, ...]]:
        """
        Supported losses.
        """
    
        return {
    
            "classification": (
    
                "cross_entropy",
    
                "focal",
    
            ),
    
            "regression": (
    
                "l1",
    
                "smooth_l1",
    
            ),
    
        }
    