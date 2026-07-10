from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    LRScheduler,
)

from app.utils.logger import logger

class SchedulerFactory:
    """
    Factory for creating learning-rate schedulers.
    """

    def __init__(
        self,
        scheduler: str = "cosine",
        epochs: int = 100,
        steps_per_epoch: int = 100,
        min_learning_rate: float = 1e-6,
        step_size: int = 30,
        gamma: float = 0.1,
        max_learning_rate: float = 1e-3,
    ) -> None:
        self.scheduler = scheduler.lower()

        self.epochs = epochs

        self.min_learning_rate = min_learning_rate

        self.steps_per_epoch = steps_per_epoch

        self.step_size = step_size

        self.gamma = gamma

        self.max_learning_rate = max_learning_rate


    def create(
        self,
        optimizer: Optimizer,
    ) -> LRScheduler:
        """
        Create scheduler.
        """
    
        logger.info(
            "Creating {} scheduler.",
            self.scheduler,
        )
    
        if self.scheduler == "cosine":
    
            return self._create_cosine_scheduler(
                optimizer,
            )
    
        if self.scheduler == "step":
    
            return self._create_step_scheduler(
                optimizer,
            )
    
        if self.scheduler == "plateau":
    
            return self._create_plateau_scheduler(
                optimizer,
            )
    
        if self.scheduler == "onecycle":
    
            return self._create_onecycle_scheduler(
                optimizer,
            )
    
        raise ValueError(
            f"Unsupported scheduler: {self.scheduler}"
        )

    def _create_cosine_scheduler(
        self,
        optimizer: Optimizer,
    ) -> LRScheduler:
        """
        Create cosine annealing scheduler.
        """
    
        return CosineAnnealingLR(
    
            optimizer,
    
            T_max=self.epochs,
    
            eta_min=self.min_learning_rate,
        )

    def _create_step_scheduler(
        self,
        optimizer: Optimizer,
    ) -> LRScheduler:
        """
        Create step scheduler.
        """
    
        return StepLR(
    
            optimizer,
    
            step_size=self.step_size,
    
            gamma=self.gamma,
        )

    
    def _create_plateau_scheduler(
        self,
        optimizer: Optimizer,
    ):
        """
        Create ReduceLROnPlateau scheduler.
        """
    
        return ReduceLROnPlateau(
    
            optimizer,
    
            mode="min",
    
            factor=self.gamma,
    
            patience=5,
    
            min_lr=self.min_learning_rate,
        )

    def _create_onecycle_scheduler(
        self,
        optimizer: Optimizer,
    ) -> LRScheduler:
        """
        Create OneCycle scheduler.
        """
    
        return OneCycleLR(
    
            optimizer,
    
            max_lr=self.max_learning_rate,
    
            epochs=self.epochs,
    
            steps_per_epoch=self.steps_per_epoch,
        )

    @property
    def supported_schedulers(
        self,
    ) -> tuple[str, ...]:
        """
        Supported schedulers.
        """
    
        return (
    
            "cosine",
    
            "step",
    
            "plateau",
    
            "onecycle",
        )