from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from app.training.callbacks import CallbackManager
from app.training.checkpoint import CheckpointManager
from app.training.early_stopping import EarlyStopping
from training.mixed_precision import MixedPrecisionTrainer
from app.training.validator import Validator

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)

class Trainer:
    """
    SafeVision training engine.

    Coordinates every component of the training pipeline.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validator: Validator,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        checkpoint: CheckpointManager,
        early_stopping: EarlyStopping,
        mixed_precision: MixedPrecisionTrainer,
        callbacks: CallbackManager,
        device: torch.device,
        epochs: int = 100,
    ) -> None:
    
        self.model = model
    
        self.train_loader = train_loader
    
        self.validator = validator
    
        self.optimizer = optimizer
    
        self.scheduler = scheduler
    
        self.checkpoint = checkpoint
    
        self.early_stopping = early_stopping
    
        self.mixed_precision = mixed_precision
    
        self.callbacks = callbacks
    
        self.device = device
    
        self.epochs = epochs
    
        self.model.to(device)
    
        logger.info(
            "Trainer initialized."
        )



    def fit(
        self,
    ) -> None:
        """
        Run complete training.
        """
    
        title(
            "Training Started"
        )
    
        self.callbacks.notify(
            "on_train_start",
            self,
        )
    
        for epoch in range(
            1,
            self.epochs + 1,
        ):
    
            self.callbacks.notify(
                "on_epoch_start",
                self,
                epoch,
            )
    
            train_loss = self.train_epoch()
    
            metrics = self.validator.validate()
    
            metrics["train_loss"] = train_loss
    
            self._step_scheduler(
                metrics,
            )
    
            self._save_checkpoints(
                epoch,
                metrics,
            )
    
            self.callbacks.notify(
                "on_epoch_end",
                self,
                epoch,
                metrics,
            )
    
            if self.early_stopping.step(
                metrics,
            ):
    
                break
    
        self.callbacks.notify(
            "on_train_end",
            self,
        )
    
        success(
            "Training completed."
        )


    def train_epoch(
        self,
    ) -> float:
        """
        Train one epoch.
        """
    
        self.model.train()
    
        total_loss = 0.0
    
        batches = 0
    
        for batch_index, (
            images,
            targets,
        ) in enumerate(
            self.train_loader,
        ):
    
            loss = self.train_batch(
                images,
                targets,
                batch_index,
            )
    
            total_loss += loss
    
            batches += 1
    
        return (
            total_loss
            / max(
                batches,
                1,
            )
        )



    def train_batch(
        self,
        images,
        targets,
        batch_index: int,
    ) -> float:
    
        self.callbacks.notify(
            "on_batch_start",
            self,
            batch_index,
        )
    
        images = images.to(
            self.device,
        )
    
        self.optimizer.zero_grad(
            set_to_none=True,
        )
    
        with self.mixed_precision.autocast():
    
            outputs = self.model(
                images,
            )
    
            #
            # Placeholder until DETR criterion
            #
    
            loss = outputs.mean()
    
        self.mixed_precision.backward(
            loss,
        )
    
        self.mixed_precision.step(
            self.optimizer,
        )
    
        self.mixed_precision.update()
    
        self.callbacks.notify(
            "on_batch_end",
            self,
            batch_index,
            float(
                loss.item(),
            ),
        )
    
        return float(
            loss.item(),
        )


    def _step_scheduler(
        self,
        metrics: dict[str, float],
    ) -> None:
    
        if self.scheduler is None:
    
            return
    
        if isinstance(
            self.scheduler,
            ReduceLROnPlateau,
        ):
    
            self.scheduler.step(
                metrics["loss"],
            )
    
        else:
    
            self.scheduler.step()



    def _save_checkpoints(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
    
        self.checkpoint.save_last(
    
            epoch=epoch,
    
            model=self.model,
    
            optimizer=self.optimizer,
    
            scheduler=self.scheduler,
    
            scaler=self.mixed_precision.scaler,
    
            metrics=metrics,
        )
    
        self.checkpoint.save_best(
    
            metric=metrics["loss"],
    
            epoch=epoch,
    
            model=self.model,
    
            optimizer=self.optimizer,
    
            scheduler=self.scheduler,
    
            scaler=self.mixed_precision.scaler,
    
            metrics=metrics,
        )