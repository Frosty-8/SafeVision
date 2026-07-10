<!--"""
SafeVision AI

Training Callbacks

Provides callback interfaces for the training loop.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from app.utils.logger import logger


class Callback(ABC):
    """
    Base callback.

    Every callback in SafeVision inherits this class.
    """

    def on_train_start(
        self,
        trainer,
    ) -> None:
        """
        Called before training starts.
        """
    
        pass


    def on_train_end(
        self,
        trainer,
    ) -> None:
        """
        Called after training finishes.
        """
    
        pass


    def on_epoch_start(
        self,
        trainer,
        epoch: int,
    ) -> None:
        """
        Called before each epoch.
        """
    
        pass


    def on_epoch_end(
        self,
        trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Called after every epoch.
        """
    
        pass

    def on_batch_start(
        self,
        trainer,
        batch_index: int,
    ) -> None:
        """
        Called before every batch.
        """
    
        pass

    def on_batch_end(
        self,
        trainer,
        batch_index: int,
        loss: float,
    ) -> None:
        """
        Called after every batch.
        """
    
        pass

    def on_validation_start(
        self,
        trainer,
    ) -> None:
        """
        Called before validation.
        """
    
        pass

    def on_validation_end(
        self,
        trainer,
        metrics: dict[str, float],
    ) -> None:
        """
        Called after validation.
        """
    
        pass

    def on_checkpoint_save(
        self,
        trainer,
        path: str,
    ) -> None:
        """
        Called after checkpoint saving.
        """
    
        pass

    def on_exception(
        self,
        trainer,
        exception: Exception,
    ) -> None:
        """
        Called if training crashes.
        """
    
        logger.exception(
            "Training crashed.",
        )

class CallbackManager:
    """
    Executes callbacks during training.
    """
    def __init__(
        self,
        callbacks: list[Callback] | None = None,
    ) -> None:
    
        self.callbacks = callbacks or []


    def register(
        self,
        callback: Callback,
    ) -> None:
        """
        Register callback.
        """
    
        self.callbacks.append(
            callback,
        )

    def notify(
        self,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Notify every callback of an event.
        """
    
        for callback in self.callbacks:
    
            handler = getattr(
                callback,
                event,
                None,
            )
    
            if callable(handler):
    
                handler(
                    *args,
                    **kwargs,
                )-->