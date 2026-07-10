"""
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
from typing import TYPE_CHECKING

if TYPE_CHECKING: 
    from app.training.trainer import Trainer

class Callback(ABC):
    """
    Base callback.

    Every callback in SafeVision inherits this class.
    """

    def on_train_start(
        self,
        trainer: Trainer,
    ) -> None:
        """
        Called before training starts.
        """
    
        pass


    def on_train_end(
        self,
        trainer: Trainer,
    ) -> None:
        """
        Called after training finishes.
        """
    
        pass


    def on_epoch_start(
        self,
        trainer: Trainer,
        epoch: int,
    ) -> None:
        """
        Called before each epoch.
        """
    
        pass


    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Called after every epoch.
        """
    
        pass

    def on_batch_start(
        self,
        trainer: Trainer,
        batch_index: int,
    ) -> None:
        """
        Called before every batch.
        """
    
        pass

    def on_batch_end(
        self,
        trainer: Trainer,
        batch_index: int,
        loss: float,
    ) -> None:
        """
        Called after every batch.
        """
    
        pass

    def on_before_backward(
        self,
        trainer: Trainer,
    ) -> None:
        """
        Called before loss.backward().
        """
    
        pass
    
    
    def on_after_backward(
        self,
        trainer: Trainer,
    ) -> None:
        """
        Called after loss.backward().
        """
    
        pass
    
    
    def on_optimizer_step(
        self,
        trainer: Trainer,
    ) -> None:
        """
        Called before optimizer.step().
        """
    
        pass
    
    
    def on_scheduler_step(
        self,
        trainer: Trainer,
    ) -> None:
        """
        Called before scheduler.step().
        """
    
        pass

    def on_validation_start(
        self,
        trainer: Trainer,
    ) -> None:
        """
        Called before validation.
        """
    
        pass

    def on_validation_end(
        self,
        trainer: Trainer,
        metrics: dict[str, float],
    ) -> None:
        """
        Called after validation.
        """
    
        pass

    def on_checkpoint_save(
        self,
        trainer: Trainer,
        path: str,
    ) -> None:
        """
        Called after checkpoint saving.
        """
    
        pass

    def on_exception(
        self,
        trainer: Trainer,
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

    def unregister(
        self,
        callback: Callback,
    ) -> None:
        """
        Remove callback.
        """
    
        self.callbacks.remove(
            callback,
        )

    def clear(
        self,
    ) -> None:
        """
        Remove every callback.
        """
    
        self.callbacks.clear()

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
    
            if not callable(handler):
                continue
            
            try:
            
                handler(
                    *args,
                    **kwargs,
                )
            
            except Exception:
            
                logger.exception(
                    "Callback '{}' failed.",
                    callback.__class__.__name__,
                )
            
                raise

    def __len__(
        self,
    ) -> int:
        """
        Number of callbacks.
        """
    
        return len(
            self.callbacks,
        )

    def __iter__(
        self,
    ):
        """
        Iterate over callbacks.
        """
    
        return iter(
            self.callbacks,
        )

    @property
    def has_callbacks(
        self,
    ) -> bool:
        """
        Whether callbacks are registered.
        """
    
        return len(
            self.callbacks,
        ) > 0

    @property
    def count(
        self,
    ) -> int:
        """
        Number of registered callbacks.
        """
    
        return len(
            self.callbacks,
        )