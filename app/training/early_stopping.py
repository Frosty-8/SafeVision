"""
SafeVision AI

Early Stopping

Stops training when validation metrics stop improving.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from typing import Any

from app.utils.logger import logger


class EarlyStopping:
    """
    Monitors a validation metric and stops
    training when no improvement is observed.
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = "loss",
        mode: str = "min",
    ) -> None:
    
        self.patience = patience
    
        self.min_delta = min_delta
    
        self.monitor = monitor
    
        self.mode = mode.lower()
    
        self.counter = 0
    
        self.should_stop = False
    
        self.best_score = (
            float("inf")
            if self.mode == "min"
            else float("-inf")
        )
    
        logger.info(
            "EarlyStopping initialized."
        )


    def step(
        self,
        metrics: dict[str, float],
    ) -> bool:
        """
        Update early stopping state.
    
        Returns
        -------
        bool
            True if training should stop.
        """
    
        score = metrics[self.monitor]
    
        if self._is_improvement(score):
    
            self.best_score = score
    
            self.counter = 0
    
            logger.info(
                "Validation {} improved to {:.6f}.",
                self.monitor,
                score,
            )
    
        else:
    
            self.counter += 1
    
            logger.info(
                "No improvement ({}/{})",
                self.counter,
                self.patience,
            )
    
            if self.counter >= self.patience:
    
                self.should_stop = True
    
                logger.warning(
                    "Early stopping triggered."
                )
    
        return self.should_stop



    def _is_improvement(
        self,
        score: float,
    ) -> bool:
        """
        Determine whether the metric improved.
        """
    
        if self.mode == "min":
    
            return (
                score
                < self.best_score - self.min_delta
            )
    
        return (
            score
            > self.best_score + self.min_delta
        )


    def reset(
        self,
    ) -> None:
        """
        Reset internal state.
        """
    
        self.counter = 0
    
        self.should_stop = False
    
        self.best_score = (
            float("inf")
            if self.mode == "min"
            else float("-inf")
        )



    def state_dict(
        self,
    ) -> dict[str, Any]:
        """
        Serialize state.
        """
    
        return {
    
            "counter": self.counter,
    
            "best_score": self.best_score,
    
            "should_stop": self.should_stop,
    
        }



    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        """
        Restore state.
        """
    
        self.counter = state["counter"]
    
        self.best_score = state["best_score"]
    
        self.should_stop = state["should_stop"]