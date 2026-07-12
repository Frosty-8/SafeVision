from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import asdict
import json
from app.schemas.detection import (
    DetectionOutput,
    DetectionTarget,
    EvaluationReport,
)

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)


class Evaluator:
    """
    SafeVision evalua
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:
    
        self.model = model
    
        self.dataloader = dataloader
    
        self.device = device
    
        self.model.to(
            device,
        )
    
        logger.info(
            "Evaluator initialized."
        )

    def _reset(
        self,
    ) -> None:
        """
        Reset evaluation statistics.
        """
    
        self.total_loss = 0.0
    
        self.total_images = 0
    
        self.total_detections = 0
    
        self.total_latency = 0.0
    
        self.total_batches = 0

    def summary(
        self,
    ) -> None:
        """
        Display evaluator configuration.
        """
    
        logger.info(
            "Evaluation Summary"
        )
    
        logger.info(
            "Device: %s",
            self.device,
        )
    
        logger.info(
            "Evaluation batches: %d",
            len(self.dataloader),
        )
    
        logger.info(
            "Model: %s",
            type(self.model).__name__,
        )

    def evaluate(
        self,
    ) -> EvaluationReport:
        """
        Evaluate the complete dataset.
        """
    
        title("Model Evaluation")
    
        logger.info(
            "Starting evaluation..."
        )
    
        self._reset()
    
        self.model.eval()
    
        with torch.no_grad():
    
            for images, targets in self.dataloader:
    
                self.evaluate_batch(
                    images,
                    targets,
                )
    
        report = self._build_report()
    
        success(
            "Evaluation completed."
        )
    
        return report


    def evaluate_batch(
        self,
        images: torch.Tensor,
        targets: list[DetectionTarget],
    ) -> None:
        """
        Evaluate a single batch.
        """
        images = images.to(
            self.device
        )

        for target in targets:
        
            target.labels = target.labels.to(
                self.device,
            )
        
            target.boxes = target.boxes.to(
                self.device,
            )

        start = time.perf_counter()

        outputs = self.model(
            features = images,
        )

        latency = time.perf_counter() - start

        self.total_latency += latency
        
        self.total_batches += 1
        
        self.total_images += images.size(
            0,
        )

        self.total_detections += (
            outputs.boxes.shape[0]
            *
            outputs.boxes.shape[1]
        )

    def _compute_metrics(
        self,
        outputs: DetectionOutput,
        targets: list[DetectionTarget],
    ) -> None:
        """
        Compute evaluation metrics for one batch.
        """
    
        loss = self._compute_loss(
            outputs,
            targets,
        )
    
        self.total_loss += loss

    def _compute_loss(
        self,
        outputs: DetectionOutput,
        targets: list[DetectionTarget],
    ) -> float:
        """
        Compute validation loss.
        """
    
        if not hasattr(
            self.model,
            "criterion",
        ):
    
            return 0.0
    
        losses = self.model.criterion(
            outputs,
            targets,
        )
    
        return float(
            losses.total.item(),
        )

    def _compute_latency(
        self,
    ) -> float:
        """
        Average inference latency.
        """
    
        if self.total_batches == 0:
    
            return 0.0
    
        return (
    
            self.total_latency
    
            / self.total_batches
    
        )

    def _compute_fps(
        self,
    ) -> float:
        """
        Compute inference FPS.
        """
    
        if self.total_latency <= 0:
    
            return 0.0
    
        return (
    
            self.total_images
    
            / self.total_latency
    
        )

    def _build_report(
        self,
    ) -> EvaluationReport:
        """
        Build evaluation report.
        """
    
        average_loss = 0.0
    
        if self.total_batches > 0:
    
            average_loss = (
    
                self.total_loss
    
                / self.total_batches
    
            )
    
        return EvaluationReport(
    
            loss=average_loss,
    
            map=0.0,
    
            precision=0.0,
    
            recall=0.0,
    
            f1_score=0.0,
    
            average_iou=0.0,
    
            fps=self._compute_fps(),
    
            latency_ms=(
                self._compute_latency()
                * 1000
            ),
    
            num_images=self.total_images,
    
            num_detections=self.total_detections,
    
        )

    def save_report(
        self,
        path: str,
        report: EvaluationReport,
    ) -> None:
        """
        Save evaluation report.
        """
    
        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
    
            json.dump(
    
                asdict(report),
    
                file,
    
                indent=4,
    
            )
    
        logger.info(
            "Evaluation report saved to %s",
            path,
        )