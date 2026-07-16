from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.inference.confidence_calibration import (
    ConfidenceCalibrator,
)

from app.inference.nms import NMS

from app.inference.postprocessing import (
    PostProcessor,
)

from app.inference.predictor import (
    Predictor,
)

from app.inference.visualization import (
    Visualizer,
)

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title
)

class InferencePipeline:
    """
    Complete inference pipeline
    """

    def __init__(
        self,
        predictor: Predictor,
        postprocessor: PostProcessor | None = None,
        nms: NMS | None = None,
        calibrator: ConfidenceCalibrator | None = None,
        visualizer: Visualizer | None = None,
    ) -> None:
    
        self.predictor = predictor
    
        self.postprocessor = (
            postprocessor
            if postprocessor is not None
            else PostProcessor()
        )
    
        self.nms = (
            nms
            if nms is not None
            else NMS()
        )
    
        self.calibrator = (
            calibrator
            if calibrator is not None
            else ConfidenceCalibrator()
        )
    
        self.visualizer = (
            visualizer
            if visualizer is not None
            else Visualizer()
        )
    
        logger.info(
            "InferencePipeline initialized."
        )

    def predict(
        self,
        image: Image.Image,
    ) -> DetectionOutput:
        """
        Run complete inference.
        """
    
        outputs = self.predictor.predict(
            image,
        )
    
        outputs = self.postprocessor(
            outputs,
        )
    
        outputs = self.nms(
            outputs,
        )
    
        outputs = self.calibrator(
            outputs,
        )
    
        return outputs


    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> DetectionOutput:
        """
        Batch inference.
        """
    
        outputs = self.predictor.predict_batch(
            images,
        )
    
        outputs = self.postprocessor(
            outputs,
        )
    
        outputs = self.nms(
            outputs,
        )
    
        outputs = self.calibrator(
            outputs,
        )
    
        return outputs

    def predict_directory(
        self,
        directory: str | Path,
    ) -> list[DetectionOutput]:
        """
        Predict an entire directory.
        """
    
        results = []
    
        for output in self.predictor.predict_directory(
            directory,
        ):
    
            output = self.postprocessor(
                output,
            )
    
            output = self.nms(
                output,
            )
    
            output = self.calibrator(
                output,
            )
    
            results.append(
                output,
            )
    
        return results

    def visualize(
        self,
        image: np.ndarray,
        outputs: DetectionOutput,
        class_names: list[str],
    ) -> np.ndarray:
        """
        Draw predictions.
        """
    
        return self.visualizer.draw(
    
            image,
    
            outputs,
    
            class_names,
    
        )

    def __call__(
        self,
        image: Image.Image,
    ) -> DetectionOutput:
        """
        Callable pipeline.
        """
    
        return self.predict(
            image,
        )

    def summary(
        self,
    ) -> None:
        """
        Display pipeline configuration.
        """
    
        title(
            "Inference Pipeline",
        )
    
        logger.info(
            "Predictor : %s",
            type(self.predictor).__name__,
        )
    
        logger.info(
            "PostProcessor : %s",
            type(self.postprocessor).__name__,
        )
    
        logger.info(
            "NMS : %s",
            type(self.nms).__name__,
        )
    
        logger.info(
            "Calibration : %s",
            type(self.calibrator).__name__,
        )
    
        logger.info(
            "Visualizer : %s",
            type(self.visualizer).__name__,
        )
    
        success(
            "Pipeline ready."
        )