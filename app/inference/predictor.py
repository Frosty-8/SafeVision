from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Compose
from typing import cast
from app.inference.postprocessing import (
    PostProcessor,
)

from app.models.detector.deformable_detr import (
    DeformableDETR,
)

from app.schemas.detection import (
    DetectionOutput,
)

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)

class Predictor:
    """
    SafeVision inference pipeline.
    """
    def __init__(
        self,
        model: DeformableDETR,
        transforms: Compose,
        device: torch.device,
        postprocessor: PostProcessor | None = None
    ) -> None:
    
        self.model = model
    
        self.transforms = transforms
    
        self.device = device
    
        self.postprocessor = (
        
            postprocessor
        
            if postprocessor is not None
        
            else PostProcessor()
        
        )
    
        self.model.to(device)
    
        self.model.eval()
    
        logger.info(
            "Predictor initialized."
        )

    def load_checkpoint(
        self,
        checkpoint: str | Path,
    ) -> None:
        """
        Load trained weights.
        """
    
        state = torch.load(
    
            checkpoint,
    
            map_location=self.device,
    
        )
    
        self.model.load_state_dict(
    
            state["model"],
    
        )
    
        success(
            "Checkpoint loaded."
        )

    def preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Prepare image for inference.
        """
    
        tensor = cast(torch.Tensor, self.transforms(image))
    
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image: Image.Image,
    ) -> DetectionOutput:
        """
        Predict one image.
        """
    
        tensor = self.preprocess(
            image,
        )
    
        with torch.no_grad():
    
            outputs = self.model(
                features=tensor,
            )
    
        return self.postprocess(
            outputs,
        )

    def postprocess(
        self,
        outputs: DetectionOutput,
    ) -> DetectionOutput:
        """
        Apply inference post-processing.
        """
    
        return self.postprocessor(
            outputs,
        )


    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> DetectionOutput:
        """
        Batch inference.
        """
    
        images = images.to(
            self.device,
        )
    
        with torch.no_grad():
    
            outputs = self.model(
                features=images,
            )
    
        return self.postprocess(
            outputs,
        )

    def predict_directory(
        self,
        directory: str | Path,
    ) -> list[DetectionOutput]:
        """
        Predict every image inside
        a directory.
        """
    
        outputs = []
    
        directory = Path(
            directory,
        )
    
        for image_path in directory.iterdir():
    
            if image_path.suffix.lower() not in {
    
                ".jpg",
    
                ".jpeg",
    
                ".png",
    
            }:
    
                continue
    
            image = Image.open(
                image_path,
            ).convert(
                "RGB",
            )
    
            outputs.append(
    
                self.predict(
                    image,
                )
    
            )
    
        return outputs

    def summary(
        self,
    ) -> None:
        """
        Display predictor information.
        """
    
        title(
            "Predictor Summary",
        )
    
        logger.info(
            "Device : %s",
            self.device,
        )
    
        logger.info(
            "Model : %s",
            type(self.model).__name__,
        )
    
        logger.info(
            "PostProcessor : %s",
            type(self.postprocessor).__name__,
        )