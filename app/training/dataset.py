from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.datasets.loaders.base_loader import BaseDatasetLoader
from app.utils.logger import logger
from typing import cast

class SafeVisionDataset(Dataset):
    """
    Pytorch Dataset for SafeVision

    Supports object detection datasets using the BaseDatasetLoader interface.
    """

    def __init__(self,
        loader: BaseDatasetLoader,
        transforms: Any | None = None
    ) -> None:
        self.loader = loader

        self.transforms = transforms

        self.image_ids = loader.image_ids

        logger.info(
            "Initialized SafeVisionDataset with {} image",
            len(self.image_ids)
        )


    def __len__(
        self,
    ) -> int:
        """
        Return dataset size.
        """
    
        return len(
            self.image_ids,
        )

    def __getitem__(
        self,
        index: int,
    ):
        """
        Return one training sample.
        """
    
        image_id = self.image_ids[index]
    
        image = self._load_image(
            image_id,
        )
    
        target = self._load_target(
            image_id,
        )
    
        if self.transforms is not None:
    
            image, target = self.transforms(
                image,
                target,
            )
    
        image = self._to_tensor(
            image,
        )
    
        target = self._to_tensor_target(
            target,
        )
    
        return image, target



    def _load_image(
        self,
        image_id: int,
    ) -> np.ndarray:
        """
        Load image from disk.
        """
    
        image_path = self.loader.get_image_path(
            image_id,
        )
    
        image = cv2.imread(
            str(image_path),
        )
    
        if image is None:
    
            raise FileNotFoundError(
                image_path,
            )
    
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )
    
        return image


    def _load_target(
        self,
        image_id: int,
    ) -> dict[str, Any]:
        """
        Load annotations for one image.
        """
    
        annotations = self.loader.get_annotations(
            image_id,
        )
    
        boxes = []
        labels = []
    
        for annotation in annotations:
    
            x, y, w, h = annotation.bbox
    
            boxes.append(
                [
                    x,
                    y,
                    x + w,
                    y + h,
                ]
            )
    
            labels.append(
                annotation.category_id,
            )
    
        return {
            "boxes": np.asarray(
                boxes,
                dtype=np.float32,
            ),
            "labels": np.asarray(
                labels,
                dtype=np.int64,
            ),
            "image_id": image_id,
        }


    def _to_tensor(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert an HWC NumPy image to a CHW float tensor in the range [0, 1].
        """
    
        tensor = cast(torch.Tensor, torch.from_numpy(image))
    
        tensor = tensor.permute(
            2,
            0,
            1,
        )
    
        return tensor.float() / 255.0

    def _to_tensor_target(
        self,
        target: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert targets to tensors.
        """
    
        target["boxes"] = torch.as_tensor(
            target["boxes"],
            dtype=torch.float32,
        )
    
        target["labels"] = torch.as_tensor(
            target["labels"],
            dtype=torch.int64,
        )
    
        target["image_id"] = torch.tensor(
            target["image_id"],
            dtype=torch.int64,
        )
    
        return target