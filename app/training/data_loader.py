from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.training.collate import detection_collate_fn
from app.training.dataset import SafeVisionDataset
from app.utils.logger import logger

class DataLoaderFactory:

    """
    Factory for creating Pytorch DataLoaders.
    """
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        seed: int = 42,
    ) -> None:
    
        self.batch_size = batch_size
    
        self.num_workers = num_workers
    
        self.pin_memory = pin_memory
    
        self.persistent_workers = persistent_workers
    
        self.prefetch_factor = prefetch_factor
    
        self.seed = seed
    
        logger.info(
            "Initialized DataLoaderFactory."
        )

    def _worker_init_fn(
        self,
        worker_id: int,
    ) -> None:
        """
        Initialize worker seed.
        """
    
        worker_seed = (
            self.seed + worker_id
        )
    
        random.seed(worker_seed)
    
        np.random.seed(worker_seed)
    
        torch.manual_seed(worker_seed)

    def _create_loader(
        self,
        dataset: SafeVisionDataset,
        *,
        shuffle: bool,
    ) -> DataLoader:
        """
        Create DataLoader.
        """
    
        return DataLoader(
            dataset=dataset,
    
            batch_size=self.batch_size,
    
            shuffle=shuffle,
    
            num_workers=self.num_workers,
    
            pin_memory=self.pin_memory,
    
            persistent_workers=(
                self.persistent_workers
                if self.num_workers > 0
                else False
            ),
    
            prefetch_factor=(
                self.prefetch_factor
                if self.num_workers > 0
                else None
            ),
    
            collate_fn=detection_collate_fn,
    
            worker_init_fn=self._worker_init_fn,
        )

    def create_train_loader(
        self,
        dataset: SafeVisionDataset,
    ) -> DataLoader:
        """
        Create training DataLoader.
        """
    
        logger.info(
            "Creating training DataLoader."
        )
    
        return self._create_loader(
            dataset,
            shuffle=True,
        )

    def create_validation_loader(
        self,
        dataset: SafeVisionDataset,
    ) -> DataLoader:
        """
        Create validation DataLoader.
        """
    
        logger.info(
            "Creating validation DataLoader."
        )
    
        return self._create_loader(
            dataset,
            shuffle=False,
        )

    def create_test_loader(
        self,
        dataset: SafeVisionDataset,
    ) -> DataLoader:
        """
        Create test DataLoader.
        """
    
        logger.info(
            "Creating test DataLoader."
        )
    
        return self._create_loader(
            dataset,
            shuffle=False,
        )