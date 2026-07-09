"""
SafeVision AI

Abstract Dataset Loader

Defines the common interface for all dataset loaders.

Every dataset implementation (COCO, KITTI, Waymo, nuScenes, etc.)
must inherit from this class.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from collections.abc import Iterator


class BaseDatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.

    Responsibilities
    ----------------
    - Load dataset
    - Provide dataset statistics
    - Expose image/annotation access
    - Maintain a common interface

    Child classes are responsible for implementing
    dataset-specific parsing logic.
    """

    def __init__(
        self,
        image_directory: str | Path,
        annotation_file: str | Path,
    ) -> None:

        self.image_directory = Path(image_directory)

        self.annotation_file = Path(annotation_file)

    # ---------------------------------------------------------
    # Loading
    # ---------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Load the dataset into memory.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    @abstractmethod
    def summary(self) -> None:
        """
        Display dataset summary.
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Image API
    # ---------------------------------------------------------

    @abstractmethod
    def get_image(
        self,
        image_id: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_images(self):
        raise NotImplementedError

    @abstractmethod
    def get_image_path(
        self,
        image_id: int,
    ) -> Path:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Annotation API
    # ---------------------------------------------------------

    @abstractmethod
    def get_annotation(
        self,
        annotation_id: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_annotations(
        self,
        image_id: int,
    ):
        raise NotImplementedError

    # ---------------------------------------------------------
    # Category API
    # ---------------------------------------------------------

    @abstractmethod
    def get_category(
        self,
        category_id: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_category_name(
        self,
        category_id: int,
    ) -> str | None:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Existence Checks
    # ---------------------------------------------------------

    @abstractmethod
    def image_exists(
        self,
        image_id: int,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def annotation_exists(
        self,
        annotation_id: int,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def category_exists(
        self,
        category_id: int,
    ) -> bool:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    @property
    @abstractmethod
    def num_images(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_annotations(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_categories(self) -> int:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Python API
    # ---------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Dataset IDs
    # ---------------------------------------------------------

    @property
    @abstractmethod
    def image_ids(self) -> tuple[int, ...]:
        """
        Return all image IDs.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def annotation_ids(self) -> tuple[int, ...]:
        """
        Return all annotation IDs.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def category_ids(self) -> tuple[int, ...]:
        """
        Return all category IDs.
        """
        raise NotImplementedError
