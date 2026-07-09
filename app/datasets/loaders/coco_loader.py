from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict
from collections.abc import Iterator
from pycocotools.coco import COCO
from app.datasets.loaders.base_loader import BaseDatasetLoader

from app.schemas.coco import CocoAnnotation, CocoImage, CocoCategory

from app.utils.logger import logger
from app.utils.ui import success, title

from rich.table import Table

from app.utils.console import console

from app.core.exceptions import DatasetNotFoundError, DatasetNotLoadedError


class CocoLoader(BaseDatasetLoader):
    """
    Production-ready COCO dataset loader.

    Responsibilities
    ----------------
    - Load COCO dataset
    - Convert dictionaries into dataclasses
    - Build lookup tables
    - Provide fast access methods

    This class SHOULD NOT:

    ❌ Validate data
    ❌ Split datasets
    ❌ Compute metadata
    ❌ Perform augmentations

    Those responsibilities belong to other modules.
    """

    def __init__(
        self, image_directory: str | Path, annotation_file: str | Path
    ) -> None:
        super().__init__(image_directory, annotation_file)
        self.image_directory = Path(image_directory)
        self.annotation_file = Path(annotation_file)

        self._images: dict[int, CocoImage] = {}

        self._annotations: dict[int, CocoAnnotation] = {}

        self._categories: dict[int, CocoCategory] = {}

        self._image_annotations: DefaultDict[int, list[CocoAnnotation]] = defaultdict(
            list
        )

        self._category_lookup: dict[int, str] = {}

        self._loaded = False

        self._coco: COCO | None = None

    def _load_categories(self) -> None:

        logger.info("Loading categories...")

        assert self._coco is not None

        for category in self._coco.loadCats(self._coco.getCatIds()):

            obj = CocoCategory(
                id=category["id"],
                name=category["name"],
                supercategory=category["supercategory"],
            )

            self._categories[obj.id] = obj

    def _load_images(self) -> None:

        logger.info("Loading images...")

        assert self._coco is not None

        for image in self._coco.loadImgs(self._coco.getImgIds()):

            obj = CocoImage(
                id=image["id"],
                file_name=image["file_name"],
                width=image["width"],
                height=image["height"],
            )

            self._images[obj.id] = obj

    def _load_annotations(self) -> None:

        logger.info("Loading annotations...")

        assert self._coco is not None

        for annotation in self._coco.loadAnns(self._coco.getAnnIds()):

            obj = CocoAnnotation(
                id=annotation["id"],
                image_id=annotation["image_id"],
                category_id=annotation["category_id"],
                bbox=annotation["bbox"],
                area=annotation["area"],
                iscrowd=annotation["iscrowd"],
            )

            self._annotations[obj.id] = obj

    def _build_indexes(self) -> None:

        logger.info("Building lookup tables...")

        for annotation in self._annotations.values():

            self._image_annotations[annotation.image_id].append(annotation)

        self._category_lookup = {
            category.id: category.name for category in self._categories.values()
        }

        logger.info("Lookup tables created.")

    def _ensure_loaded(self) -> None:
        """
        Ensure that the dataset has been loaded.
        """

        if not self._loaded:
            raise DatasetNotLoadedError(
                "Dataset has not been loaded. " "Call 'load()' before accessing data."
            )

    def get_image(
        self,
        image_id: int,
    ) -> CocoImage | None:

        self._ensure_loaded()

        return self._images.get(image_id)

    def get_images(
        self,
    ) -> list[CocoImage]:

        self._ensure_loaded()

        return list(self._images.values())

    def get_annotation(
        self,
        annotation_id: int,
    ) -> CocoAnnotation | None:

        self._ensure_loaded()

        return self._annotations.get(annotation_id)

    def get_annotations(
        self,
        image_id: int,
    ) -> list[CocoAnnotation]:

        self._ensure_loaded()

        return self._image_annotations.get(
            image_id,
            [],
        )

    def get_category(
        self,
        category_id: int,
    ) -> CocoCategory | None:

        self._ensure_loaded()

        return self._categories.get(category_id)

    def get_category_name(
        self,
        category_id: int,
    ) -> str | None:

        self._ensure_loaded()

        return self._category_lookup.get(category_id)

    def image_exists(
        self,
        image_id: int,
    ) -> bool:

        self._ensure_loaded()

        return image_id in self._images

    def annotation_exists(
        self,
        annotation_id: int,
    ) -> bool:

        self._ensure_loaded()

        return annotation_id in self._annotations

    def category_exists(
        self,
        category_id: int,
    ) -> bool:

        self._ensure_loaded()

        return category_id in self._categories

    @property
    def num_images(self) -> int:

        return len(self._images)

    @property
    def num_annotations(self) -> int:

        return len(self._annotations)

    @property
    def num_categories(self) -> int:

        return len(self._categories)

    @property
    def images(self) -> tuple[CocoImage, ...]:

        return tuple(self._images.values())

    @property
    def annotations(
        self,
    ) -> tuple[CocoAnnotation, ...]:

        return tuple(self._annotations.values())

    @property
    def categories(
        self,
    ) -> tuple[CocoCategory, ...]:

        return tuple(self._categories.values())

    def get_image_path(
        self,
        image_id: int,
    ) -> Path:

        image = self.get_image(image_id)

        if image is None:

            raise FileNotFoundError(f"Image {image_id} not found.")

        return self.image_directory / image.file_name

    def get_category_distribution(
        self,
    ) -> dict[str, int]:

        distribution: dict[str, int] = {}

        for annotation in self._annotations.values():

            name = self.get_category_name(annotation.category_id)

            if name is None:
                continue

            distribution[name] = distribution.get(name, 0) + 1

        return distribution

    def summary(self) -> None:
        """
        Display a dataset summary.
        """

        self._ensure_loaded()

        table = Table(
            title="COCO Dataset Summary",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Metric")
        table.add_column("Value", justify="right")

        table.add_row("Images", str(self.num_images))
        table.add_row("Annotations", str(self.num_annotations))
        table.add_row("Categories", str(self.num_categories))

        table.add_row(
            "Image Directory",
            str(self.image_directory),
        )

        table.add_row(
            "Annotation File",
            str(self.annotation_file),
        )

        console.print(table)

    @property
    def image_ids(self) -> tuple[int, ...]:
        return tuple(self._images.keys())

    @property
    def annotation_ids(self) -> tuple[int, ...]:
        return tuple(self._annotations.keys())

    @property
    def category_ids(self) -> tuple[int, ...]:
        return tuple(self._categories.keys())

    def __len__(self) -> int:
        """
        Return total number of images.
        """

        return self.num_images

    def __repr__(self) -> str:

        return (
            f"CocoLoader("
            f"images={self.num_images}, "
            f"annotations={self.num_annotations}, "
            f"categories={self.num_categories})"
        )

    def __iter__(self) -> Iterator[CocoImage]:

        yield from self._images.values()

    def get_images_by_category(
        self,
        category_name: str,
    ) -> list[CocoImage]:
        """
        Return all images containing a category.
        """

        self._ensure_loaded()

        category_ids = [
            category.id
            for category in self._categories.values()
            if category.name == category_name
        ]

        if not category_ids:
            return []

        image_ids = set()

        for annotation in self._annotations.values():

            if annotation.category_id in category_ids:

                image_ids.add(annotation.image_id)

        return [self._images[image_id] for image_id in image_ids]

    def category_statistics(self) -> None:

        table = Table(
            title="Category Distribution",
            header_style="bold magenta",
        )

        table.add_column("Category")

        table.add_column(
            "Annotations",
            justify="right",
        )

        distribution = self.get_category_distribution()

        for category, count in sorted(distribution.items()):
            table.add_row(
                category,
                str(count),
            )

        console.print(table)

    def image_has_category(
        self,
        image_id: int,
        category_name: str,
    ) -> bool:
        """
        Check if an image contains a category.
        """

        annotations = self.get_annotations(image_id)

        for annotation in annotations:

            if self.get_category_name(annotation.category_id) == category_name:
                return True

        return False

    """
    Load COCO dataset into memory
    """

    def load(self) -> None:
        """
        Load COCO dataset into memory.
        """

        title("Loading COCO Dataset")

        if not self.annotation_file.exists():
            raise DatasetNotFoundError(
                f"Annotation file not found:\n{self.annotation_file}"
            )

        if not self.image_directory.exists():
            raise DatasetNotFoundError(
                f"Image directory not found:\n{self.image_directory}"
            )

        logger.info("Loading annotation file...")

        self._coco = COCO(str(self.annotation_file))

        self._load_categories()

        self._load_images()

        self._load_annotations()

        self._build_indexes()

        self._loaded = True

        success("COCO dataset loaded successfully.")

        logger.info(
            "Images: {} | Annotations: {} | Categories: {}",
            len(self._images),
            len(self._annotations),
            len(self._categories),
        )
