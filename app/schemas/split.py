from dataclasses import dataclass

from app.schemas.dataset import CocoAnnotation
from app.schemas.dataset import CocoCategory
from app.schemas.dataset import CocoImage


@dataclass(slots=True)
class DatasetSplit:

    images: list[CocoImage] = []

    annotations: list[CocoAnnotation] = []

    categories: list[CocoCategory] = []


@dataclass(slots=True)
class SplitReport:

    train_images: int

    validation_images: int

    test_images: int

    train_annotations: int

    validation_annotations: int

    test_annotations: int
