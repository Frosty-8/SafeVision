from dataclasses import dataclass


@dataclass(slots=True)
class DatasetSplit:

    train_images: list[int]

    validation_images: list[int]

    test_images: list[int]
