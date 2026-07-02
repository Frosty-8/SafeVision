from dataclasses import dataclass


@dataclass(slots=True)
class DatasetInfo:

    dataset_name: str

    root_directory: str

    image_directory: str

    annotation_file: str

    total_images: int

    total_annotations: int