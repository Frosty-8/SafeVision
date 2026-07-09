from dataclasses import dataclass


@dataclass(slots=True)
class DatasetStatistics:

    total_images: int

    total_objects: int

    average_objects_per_image: float

    average_width: float

    average_height: float

    average_aspect_ratio: float

    average_brightness: float

    average_contrast: float

    average_sharpness: float

    total_persons: int

    total_bicycles: int

    total_vehicles: int

    images_with_vru: int

    images_with_vehicle: int
