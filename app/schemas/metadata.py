from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ImageMetadata:

    image_id: int

    file_name: str

    width: int

    height: int

    aspect_ratio: float

    brightness: float

    contrast: float

    sharpness: float

    object_count: int

    person_count: int

    bicycle_count: int

    vehicle_count: int

    contains_vru: bool

    contains_vehicle: bool

    blur_score: float = 0.0

    edge_density: float = 0.0
