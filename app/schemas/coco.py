from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CocoCategory:
    id: int
    name: str
    supercategory: str


@dataclass(slots=True, frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(slots=True, frozen=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    iscrowd: int
