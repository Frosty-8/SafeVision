from dataclasses import dataclass, field

@dataclass(slots=True)
class DatasetStatistics:

    total_images: int

    total_annotations: int

    total_classes: int

    average_objects_per_image: float

    average_width: float

    average_height: float

    average_brightness: float
    
    average_contrast: float
    
    average_sharpness: float