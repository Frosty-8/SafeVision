from dataclasses import dataclass, field


@dataclass(slots=True)
class ValidationReport:

    valid: bool = True

    total_images: int = 0

    total_annotations: int = 0

    missing_images: int = 0

    corrupted_images: int = 0

    invalid_annotations: int = 0

    invalid_bounding_boxes: int = 0

    duplicate_annotations: int = 0

    passed_checks: int = 0

    failed_checks: int = 0

    execution_time: float = 0.0

    warnings: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
