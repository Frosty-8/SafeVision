from __future__ import annotations

from pathlib import Path
import time
from app.datasets.loaders.base_loader import BaseDatasetLoader
from app.schemas.validation import ValidationReport
from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)
from rich.table import Table
from app.utils.console import console
from app.utils.progress import progress
from PIL import Image
from PIL import UnidentifiedImageError


class DatasetValidator:
    """
    Generic dataset validator.

    Responsibilities
    ----------------
    • Validate image integrity

    • Validate annotations

    • Validate categories

    • Validate bounding boxes

    • Produce validation report

    This class is dataset agnostic and works with any
    loader implementing BaseDatasetLoader.
    """

    def __init__(self, loader: BaseDatasetLoader) -> None:
        self.loader = loader
        self.report = ValidationReport()

    def validate(
        self,
    ) -> ValidationReport:
        """
        Run every validation stage
        """
        title("Dataset Validation")

        logger.info("Starting validation pipeline")

        start_time = time.perf_counter()

        self.report = ValidationReport(
            valid=True,
            total_annotations=self.loader.num_annotations,
            total_images=self.loader.num_images,
        )

        elapsed_time = time.perf_counter() - start_time

        self.report.execution_time = elapsed_time

        if self.report.failed_checks == 0:
            success("Dataset validation completed.")

            logger.info("Validation finished successfully")
        else:
            logger.warning(
                "{} validation checks failed.",
                self.report.failed_checks,
            )

        return self.report

    def _validate_images(self) -> None:
        """
        Validate all dataset images.

        Checks
        ------
        • Image exists

        • Image is readable

        • Image dimensions > 0
        """

        logger.info("Validating images...")

        images = self.loader.get_images()

        with progress:

            task = progress.add_task(
                "[cyan]Checking Images...",
                total=len(images),
            )

            for image in images:

                image_path = self.loader.get_image_path(image.id)

                self._validate_single_image(
                    image_path=image_path,
                    image=image,
                )

                progress.advance(task)

    def _validate_single_image(
        self,
        image_path: Path,
        image,
    ) -> None:
        """
        Validate a single image.
        """

        if not image_path.exists():

            self.report.missing_images += 1

            self._fail_check(f"Missing image: {image_path.name}")

            return

        try:

            with Image.open(image_path) as img:

                img.verify()

        except (
            UnidentifiedImageError,
            OSError,
        ):

            self.report.corrupted_images += 1

            self._fail_check(f"Missing image: {image_path.name}")

            return

        try:

            with Image.open(image_path) as img:

                width, height = img.size

                if width <= 0 or height <= 0:

                    self._fail_check(f"Missing image: {image_path.name}")

                    return

        except Exception:

            self._fail_check(f"Missing image: {image_path.name}")

            return

        self._pass_check()

    def _pass_check(self) -> None:
        self.report.passed_checks += 1

    def _fail_check(
        self,
        message: str | None = None,
    ) -> None:
        self.report.failed_checks += 1

        self.report.valid = False

        if message:
            self.report.errors.append(message)

            logger.error(message)

    def _warning(
        self,
        message: str,
    ) -> None:
        self.report.warnings.append(message)

        logger.warning(message)

    def _validate_annotations(self) -> None:

        logger.info("Validating annotations....")

        image_ids = set(self.loader.image_ids)

        annotation_ids: set[int] = set()

        for image in self.loader.get_images():

            annotations = self.loader.get_annotations(image.id)

            if not annotations:
                self._warning(f"{image.file_name} has no annotations.")

                continue

            for annotation in annotations:
                if annotation.id in annotation_ids:

                    self.report.duplicate_annotations += 1

                    self._fail_check(f"Duplicate annotation ID : {annotation.id}")

                    continue

                annotation_ids.add(annotation.id)

                if annotation.image_id not in image_ids:

                    self.report.invalid_annotations += 1

                    self._fail_check(
                        f"Annotation {annotation.id} "
                        f"references missing image "
                        f"{annotation.image_id}"
                    )

                    continue

                self._pass_check()

    def _validate_categories(self) -> None:
        """
        Validate category references.
        """

        logger.info("Validating categories...")

        for image in self.loader.get_images():

            for annotation in self.loader.get_annotations(image.id):

                if not self.loader.category_exists(annotation.category_id):

                    self.report.invalid_annotations += 1

                    self._fail_check(
                        f"Invalid category ID " f"{annotation.category_id}"
                    )

                    continue

                self._pass_check()

    def _validate_bounding_boxes(self) -> None:
        """
        Validate all bounding boxes.
        """

        logger.info("Validating bounding boxes...")

        for image in self.loader.get_images():

            image_width = image.width
            image_height = image.height

            annotations = self.loader.get_annotations(image.id)

            for annotation in annotations:

                x, y, w, h = annotation.bbox

                #
                # Width / Height
                #

                if w <= 0 or h <= 0:

                    self.report.invalid_bounding_boxes += 1

                    self._fail_check(
                        f"Annotation {annotation.id} " "has non-positive width/height."
                    )

                    continue

                #
                # Negative Coordinates
                #

                if x < 0 or y < 0:

                    self.report.invalid_bounding_boxes += 1

                    self._fail_check(
                        f"Annotation {annotation.id} " "has negative coordinates."
                    )

                    continue

                #
                # Outside Image
                #

                if x + w > image_width:

                    self.report.invalid_bounding_boxes += 1

                    self._fail_check(
                        f"Annotation {annotation.id} " "extends beyond image width."
                    )

                    continue

                if y + h > image_height:

                    self.report.invalid_bounding_boxes += 1

                    self._fail_check(
                        f"Annotation {annotation.id} " "extends beyond image height."
                    )

                    continue

                self._pass_check()

    def _validate_dataset(self) -> None:
        """
        Validate overall dataset integrity.
        """

        logger.info("Validating dataset integrity...")

        image_ids = set(self.loader.image_ids)

        for image in self.loader.get_images():

            annotations = self.loader.get_annotations(image.id)

            if not annotations:

                self._warning(f"{image.file_name} contains no objects.")

        for annotation_id in self.loader.annotation_ids:

            annotation = self.loader.get_annotation(annotation_id)

            if annotation is None:

                self.report.invalid_annotations += 1

                self._fail_check(f"Annotation {annotation_id} " "could not be loaded.")

                continue

            if annotation.image_id not in image_ids:

                self.report.invalid_annotations += 1

                self._fail_check(
                    f"Annotation {annotation.id} " "references missing image."
                )

                continue

            self._pass_check()

    def print_summary(self) -> None:
        """
        Print validation summary.
        """

        table = Table(
            title="Validation Summary",
            header_style="bold green",
        )

        table.add_column("Metric")
        table.add_column("Value", justify="right")

        table.add_row(
            "Images",
            str(self.report.total_images),
        )

        table.add_row(
            "Annotations",
            str(self.report.total_annotations),
        )

        table.add_row(
            "Missing Images",
            str(self.report.missing_images),
        )

        table.add_row(
            "Corrupted Images",
            str(self.report.corrupted_images),
        )

        table.add_row(
            "Invalid Boxes",
            str(self.report.invalid_bounding_boxes),
        )

        table.add_row(
            "Invalid Annotations",
            str(self.report.invalid_annotations),
        )

        table.add_row(
            "Duplicate Annotations",
            str(self.report.duplicate_annotations),
        )

        table.add_row(
            "Passed Checks",
            str(self.report.passed_checks),
        )

        table.add_row(
            "Failed Checks",
            str(self.report.failed_checks),
        )

        table.add_row(
            "Execution Time",
            f"{self.report.execution_time:.2f}s",
        )

        table.add_row(
            "Status",
            "PASS" if self.report.valid else "FAIL",
        )

        console.print(table)
