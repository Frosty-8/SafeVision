"""
SafeVision AI

Metadata Generator

Extracts image-level metadata for downstream analytics,
training, safety analysis, and reporting.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from app.datasets.loaders.base_loader import BaseDatasetLoader
from app.schemas.metadata import ImageMetadata

from app.utils.logger import logger
from app.utils.ui import success, title
from app.utils.progress import progress

import json

from dataclasses import asdict

from rich.table import Table

from app.utils.console import console


class MetadataGenerator:
    """
    Generates metadata for every image.

    Responsibilities
    ----------------

    • Brightness

    • Contrast

    • Sharpness

    • Object Counts

    • Aspect Ratio

    • Export CSV / JSON
    """

    def __init__(self, loader: BaseDatasetLoader) -> None:
        self.loader = loader

        self.metadata: list[ImageMetadata] = []

    def generate(self) -> list[ImageMetadata]:
        """
        Generate metadata for every image
        """
        title("Generating Image Metadata")

        logger.info("Starting metadata generation...")

        self.metadata.clear()

        images = self.loader.get_images()

        with progress:

            task = progress.add_task("[green]Processing Images.....", total=len(images))

            for image in images:

                metadata = self._process_image(
                    image.id,
                )

                self.metadata.append(
                    metadata,
                )

                progress.advance(task)

            success(f"Generated metadata for " f"{len(self.metadata)} images.")

            logger.info("Metadata generation completed")

            return self.metadata

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return metadata as DataFrame.
        """

        return self._build_dataframe()

    def _process_image(self, image_id: int) -> ImageMetadata:
        """
        Process a single image and generate metadata
        """

        image = self.loader.get_image(image_id)

        if image is None:
            raise ValueError(f"Image {image_id} not found.")

        image_path = self.loader.get_image_path(image_id)

        frame = self._load_image(image_path)

        brightness = self._compute_brightness(frame)

        contrast = self._compute_contrast(frame)

        sharpness = self._compute_sharpness(frame)

        (
            object_count,
            person_count,
            bicycle_count,
            vehicle_count,
            contains_vru,
            contains_vehicle,
        ) = self._compute_object_counts(image_id)

        return self._build_metadata(
            image=image,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            object_count=object_count,
            person_count=person_count,
            bicycle_count=bicycle_count,
            vehicle_count=vehicle_count,
            contains_vru=contains_vru,
            contains_vehicle=contains_vehicle,
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load an image using OpenCV
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Unable to load image :\n{image_path}")

        return image

    def _compute_brightness(self, image: np.ndarray) -> float:
        """
        Compute average brightness
        """
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        return float(np.mean(gray))

    def _compute_contrast(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute image contrast.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        return float(np.std(gray))

    def _compute_sharpness(
        self,
        image: np.ndarray,
    ) -> float:
        """
        Compute image sharpness using
        Laplacian variance.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        return float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

    def _compute_object_counts(
        self,
        image_id: int,
    ) -> tuple[
        int,
        int,
        int,
        int,
        bool,
        bool,
    ]:
        """
        Compute object statistics.
        """

        annotations = self.loader.get_annotations(image_id)

        object_count = len(annotations)

        person_count = 0

        bicycle_count = 0

        vehicle_count = 0

        for annotation in annotations:

            category = self.loader.get_category_name(annotation.category_id)

            if category == "person":

                person_count += 1

            elif category == "bicycle":

                bicycle_count += 1

            elif category in {
                "car",
                "bus",
                "truck",
                "motorcycle",
            }:

                vehicle_count += 1

        contains_vru = person_count > 0 or bicycle_count > 0

        contains_vehicle = vehicle_count > 0

        return (
            object_count,
            person_count,
            bicycle_count,
            vehicle_count,
            contains_vru,
            contains_vehicle,
        )

    def _build_metadata(
        self,
        *,
        image,
        brightness: float,
        contrast: float,
        sharpness: float,
        object_count: int,
        person_count: int,
        bicycle_count: int,
        vehicle_count: int,
        contains_vru: bool,
        contains_vehicle: bool,
    ) -> ImageMetadata:
        """
        Build ImageMetadata object.
        """

        return ImageMetadata(
            image_id=image.id,
            file_name=image.file_name,
            width=image.width,
            height=image.height,
            aspect_ratio=image.width / image.height,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            object_count=object_count,
            person_count=person_count,
            bicycle_count=bicycle_count,
            vehicle_count=vehicle_count,
            contains_vru=contains_vru,
            contains_vehicle=contains_vehicle,
        )

    def _build_dataframe(self) -> pd.DataFrame:
        """
        Convert metadata objects into a pandas DataFrame.
        """

        logger.info("Building metadata DataFrame...")

        return pd.DataFrame([asdict(item) for item in self.metadata])

    def _save_csv(
        self,
        dataframe: pd.DataFrame,
        output_directory: Path,
    ) -> None:
        """
        Save metadata as CSV.
        """

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_path = output_directory / "image_metadata.csv"

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        logger.info(
            "CSV saved to {}",
            csv_path,
        )

    def _save_json(
        self,
        output_directory: Path,
    ) -> None:
        """
        Save metadata as JSON.
        """

        json_path = output_directory / "image_metadata.json"

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                [asdict(item) for item in self.metadata],
                file,
                indent=4,
            )

        logger.info(
            "JSON saved to {}",
            json_path,
        )

    def export(
        self,
        output_directory: str | Path,
    ) -> None:
        """
        Export metadata.
        """

        title("Exporting Metadata")

        output_directory = Path(output_directory)

        dataframe = self._build_dataframe()

        self._save_csv(
            dataframe,
            output_directory,
        )

        self._save_json(
            output_directory,
        )

        self._summary(
            dataframe,
        )

        success("Metadata exported successfully.")

    def _summary(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Display metadata summary.
        """

        table = Table(
            title="Metadata Summary",
            header_style="bold cyan",
        )

        table.add_column("Metric")
        table.add_column(
            "Value",
            justify="right",
        )

        table.add_row(
            "Images",
            str(len(dataframe)),
        )

        table.add_row(
            "Average Brightness",
            f"{dataframe['brightness'].mean():.2f}",
        )

        table.add_row(
            "Average Contrast",
            f"{dataframe['contrast'].mean():.2f}",
        )

        table.add_row(
            "Average Sharpness",
            f"{dataframe['sharpness'].mean():.2f}",
        )

        table.add_row(
            "Average Objects",
            f"{dataframe['object_count'].mean():.2f}",
        )

        table.add_row(
            "Images With VRUs",
            str(dataframe["contains_vru"].sum()),
        )

        table.add_row(
            "Images With Vehicles",
            str(dataframe["contains_vehicle"].sum()),
        )

        console.print(table)
