"""
SafeVision AI

Dataset Statistics Generator

Computes dataset-level statistics from image metadata.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rich.table import Table

from app.schemas.metadata import ImageMetadata
from app.schemas.statistics import DatasetStatistics

from app.utils.console import console
from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)


class StatisticsGenerator:
    """
    Generates dataset statistics from metadata
    """

    def __init__(self, metadata: list[ImageMetadata]) -> None:
        self.metadata = metadata

        self.statistics: DatasetStatistics | None = None

    def generate(self) -> DatasetStatistics:
        """
        Generate dataset statistics.
        """

        title("Generating Dataset Statistics")

        logger.info("Computing dataset statistics...")

        dataframe = self._build_dataframe()

        self.statistics = self._compute_statistics(
            dataframe,
        )

        success("Statistics generated successfully.")

        return self.statistics

    def _build_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Convert metadata into DataFrame.
        """

        return pd.DataFrame([asdict(item) for item in self.metadata])

    def _compute_statistics(
        self,
        dataframe: pd.DataFrame,
    ) -> DatasetStatistics:
        """
        Compute dataset-level statistics.

        Parameters
        ----------
        dataframe:
            Metadata dataframe.

        Returns
        -------
        DatasetStatistics
        """

        logger.info("Computing aggregate statistics...")

        if dataframe.empty:
            raise ValueError("Cannot compute statistics from an empty dataframe.")

        # -------------------------------
        # Image Statistics
        # -------------------------------

        total_images = len(dataframe)

        average_width = float(dataframe["width"].mean())

        average_height = float(dataframe["height"].mean())

        average_aspect_ratio = float(dataframe["aspect_ratio"].mean())

        # -------------------------------
        # Object Statistics
        # -------------------------------

        total_objects = int(dataframe["object_count"].sum())

        average_objects_per_image = float(dataframe["object_count"].mean())

        # -------------------------------
        # Image Quality
        # -------------------------------

        average_brightness = float(dataframe["brightness"].mean())

        average_contrast = float(dataframe["contrast"].mean())

        average_sharpness = float(dataframe["sharpness"].mean())

        # -------------------------------
        # Class Counts
        # -------------------------------

        total_persons = int(dataframe["person_count"].sum())

        total_bicycles = int(dataframe["bicycle_count"].sum())

        total_vehicles = int(dataframe["vehicle_count"].sum())

        # -------------------------------
        # Presence Statistics
        # -------------------------------

        images_with_vru = int(dataframe["contains_vru"].sum())

        images_with_vehicle = int(dataframe["contains_vehicle"].sum())

        logger.info("Statistics computed successfully.")

        return DatasetStatistics(
            total_images=total_images,
            total_objects=total_objects,
            average_objects_per_image=average_objects_per_image,
            average_width=average_width,
            average_height=average_height,
            average_aspect_ratio=average_aspect_ratio,
            average_brightness=average_brightness,
            average_contrast=average_contrast,
            average_sharpness=average_sharpness,
            total_persons=total_persons,
            total_bicycles=total_bicycles,
            total_vehicles=total_vehicles,
            images_with_vru=images_with_vru,
            images_with_vehicle=images_with_vehicle,
        )

    def _save_json(
        self,
        output_directory: Path,
    ) -> None:
        """
        Save statistics as JSON.
        """

        if self.statistics is None:
            raise RuntimeError("Statistics have not been generated.")

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = output_directory / "dataset_statistics.json"

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(self.statistics),
                file,
                indent=4,
            )

        logger.info(
            "Statistics saved to {}",
            json_path,
        )

    def _save_csv(
        self,
        output_directory: Path,
    ) -> None:
        """
        Save statistics as CSV.
        """

        if self.statistics is None:
            raise RuntimeError("Statistics have not been generated.")

        dataframe = pd.DataFrame([asdict(self.statistics)])

        csv_path = output_directory / "dataset_statistics.csv"

        dataframe.to_csv(
            csv_path,
            index=False,
        )

        logger.info(
            "Statistics saved to {}",
            csv_path,
        )

    def export(
        self,
        output_directory: str | Path,
    ) -> None:
        """
        Export statistics.
        """

        title("Exporting Statistics")

        output_directory = Path(output_directory)

        self._save_json(output_directory)

        self._save_csv(output_directory)

        self.print_summary()

        success("Statistics exported successfully.")

    def print_summary(self) -> None:
        """
        Print dataset statistics.
        """

        if self.statistics is None:
            raise RuntimeError("Statistics have not been generated.")

        stats = self.statistics

        table = Table(
            title="Dataset Statistics",
            header_style="bold blue",
        )

        table.add_column("Metric")
        table.add_column(
            "Value",
            justify="right",
        )

        table.add_row(
            "Images",
            str(stats.total_images),
        )

        table.add_row(
            "Objects",
            str(stats.total_objects),
        )

        table.add_row(
            "Objects / Image",
            f"{stats.average_objects_per_image:.2f}",
        )

        table.add_row(
            "Brightness",
            f"{stats.average_brightness:.2f}",
        )

        table.add_row(
            "Contrast",
            f"{stats.average_contrast:.2f}",
        )

        table.add_row(
            "Sharpness",
            f"{stats.average_sharpness:.2f}",
        )

        table.add_row(
            "Persons",
            str(stats.total_persons),
        )

        table.add_row(
            "Bicycles",
            str(stats.total_bicycles),
        )

        table.add_row(
            "Vehicles",
            str(stats.total_vehicles),
        )

        table.add_row(
            "Images with VRU",
            str(stats.images_with_vru),
        )

        table.add_row(
            "Images with Vehicle",
            str(stats.images_with_vehicle),
        )

        console.print(table)
