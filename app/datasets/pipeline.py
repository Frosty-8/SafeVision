"""
SafeVision AI

Dataset Pipeline

Coordinates the complete dataset preparation workflow.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations

from pathlib import Path

from app.datasets.loaders.base_loader import BaseDatasetLoader
from app.datasets.metadata import MetadataGenerator
from app.datasets.statistics import StatisticsGenerator
from app.datasets.validators import DatasetValidator

from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)


class DatasetPipeline:
    """
    Complete dataset preparation pipeline.

    Pipeline Stages
    ---------------

    1. Load Dataset

    2. Validate Dataset

    3. Generate Metadata

    4. Generate Statistics

    5. Split Dataset (future)

    6. Export Results
    """

    def __init__(self, loader: BaseDatasetLoader) -> None:
        self.loader = loader

        self.validator = DatasetValidator(
            loader,
        )

        self.metadata_generator = MetadataGenerator(
            loader,
        )

        self.statistics_generator: StatisticsGenerator | None = None

    def load(self) -> None:
        """
        Load Dataset
        """

        logger.info("Loading Dataset...")

        self.loader.load()

    def validate(self):
        """
        Validate dataset
        """
        return self.validator.validate()

    def generate_metadata(self):
        """
        Generate image metadata.
        """
        metadata = self.metadata_generator.generate()

        self.statistics_generator = StatisticsGenerator(metadata)

    def generate_statistics(self):
        """
        Generate dataset statistics
        """
        if self.statistics_generator is None:

            raise RuntimeError("Generate metadata first")

        return self.statistics_generator.generate()

    def export(
        self,
        output_directory: str | Path,
    ) -> None:
        """
        Export all generated artifacts.
        """

        output_directory = Path(output_directory)

        self.metadata_generator.export(output_directory / "metadata")

        if self.statistics_generator:

            self.statistics_generator.export(output_directory / "statistics")

    def run(
        self,
        output_directory: str | Path,
    ):
        """
        Execute the complete pipeline.
        """

        title("SafeVision Dataset Pipeline")

        logger.info("Starting dataset pipeline...")

        self.load()

        validation_report = self.validate()

        if not validation_report.valid:

            raise RuntimeError("Dataset validation failed.")

        self.generate_metadata()

        self.generate_statistics()

        self.export(output_directory)

        success("Dataset pipeline completed successfully.")

        logger.info("Pipeline finished.")
