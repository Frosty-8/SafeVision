"""
SafeVision AI

Dataset Splitter

Splits datasets into Train / Validation / Test.

Author: Sarthak Dongare
License: Apache-2.0
"""

from __future__ import annotations
from typing import Any
import json
import random

from dataclasses import asdict
from pathlib import Path

from rich.table import Table

from app.datasets.loaders.base_loader import BaseDatasetLoader

from app.schemas.split import DatasetSplit
from app.schemas.split import SplitReport
from sklearn.model_selection import train_test_split
from app.utils.console import console
from app.utils.logger import logger
from app.utils.ui import (
    success,
    title,
)


class DatasetSplitter:
    """
    Split datasets into train, validation and test subsets
    """
    def __init__(
        self,
        loader: BaseDatasetLoader,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:

        for ratio in (
            train_ratio,
            validation_ratio,
            test_ratio,
        ):
    
            if ratio <= 0:
    
                raise ValueError(
                    "Split ratios must be positive."
                )
                  
                  
        # ------------------------------
        # Validate total
        # ------------------------------
        if abs(
            train_ratio
            + validation_ratio
            + test_ratio
            - 1.0
        ) > 1e-6:
    
            raise ValueError(
                "Split ratios must sum to 1."
            )
    
        self.loader = loader
    
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
    
        self.seed = seed
    
        self.train_split: DatasetSplit | None = None
        self.validation_split: DatasetSplit | None = None
        self.test_split: DatasetSplit | None = None
    
        self.report: SplitReport | None = None
    
        random.seed(seed)

    def split(
        self,
        stratified: bool = True,
    ) -> SplitReport:
        """
        Split dataset.
        """
    
        title("Dataset Splitter")
    
        logger.info(
            "Starting dataset split..."
        )
    
        if stratified:
            try:
                self._stratified_split()

            except ValueError as exc:

                logger.warning(
                    "Stratified split failed : {}",
                    exc,
                )

                logger.warning(
                    "Falling back to random split."
                )

                self._random_split()
                
        else:
    
            self._random_split()
    
        self._build_report()
    
        success(
            "Dataset successfully split."
        )
    
        return self.report

    def _get_primary_category(
        self,
        image_id: int,
    ) -> str:
        """
        Determine the dominant category for an image.
    
        Used for stratified splitting.
        """
    
        annotations = self.loader.get_annotations(image_id)
    
        if not annotations:
            return "background"
    
        counts: dict[str, int] = {}
    
        for annotation in annotations:
    
            category = self.loader.get_category_name(
                annotation.category_id
            )
    
            if category is None:
                continue
    
            counts[category] = counts.get(category, 0) + 1
    
        return max(counts.items(),key=lambda item: item[1])[0]

    def _random_split(
        self,
    ) -> None:
        """
        Randomly split dataset.
        """
    
        images = self.loader.get_images()
    
        image_ids = [
            image.id
            for image in images
        ]
    
        train_ids, remaining_ids = train_test_split(
            image_ids,
            train_size=self.train_ratio,
            random_state=self.seed,
            shuffle=True,
        )
    
        validation_size = (
            self.validation_ratio
            / (
                self.validation_ratio
                + self.test_ratio
            )
        )
    
        validation_ids, test_ids = train_test_split(
            remaining_ids,
            train_size=validation_size,
            random_state=self.seed,
            shuffle=True,
        )
    
        self.train_split = self._build_split(
            train_ids,
        )
    
        self.validation_split = self._build_split(
            validation_ids,
        )
    
        self.test_split = self._build_split(
            test_ids,
        )


    def _stratified_split(
        self,
    ) -> None:
        """
        Stratified dataset split.
        """
    
        images = self.loader.get_images()
    
        image_ids: list[int] = []
    
        labels: list[str] = []
    
        for image in images:
    
            image_ids.append(
                image.id,
            )
    
            labels.append(
                self._get_primary_category(
                    image.id,
                )
            )
    
        train_ids, remaining_ids, _, remaining_labels = train_test_split(
            image_ids,
            labels,
            train_size=self.train_ratio,
            stratify=labels,
            random_state=self.seed,
        )
    
        validation_ratio = (
            self.validation_ratio
            / (
                self.validation_ratio
                + self.test_ratio
            )
        )
    
        validation_ids, test_ids = train_test_split(
            remaining_ids,
            train_size=validation_ratio,
            stratify=remaining_labels,
            random_state=self.seed,
        )
    
        self.train_split = self._build_split(
            train_ids,
        )
    
        self.validation_split = self._build_split(
            validation_ids,
        )
    
        self.test_split = self._build_split(
            test_ids,
        )

    def _build_split(
        self,
        image_ids: list[int],
    ) -> DatasetSplit:
        """
        Build a dataset split.
        """
    
        images: list = []
    
        annotations: list = []
    
        for image_id in image_ids:
    
            image = self.loader.get_image(
                image_id,
            )
    
            if image is not None:
                images.append(
                    image,
                )
    
            annotations.extend(
                self.loader.get_annotations(
                    image_id,
                )
            )
    
        categories: list = []
    
        for category_id in self.loader.category_ids:
    
            category = self.loader.get_category(
                category_id,
            )
    
            if category is not None:
                categories.append(
                    category,
                )
    
        return DatasetSplit(
            images=images,
            annotations=annotations,
            categories=categories,
        )

    def _build_report(self) -> None:
        """
        Build split report.
        """
    
        if (
            self.train_split is None
            or self.validation_split is None
            or self.test_split is None
        ):
            raise RuntimeError(
                "Dataset has not been split."
            )
    
        self.report = SplitReport(
            train_images=len(self.train_split.images),
            validation_images=len(self.validation_split.images),
            test_images=len(self.test_split.images),
    
            train_annotations=len(self.train_split.annotations),
            validation_annotations=len(self.validation_split.annotations),
            test_annotations=len(self.test_split.annotations),
        )

    def _split_to_dict(
        self,
        split: DatasetSplit,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Convert DatasetSplit into COCO format.
        """
    
        return {
            "images": [
                asdict(image)
                for image in split.images
            ],
            "annotations": [
                asdict(annotation)
                for annotation in split.annotations
            ],
            "categories": [
                asdict(category)
                for category in split.categories
            ],
        }

    def _save_json(
        self,
        split: DatasetSplit,
        path: Path,
    ) -> None:
        """
        Save dataset split.
        """
    
        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
    
            json.dump(
                self._split_to_dict(
                    split,
                ),
                file,
                indent=4,
            )
    
        logger.info(
            "Saved {}",
            path.name,
        )

    def _save_summary(
        self,
        output_directory: Path,
    ) -> None:
        """
        Save split summary.
        """
    
        if self.report is None:
            raise RuntimeError(
                "Report has not been generated."
            )
    
        summary_path = (
            output_directory
            / "split_summary.json"
        )
    
        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:
    
            json.dump(
                asdict(self.report),
                file,
                indent=4,
            )
    
        logger.info(
            "Saved split summary."
        )

    def export(
        self,
        output_directory: str | Path,
    ) -> None:
        """
        Export dataset splits.
        """
    
        if (
            self.train_split is None
            or self.validation_split is None
            or self.test_split is None
        ):
            raise RuntimeError(
                "Split dataset first."
            )
    
        output_directory = Path(
            output_directory,
        )
    
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        title("Exporting Dataset Splits")
    
        self._save_json(
            self.train_split,
            output_directory / "train.json",
        )
    
        self._save_json(
            self.validation_split,
            output_directory / "validation.json",
        )
    
        self._save_json(
            self.test_split,
            output_directory / "test.json",
        )
    
        self._save_summary(
            output_directory,
        )
    
        self.print_summary()
    
        success(
            "Dataset exported successfully."
        )

    def print_summary(
        self,
    ) -> None:
        """
        Display split summary.
        """
    
        if self.report is None:
            raise RuntimeError(
                "Report has not been generated."
            )
    
        report = self.report
    
        table = Table(
            title="Dataset Split Summary",
            header_style="bold green",
        )
    
        table.add_column("Subset")
        table.add_column(
            "Images",
            justify="right",
        )
        table.add_column(
            "Annotations",
            justify="right",
        )
    
        table.add_row(
            "Train",
            str(report.train_images),
            str(report.train_annotations),
        )
    
        table.add_row(
            "Validation",
            str(report.validation_images),
            str(report.validation_annotations),
        )
    
        table.add_row(
            "Test",
            str(report.test_images),
            str(report.test_annotations),
        )
    
        console.print(table)