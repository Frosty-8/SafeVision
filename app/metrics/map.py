from __future__ import annotations

from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from app.utils.logger import logger


class MAPMetric:
    """
    Computes COCO Mean Average Precision.
    """

    def __init__(
        self,
        annotation_file: str | Path,
    ) -> None:

        self.annotation_file = Path(
            annotation_file,
        )

        self.coco_gt = COCO(
            str(self.annotation_file),
        )

        self.results = None

        logger.info("MAPMetric initialized.")

    def evaluate(
        self,
        prediction_file: str | Path,
    ) -> None:
        """
        Evaluate predictions using COCO.
        """

        coco_predictions = self.coco_gt.loadRes(
            str(prediction_file),
        )

        evaluator = COCOeval(
            self.coco_gt,
            coco_predictions,
            "bbox",
        )

        evaluator.evaluate()

        evaluator.accumulate()

        evaluator.summarize()

        self.results = evaluator.stats

    @property
    def map(
        self,
    ) -> float:

        return float(
            self.results[0],
        )

    @property
    def map50(
        self,
    ) -> float:

        return float(
            self.results[1],
        )

    @property
    def map75(
        self,
    ) -> float:

        return float(
            self.results[2],
        )

    @property
    def aps(
        self,
    ) -> float:

        return float(
            self.results[3],
        )

    @property
    def apm(
        self,
    ) -> float:

        return float(
            self.results[4],
        )

    @property
    def apl(
        self,
    ) -> float:

        return float(
            self.results[5],
        )

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Return COCO evaluation metrics.
        """

        return {
            "map": self.map,
            "map50": self.map50,
            "map75": self.map75,
            "aps": self.aps,
            "apm": self.apm,
            "apl": self.apl,
        }
