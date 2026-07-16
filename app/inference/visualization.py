from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.schemas.detection import DetectionOutput

from app.utils.logger import logger

class Visualizer:
    """
    Draw inference results.
    """
    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> None:
    
        self.font_scale = font_scale
    
        self.thickness = thickness
    
        logger.info(
            "Visualizer initialized."
        )

    def draw(
        self,
        image: np.ndarray,
        outputs: DetectionOutput,
        class_names: list[str],
    ) -> np.ndarray:
        """
        Draw complete prediction.
        """
    
        image = image.copy()
    
        image = self.draw_boxes(
            image,
            outputs,
        )
    
        image = self.draw_labels(
            image,
            outputs,
            class_names,
        )
    
        image = self.draw_confidence(
            image,
            outputs,
        )
    
        return image

    def draw_boxes(
        self,
        image: np.ndarray,
        outputs: DetectionOutput,
    ) -> np.ndarray:
        """
        Draw bounding boxes.
        """
    
        height, width = image.shape[:2]
    
        for box in outputs.boxes.cpu().numpy():
    
            x1, y1, x2, y2 = box
    
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
    
            cv2.rectangle(
    
                image,
    
                (x1, y1),
    
                (x2, y2),
    
                (0, 255, 0),
    
                self.thickness,
    
            )
    
        return image



    def draw_labels(
        self,
        image: np.ndarray,
        outputs: DetectionOutput,
        class_names: list[str],
    ) -> np.ndarray:
        """
        Draw class labels.
        """
    
        if outputs.labels is None:
    
            return image
    
        height, width = image.shape[:2]
    
        for box, label in zip(
    
            outputs.boxes,
    
            outputs.labels,
    
            strict=True,
    
        ):
    
            x1 = int(box[0].item() * width)
            y1 = int(box[1].item() * height)
    
            cv2.putText(
    
                image,
    
                class_names[int(label)],
    
                (x1, y1 - 5),
    
                cv2.FONT_HERSHEY_SIMPLEX,
    
                self.font_scale,
    
                (0, 255, 0),
    
                self.thickness,
    
            )
    
        return image

    def draw_confidence(
        self,
        image: np.ndarray,
        outputs: DetectionOutput,
    ) -> np.ndarray:
        """
        Draw confidence scores.
        """
    
        height, width = image.shape[:2]
    
        for box, score in zip(
    
            outputs.boxes,
    
            outputs.confidence,
    
            strict=True,
    
        ):
    
            x1 = int(box[0].item() * width)
            y2 = int(box[3].item() * height)
    
            cv2.putText(
    
                image,
    
                f"{score:.2f}",
    
                (x1, y2 + 20),
    
                cv2.FONT_HERSHEY_SIMPLEX,
    
                self.font_scale,
    
                (255, 255, 0),
    
                self.thickness,
    
            )
    
        return image

    def draw_fps(
        self,
        image: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """
        Draw FPS.
        """
    
        cv2.putText(
    
            image,
    
            f"FPS: {fps:.2f}",
    
            (20, 30),
    
            cv2.FONT_HERSHEY_SIMPLEX,
    
            self.font_scale,
    
            (0, 255, 255),
    
            self.thickness,
    
        )
    
        return image


    def draw_risk(
        self,
        image: np.ndarray,
        risk_score: float,
    ) -> np.ndarray:
        """
        Draw overall risk score.
        """
    
        cv2.putText(
    
            image,
    
            f"Risk: {risk_score:.2f}",
    
            (20, 60),
    
            cv2.FONT_HERSHEY_SIMPLEX,
    
            self.font_scale,
    
            (0, 0, 255),
    
            self.thickness,
    
        )
    
        return image


    def save(
        self,
        image: np.ndarray,
        path: str | Path,
    ) -> None:
        """
        Save annotated image.
        """
    
        cv2.imwrite(
    
            str(path),
    
            image,
    
        )
    
        logger.info(
            "Visualization saved: %s",
            path,
        )

    def show(
        self,
        image: np.ndarray,
        window_name: str = "SafeVision",
    ) -> None:
        """
        Display image.
        """
    
        cv2.imshow(
            window_name,
            image,
        )
    
        cv2.waitKey(0)
    
        cv2.destroyAllWindows()