from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from app.schemas.xgboost import DetectionFeatures
from app.utils.logger import logger

class FeatureEngineer:
    """
    Converts detector outputs into structured XGBoost features.
    """
    def __init__(self)-> None:
        logger.info(
            "FeatuerEngineer Initialized."
        )

    def extract(
        self,
        detections: list[DetectionFeatures],
    ) -> list[DetectionFeatures]:
        """
        Validate and normalize features.
        """
    
        features = []
    
        for detection in detections:
    
            self.validate(
                detection,
            )
    
            features.append(
    
                self.normalize(
                    detection,
                )
    
            )
    
        logger.info(
            "Extracted {} feature vectors.",
            len(features),
        )
    
        return features


    def validate(
        self,
        detection: DetectionFeatures,
    ) -> None:
        """
        Validate feature ranges.
        """
    
        if not 0.0 <= detection.confidence <= 1.0:
    
            raise ValueError(
                "Confidence must be between 0 and 1."
            )
    
        if detection.box_width <= 0:
    
            raise ValueError(
                "Invalid box width."
            )
    
        if detection.box_height <= 0:
    
            raise ValueError(
                "Invalid box height."
            )
    
        if detection.box_area <= 0:
    
            raise ValueError(
                "Invalid box area."
            )

    def normalize(
        self,
        detection: DetectionFeatures,
    ) -> DetectionFeatures:
        """
        Normalize numerical features.
        """
    
        detection.confidence = float(
    
            np.clip(
    
                detection.confidence,
    
                0.0,
    
                1.0,
    
            )
    
        )
    
        detection.occlusion_score = float(
    
            np.clip(
    
                detection.occlusion_score,
    
                0.0,
    
                1.0,
    
            )
    
        )
    
        detection.attention_score = float(
    
            np.clip(
    
                detection.attention_score,
    
                0.0,
    
                1.0,
    
            )
    
        )
    
        return detection


    def to_dataframe(
        self,
        features: list[DetectionFeatures],
    ) -> pd.DataFrame:
        """
        Convert features into a DataFrame.
        """
    
        dataframe = pd.DataFrame(
    
            [
    
                asdict(
                    feature,
                )
    
                for feature in features
    
            ]
    
        )
    
        logger.info(
            "Created DataFrame with {} rows.",
            len(dataframe),
        )
    
        return dataframe


    @property
    def feature_names(
        self,
    ) -> list[str]:
        """
        Ordered feature names.
        """
    
        return list(
    
            DetectionFeatures.__dataclass_fields__.keys()
    
        )