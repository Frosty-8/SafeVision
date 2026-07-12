from __future__ import annotations

from app.utils.logger import logger

class RobustnessMetric:
    """
    Computes the overall robustness score.
    """

    def __init__(
        self 
    ) -> None:
        self.reset()

        logger.info(
            "RobustnessMetric initialized."
        )

    def reset(
        self,
    ) -> None:
        """
        Reset robustness statistics.
        """
    
        self.weather_score = 0.0
    
        self.night_score = 0.0
    
        self.occlusion_score = 0.0
    
        self.slice_score = 0.0

    def update(
        self,
        weather: float,
        night: float,
        occlusion: float,
        slice_score: float,
    ) -> None:
        """
        Update robustness scores.
        """
    
        for score in (
            weather,
            night,
            occlusion,
            slice_score,
        ):
    
            if not 0.0 <= score <= 1.0:
    
                raise ValueError(
                    "Scores must be between 0 and 1."
                )
    
        self.weather_score = weather
    
        self.night_score = night
    
        self.occlusion_score = occlusion
    
        self.slice_score = slice_score

    def compute(
        self,
    ) -> float:
        """
        Compute weighted robustness score.
        """
    
        return (
    
            0.30 * self.weather_score
    
            +
    
            0.25 * self.night_score
    
            +
    
            0.25 * self.occlusion_score
    
            +
    
            0.20 * self.slice_score
    
        )

    def overall_score(
        self,
    ) -> float:
        """
        Overall robustness score.
        """
    
        return self.compute()

    def summary(
        self,
    ) -> dict[str, float]:
        """
        Return robustness statistics.
        """
    
        return {
    
            "weather": self.weather_score,
    
            "night": self.night_score,
    
            "occlusion": self.occlusion_score,
    
            "slice": self.slice_score,
    
            "overall": self.compute(),
    
        }