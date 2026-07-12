from __future__ import annotations

from app.utils.logger import logger


class LatencyMetric:
    """
    Tracks inference latency.
    """

    def __init__(
        self,
    ) -> None:
        self.reset()

        logger.info(
            "LatencyMetric initialized."
        )

    def reset(
        self,
    ) -> None:
        """
        Reset latency statistics.
        """

        self.total_latency = 0.0

        self.total_samples = 0

        self.total_batches = 0

    def update(
        self,
        latency: float,
        batch_size: int,
    ) -> None:
        """
        Update latency statistics.

        Parameters
        ----------
        latency
            Batch inference time in seconds.

        batch_size
            Number of processed samples.
        """

        self.total_latency += latency

        self.total_batches += 1

        self.total_samples += batch_size

        if latency < 0:
        
            raise ValueError(
                "Latency must be non-negative."
            )
        
        if batch_size <= 0:
        
            raise ValueError(
                "Batch size must be positive."
            )

    def average_latency(
        self,
    ) -> float:
        """
        Average latency per batch in seconds.
        """

        if self.total_batches == 0:

            return 0.0

        return (

            self.total_latency

            / self.total_batches

        )

    def average_latency_ms(
        self,
    ) -> float:
        """
        Average latency per batch in milliseconds.
        """

        return (

            self.average_latency()

            * 1000.0

        )

    def fps(
        self,
    ) -> float:
        """
        Frames (samples) processed per second.
        """

        if self.total_latency <= 0.0:

            return 0.0

        return (

            self.total_samples

            / self.total_latency

        )

    def throughput(
        self,
    ) -> float:
        """
        Samples processed per second.
        """

        return self.fps()

    def summary(
        self,
    ) -> dict[str, float | int]:
        """
        Return latency statistics.
        """

        return {

            "latency_ms": self.average_latency_ms(),

            "latency_seconds": self.average_latency(),

            "fps": self.fps(),

            "throughput": self.throughput(),

            "samples": self.total_samples,

            "batches": self.total_batches,

        }

    @property
    def batches(
        self,
    ) -> int:
        """
        Number of processed batches.
        """
    
        return self.total_batches

    @property
    def samples(
        self,
    ) -> int:
        """
        Number of processed samples.
        """
    
        return self.total_samples

    def average_sample_latency_ms(
        self,
    ) -> float:
        """
        Average latency per sample.
        """
    
        if self.total_samples == 0:
    
            return 0.0
    
        return (
    
            self.total_latency
    
            / self.total_samples
    
        ) * 1000.0