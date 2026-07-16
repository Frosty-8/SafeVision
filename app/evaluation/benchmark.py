from __future__ import annotations

import time, psutil

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.metrics.latency import LatencyMetric

from app.utils.logger import logger
from app.utils.ui import success, title


class Benchmark:
    """
    Runs inference benchmarks.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> None:

        self.model = model

        self.dataloader = dataloader

        self.device = device

        self.latency = LatencyMetric()

        self.model.to(device)

        logger.info("Benchmark initialized.")

    def warmup(
        self,
        iterations: int = 20,
    ) -> None:
        """
        Warm up model.
        """

        logger.info("Running warmup...")

        self.model.eval()

        with torch.no_grad():

            iterator = iter(
                self.dataloader,
            )

            for _ in range(iterations):

                try:

                    images, _ = next(
                        iterator,
                    )

                except StopIteration:

                    break

                images = images.to(
                    self.device,
                )

                _ = self.model(
                    features=images,
                )

        logger.info("Warmup completed.")

    def benchmark(
        self,
    ) -> dict[str, float]:
        """
        Benchmark detector.
        """

        title(
            "Inference Benchmark",
        )

        self.warmup()

        self.model.eval()

        self.latency.reset()

        with torch.no_grad():

            for images, _ in self.dataloader:

                images = images.to(
                    self.device,
                )

                if torch.cuda.is_available():

                    torch.cuda.synchronize()

                start = time.perf_counter()

                _ = self.model(
                    features=images,
                )

                if torch.cuda.is_available():

                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start

                self.latency.update(
                    elapsed,
                    images.size(0),
                )

        success("Benchmark completed.")

        return self._build_report()

    def _gpu_memory(
        self,
    ) -> float:
        """
        GPU memory in MB.
        """

        if not torch.cuda.is_available():

            return 0.0

        return torch.cuda.max_memory_allocated() / (1024**2)

    def _cpu_memory(
        self,
    ) -> float:
        """
        CPU memory in MB.
        """

        process = psutil.Process()

        return process.memory_info().rss / (1024**2)

    def _build_report(
        self,
    ) -> dict[str, float]:
        """
        Benchmark summary.
        """

        report = self.latency.summary()

        report.update(
            {
                "gpu_memory_mb": self._gpu_memory(),
                "cpu_memory_mb": self._cpu_memory(),
            }
        )

        return report

    def summary(
        self,
    ) -> None:
        """
        Display benchmark results.
        """

        report = self._build_report()

        logger.info("Benchmark Summary")

        for key, value in report.items():

            logger.info(
                "%s : %s",
                key,
                value,
            )
