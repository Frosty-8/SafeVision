from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkReport:

    latency_ms: float

    fps: float

    throughput: float

    gpu_memory_mb: float

    cpu_memory_mb: float
