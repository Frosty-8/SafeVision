from pathlib import Path

from loguru import logger

from rich.logging import RichHandler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()

logger.add(
    LOG_DIR / "safevision.log",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

logger.add(
    RichHandler(
        rich_tracebacks=True,
        markup=True,
    ),
    level="INFO",
)

__all__ = ["logger"]