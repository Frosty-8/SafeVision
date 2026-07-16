class BaseMetric:
    """
    Base class for all evaluation metrics.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
    ) -> None:

        self.iou_threshold = iou_threshold

        self.reset()

    def reset(
        self,
    ) -> None: ...
