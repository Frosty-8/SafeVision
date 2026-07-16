<!--Yes. Since `_copy()` is a **private helper method**, it should be defined **inside the `PostProcessor` class**, just like `filter_scores()`, `convert_boxes()`, etc.

A good placement is immediately after `__init__()` because every processing method can use it.

```python
class PostProcessor:
    """
    Cleans detector predictions.
    """

    def __init__(
        self,
        score_threshold: float = 0.25,
    ) -> None:

        self.score_threshold = score_threshold

        logger.info(
            "PostProcessor initialized."
        )

    def _copy(
        self,
        outputs: DetectionOutput,
        *,
        class_logits: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        queries: torch.Tensor | None = None,
    ) -> DetectionOutput:
        """
        Create a new DetectionOutput by copying
        unchanged fields and replacing only the
        provided ones.
        """

        return DetectionOutput(
            class_logits=(
                outputs.class_logits
                if class_logits is None
                else class_logits
            ),
            boxes=(
                outputs.boxes
                if boxes is None
                else boxes
            ),
            confidence=(
                outputs.confidence
                if confidence is None
                else confidence
            ),
            labels=(
                outputs.labels
                if labels is None
                else labels
            ),
            queries=(
                outputs.queries
                if queries is None
                else queries
            ),
        )

    def __call__(...)
        ...
```

Then your methods become much cleaner.

For example, `clip_boxes()` becomes:

```python
def clip_boxes(
    self,
    outputs: DetectionOutput,
) -> DetectionOutput:
    """
    Clip normalized coordinates.
    """

    return self._copy(
        outputs,
        boxes=outputs.boxes.clamp(0.0, 1.0),
    )
```

`prepare_for_nms()` becomes:

```python
def prepare_for_nms(
    self,
    outputs: DetectionOutput,
) -> DetectionOutput:
    """
    Decode predicted labels.
    """

    return self._copy(
        outputs,
        labels=outputs.class_logits.argmax(dim=-1),
    )
```

And `convert_boxes()` becomes:

```python
def convert_boxes(
    self,
    outputs: DetectionOutput,
) -> DetectionOutput:

    cx, cy, w, h = outputs.boxes.unbind(dim=-1)

    boxes = torch.stack(
        (
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ),
        dim=-1,
    )

    return self._copy(
        outputs,
        boxes=boxes,
    )
```

### One additional suggestion

If you're adopting this immutable design, consider making `DetectionOutput` itself immutable:

```python
@dataclass(slots=True, frozen=True)
class DetectionOutput:
    ...
```

With `frozen=True`, Python will prevent accidental in-place modifications anywhere in the codebase, ensuring all changes go through `_copy()` (or by constructing a new `DetectionOutput`). This aligns well with the immutable pipeline you're building.-->
