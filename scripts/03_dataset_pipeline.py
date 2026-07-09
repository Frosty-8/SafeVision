from app.datasets.loaders import CocoLoader
from app.datasets.pipeline import DatasetPipeline

loader = CocoLoader(
    image_directory="data/coco/subset/images",
    annotation_file="data/coco/subset/annotations/instances_subset.json",
)

pipeline = DatasetPipeline(loader)

pipeline.run(output_directory="data/processed")
