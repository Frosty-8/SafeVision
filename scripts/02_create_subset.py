from pathlib import Path
from pycocotools.coco import COCO

import shutil
import random
import json

# ----------------------------
# Configuration
# ----------------------------

NUM_IMAGES = 1000

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

ROOT = Path("data/coco")

IMAGE_DIR = ROOT / "images" / "val2017"

ANNOTATION_FILE = ROOT / "annotations" / "instances_val2017.json"

OUTPUT_IMAGE_DIR = ROOT / "subset" / "images"

OUTPUT_ANNOTATION_DIR = ROOT / "subset" / "annotations"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load COCO
# ----------------------------

print("Loading COCO...")

coco = COCO(str(ANNOTATION_FILE))

# ----------------------------
# Category IDs
# ----------------------------

category_ids = coco.getCatIds(catNms=TARGET_CLASSES)

image_ids = set()

print("Selected Categories:")

for cat_id in category_ids:
    ids = coco.getImgIds(catIds=[cat_id])
    image_ids.update(ids)

# ----------------------------
# Image IDs
# ----------------------------

image_ids = list(image_ids)

print()

print(f"Found {len(image_ids)} images.")

# ----------------------------
# Random Sample
# ----------------------------

random.seed(42)

if NUM_IMAGES > len(image_ids):
    NUM_IMAGES = len(image_ids)

selected_ids = random.sample(image_ids, NUM_IMAGES)

print(f"Selected {len(selected_ids)} images.")

# ----------------------------
# Build Annotation
# ----------------------------

new_images = []

new_annotations = []

annotation_id = 1

for image_id in selected_ids:

    image = coco.loadImgs(image_id)[0]

    new_images.append(image)

    anns = coco.loadAnns(
        coco.getAnnIds(imgIds=image_id, catIds=category_ids, iscrowd=None)
    )

    for ann in anns:

        ann["id"] = annotation_id

        annotation_id += 1

        new_annotations.append(ann)

    src = IMAGE_DIR / image["file_name"]

    dst = OUTPUT_IMAGE_DIR / image["file_name"]

    if not dst.exists():
        shutil.copy(src, dst)

# ----------------------------
# Categories
# ----------------------------

categories = coco.loadCats(category_ids)

subset = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": categories,
}

output_json = OUTPUT_ANNOTATION_DIR / "instances_subset.json"

with open(output_json, "w") as f:

    json.dump(subset, f)

print()

print("=" * 50)

print("Subset Created Successfully")

print("=" * 50)

print(f"Images       : {len(new_images)}")

print(f"Annotations  : {len(new_annotations)}")

print(f"Saved Images : {OUTPUT_IMAGE_DIR}")

print(f"JSON         : {output_json}")
