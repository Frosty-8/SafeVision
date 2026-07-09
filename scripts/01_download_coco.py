from pathlib import Path
import requests
from tqdm import tqdm
import zipfile

BASE_DIR = Path("data/coco")

DOWNLOAD_DIR = BASE_DIR / "downloads"
IMAGE_DIR = BASE_DIR / "images"
ANNOTATION_DIR = BASE_DIR / "annotations"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url: str, destination: Path):
    if destination.exists():
        print(f"[SKIP] {destination.name} already exists.")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with (
        open(destination, "wb") as file,
        tqdm(
            desc=destination.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress,
    ):

        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                progress.update(len(chunk))


def unzip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def organize():

    annotation_source = BASE_DIR / "annotations"

    extracted = annotation_source / "annotations"

    if extracted.exists():

        for file in extracted.glob("*.json"):
            file.rename(annotation_source / file.name)

        extracted.rmdir()


def main():

    print("=" * 60)
    print("Downloading COCO Dataset")
    print("=" * 60)

    for filename, url in FILES.items():

        destination = DOWNLOAD_DIR / filename

        download(url, destination)

        if filename == "val2017.zip":
            unzip(destination, IMAGE_DIR)

        else:
            unzip(destination, ANNOTATION_DIR)

    organize()

    print()
    print("Done!")
    print()
    print(BASE_DIR)


if __name__ == "__main__":
    main()
