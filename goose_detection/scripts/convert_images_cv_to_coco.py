"""
convert_images_cv_to_coco.py
Converts the images.cv classification dataset to COCO format.

Since images.cv has no bounding box annotations, each image gets a single
whole-image bounding box (x=0, y=0, w=img_width, h=img_height).
All images contain geese, so the whole image is a reasonable approximation.

Images are found recursively under the dataset root — works regardless of
the nested folder structure images.cv uses (e.g. goose/, animal goose/).

Usage:
    python scripts/convert_images_cv_to_coco.py \
        --dataset_root data/raw/images_cv \
        --output data/raw/images_cv/coco_annotations.json
"""

import argparse
import json
from pathlib import Path

from PIL import Image


GOOSE_CLASS = {"id": 1, "name": "canadian_goose", "supercategory": "animal"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: Path) -> list[Path]:
    """Recursively find all image files under root."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_size(img_path: Path) -> tuple[int, int]:
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def convert(dataset_root: Path, output_path: Path) -> None:
    images = find_images(dataset_root)
    if not images:
        raise FileNotFoundError(f"No images found under {dataset_root}")

    print(f"Found {len(images)} images under {dataset_root}")

    coco = {
        "info": {
            "description": "images.cv goose classification dataset — whole-image bboxes",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [GOOSE_CLASS],
        "images": [],
        "annotations": [],
    }

    skipped = 0
    for img_id, img_path in enumerate(images, start=1):
        try:
            width, height = get_image_size(img_path)
        except Exception as e:
            print(f"  [SKIP] Cannot read {img_path.name}: {e}")
            skipped += 1
            continue

        coco["images"].append(
            {
                "id": img_id,
                "file_name": str(img_path),   # store absolute path for merge_datasets.py
                "width": width,
                "height": height,
            }
        )

        # Whole-image bounding box in COCO format: [x, y, width, height]
        coco["annotations"].append(
            {
                "id": img_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [0, 0, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )

        if img_id % 200 == 0:
            print(f"  Processed {img_id}/{len(images)}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nDone: {len(coco['images'])} images, {skipped} skipped")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert images.cv classification dataset to COCO JSON with whole-image bboxes"
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Root folder of the images.cv download (e.g. data/raw/images_cv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output COCO JSON path (e.g. data/raw/images_cv/coco_annotations.json)",
    )
    args = parser.parse_args()

    convert(Path(args.dataset_root), Path(args.output))


if __name__ == "__main__":
    main()
