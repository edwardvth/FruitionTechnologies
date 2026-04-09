"""
download_negatives.py
Downloads ~500 non-goose images from COCO 2017 val set to use as negative
training examples (images with NO goose annotations = empty bboxes).

Excludes bird images (COCO category 16) to avoid confusing the model.

Usage:
    python scripts/download_negatives.py --n 500
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

COCO_INSTANCES_URL = (
    "http://images.cocodataset.org/annotations/instances_val2017.json"
)
COCO_IMG_BASE = "http://images.cocodataset.org/val2017/"

BASE_DIR  = Path(__file__).parent.parent
OUT_DIR   = BASE_DIR / "data" / "raw" / "negatives" / "images"
ANN_PATH  = BASE_DIR / "data" / "raw" / "negatives" / "instances_val2017.json"

EXCLUDE_CATEGORIES = {16}  # bird — don't use these as negatives


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {desc or url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="Number of negative images to download (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # 1. Download COCO val annotation JSON (~26 MB)
    if not download_file(COCO_INSTANCES_URL, ANN_PATH, "COCO val2017 annotations (~26MB)"):
        print("Failed to download COCO annotations. Check your internet connection.")
        return

    print("Parsing COCO annotations...")
    with open(ANN_PATH) as f:
        coco = json.load(f)

    # 2. Find image IDs that contain excluded categories (birds)
    excluded_image_ids = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in EXCLUDE_CATEGORIES:
            excluded_image_ids.add(ann["image_id"])

    # 3. Filter to safe negative images
    safe_images = [
        img for img in coco["images"]
        if img["id"] not in excluded_image_ids
    ]
    print(f"Safe negative images (no birds): {len(safe_images)} / {len(coco['images'])}")

    # 4. Random sample
    sample = random.sample(safe_images, min(args.n, len(safe_images)))
    print(f"Downloading {len(sample)} images...\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for i, img_record in enumerate(sample):
        filename = img_record["file_name"]
        dest = OUT_DIR / filename
        url  = COCO_IMG_BASE + filename

        if dest.exists():
            downloaded.append(img_record)
            continue

        try:
            urllib.request.urlretrieve(url, dest)
            downloaded.append(img_record)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(sample)} downloaded...")
        except Exception as e:
            print(f"  [SKIP] {filename}: {e}")

    # 5. Save COCO JSON with empty annotations for downloaded images
    neg_coco = {
        "info": {"description": "COCO negatives — no goose annotations"},
        "licenses": [],
        "categories": [{"id": 1, "name": "canadian_goose", "supercategory": "animal"}],
        "images": [
            {
                "id":        img["id"],
                "file_name": str(OUT_DIR / img["file_name"]),
                "width":     img["width"],
                "height":    img["height"],
            }
            for img in downloaded
        ],
        "annotations": [],  # intentionally empty — these images have no geese
    }

    neg_ann_path = BASE_DIR / "data" / "raw" / "negatives" / "coco_annotations.json"
    with open(neg_ann_path, "w") as f:
        json.dump(neg_coco, f, indent=2)

    print(f"\nDone: {len(downloaded)} negative images downloaded")
    print(f"Annotations saved to: {neg_ann_path}")
    print(f"Images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
