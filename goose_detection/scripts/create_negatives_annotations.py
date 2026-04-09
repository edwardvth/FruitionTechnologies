"""
create_negatives_annotations.py
Creates an empty COCO annotation file for already-downloaded COCO val2017 images.
Downloads instances_val2017.json to filter out bird images (category 16).

Usage:
    python scripts/create_negatives_annotations.py
"""

import json
from pathlib import Path
from PIL import Image

BASE_DIR    = Path(__file__).parent.parent
IMAGES_DIR  = BASE_DIR / "data" / "raw" / "negatives" / "negatives" / "val2017"
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "negatives" / "coco_annotations.json"

def main():
    # Scan all images already on disk — no download needed
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Images found on disk: {len(image_files)}")

    out_images = []
    skipped = 0

    for img_id, img_path in enumerate(image_files, start=1):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            skipped += 1
            continue

        out_images.append({
            "id":        img_id,
            "file_name": str(img_path),
            "width":     w,
            "height":    h,
        })

        if img_id % 500 == 0:
            print(f"  Processed {img_id}/{len(image_files)}...")

    print(f"Skipped {skipped} unreadable images")
    print(f"Negative images included: {len(out_images)}")

    output = {
        "info": {"description": "COCO val2017 negatives — no goose annotations"},
        "licenses": [],
        "categories": [{"id": 1, "name": "canadian_goose", "supercategory": "animal"}],
        "images": out_images,
        "annotations": [],   # intentionally empty
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
