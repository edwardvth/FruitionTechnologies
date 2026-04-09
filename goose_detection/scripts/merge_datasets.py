"""
merge_datasets.py
Merges multiple COCO JSON annotation files into one unified dataset.
- Remaps all categories to a single class: canadian_goose (id=1)
- Deduplicates images using MD5 hashing of image files
- Copies images into a single output directory
- Splits dataset into train/val/test (80/10/10)

Usage:
    python scripts/merge_datasets.py \
        --sources \
            data/raw/steggie3/coco_annotations.json:data/raw/steggie3/images \
            data/raw/images_cv/coco_annotations.json:data/raw/images_cv/images \
            data/raw/roboflow/coco_annotations.json:data/raw/roboflow/images \
        --output_images data/processed/images \
        --output_dir data/processed
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


GOOSE_CLASS = {"id": 1, "name": "canadian_goose", "supercategory": "animal"}
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42


def md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_coco(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def merge(sources: list[tuple[Path, Path]], output_images_dir: Path) -> tuple[list, list]:
    """
    Merge multiple COCO datasets.
    sources: list of (annotations_json_path, images_dir_path)
    Returns (all_images, all_annotations) in COCO format with new sequential IDs.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = {}   # md5 -> new image_id
    all_images = []
    all_annotations = []
    new_image_id = 1
    new_ann_id = 1

    for ann_path, img_dir in sources:
        coco = load_coco(ann_path)

        # Build old_image_id → image_record map
        id_to_image = {img["id"]: img for img in coco["images"]}
        # Build old category_id → new category_id map (everything → 1)
        old_cat_ids = {cat["id"] for cat in coco.get("categories", [])}
        cat_remap = {cid: 1 for cid in old_cat_ids}

        # Track old→new image id mapping for annotations
        old_to_new_image_id = {}

        for img_record in coco["images"]:
            filename = img_record["file_name"]
            candidate = Path(filename)
            if candidate.exists():
                # Absolute path or relative path from cwd — works as-is
                src_img_path = candidate
            elif (img_dir / candidate.name).exists():
                # Just the bare filename inside img_dir
                src_img_path = img_dir / candidate.name
            else:
                src_img_path = img_dir / filename

            if not src_img_path.exists():
                print(f"  [SKIP] Image not found: {src_img_path}")
                continue

            file_hash = md5(src_img_path)

            if file_hash in seen_hashes:
                # Duplicate — map old id to existing new id
                old_to_new_image_id[img_record["id"]] = seen_hashes[file_hash]
                print(f"  [DUP] {filename} (hash match)")
                continue

            # Copy image with new unique name to avoid filename collisions
            new_filename = f"{new_image_id:06d}_{src_img_path.name}"
            dst_img_path = output_images_dir / new_filename
            shutil.copy2(src_img_path, dst_img_path)

            seen_hashes[file_hash] = new_image_id
            old_to_new_image_id[img_record["id"]] = new_image_id

            all_images.append(
                {
                    "id": new_image_id,
                    "file_name": new_filename,
                    "width": img_record.get("width", 0),
                    "height": img_record.get("height", 0),
                }
            )
            new_image_id += 1

        for ann in coco.get("annotations", []):
            old_img_id = ann["image_id"]
            if old_img_id not in old_to_new_image_id:
                continue  # image was skipped or not found
            new_img_id = old_to_new_image_id[old_img_id]
            old_cat = ann.get("category_id", 1)
            new_cat = cat_remap.get(old_cat, 1)

            all_annotations.append(
                {
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": new_cat,
                    "bbox": ann["bbox"],
                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            new_ann_id += 1

    return all_images, all_annotations


def split_and_save(all_images: list, all_annotations: list, output_dir: Path) -> None:
    """Split merged dataset into train/val/test and save separate COCO JSON files."""
    random.seed(SEED)
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train : n_train + n_val],
        "test": all_images[n_train + n_val :],
    }

    ann_by_image = {}
    for ann in all_annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    base_coco = {
        "info": {"description": "Merged Canadian Goose Detection Dataset", "version": "1.0"},
        "licenses": [],
        "categories": [GOOSE_CLASS],
    }

    for split_name, images in splits.items():
        image_ids = {img["id"] for img in images}
        split_anns = [a for iid in image_ids for a in ann_by_image.get(iid, [])]

        coco_out = {**base_coco, "images": images, "annotations": split_anns}
        out_path = output_dir / f"{split_name}_annotations.json"
        with open(out_path, "w") as f:
            json.dump(coco_out, f, indent=2)

        print(f"  {split_name}: {len(images)} images, {len(split_anns)} annotations → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple COCO datasets")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        metavar="JSON:IMAGES_DIR",
        help="Colon-separated pairs of annotation JSON and image directory",
    )
    parser.add_argument("--output_images", required=True, help="Directory for merged images")
    parser.add_argument("--output_dir", required=True, help="Directory for split JSON files")
    args = parser.parse_args()

    sources = []
    for s in args.sources:
        json_path, img_dir = s.split(":", 1)
        sources.append((Path(json_path), Path(img_dir)))

    output_images_dir = Path(args.output_images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Merging datasets...")
    all_images, all_annotations = merge(sources, output_images_dir)
    print(f"Total after dedup: {len(all_images)} images, {len(all_annotations)} annotations")

    print("Splitting into train/val/test...")
    split_and_save(all_images, all_annotations, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
