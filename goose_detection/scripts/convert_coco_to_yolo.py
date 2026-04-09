"""
convert_coco_to_yolo.py
Converts the processed COCO-format splits into YOLO format and merges in the
Gabe-o GitHub dataset.

Output structure:
    data/yolo/
        images/train/   images/val/   images/test/
        labels/train/   labels/val/   labels/test/

Usage:
    python scripts/convert_coco_to_yolo.py
"""

import json
import shutil
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
GABE_DIR   = BASE_DIR / "data" / "raw" / "gabe_yolo" / "data"
OUT_DIR    = BASE_DIR / "data" / "yolo"

SPLITS = {
    "train": PROC_DIR / "train_annotations.json",
    "val":   PROC_DIR / "val_annotations.json",
    "test":  PROC_DIR / "test_annotations.json",
}


def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, bw, bh = bbox
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return cx, cy, nw, nh


def convert_split(split_name, ann_path):
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        coco = json.load(f)

    # Build annotation lookup
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    copied = 0
    for img_rec in coco["images"]:
        src = Path(img_rec["file_name"])
        if not src.exists():
            src = PROC_DIR / "images" / Path(img_rec["file_name"]).name
        if not src.exists():
            continue

        dst_img = img_out / src.name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)

        # Write label file
        dst_lbl = lbl_out / (src.stem + ".txt")
        anns = anns_by_image.get(img_rec["id"], [])
        w, h = img_rec["width"], img_rec["height"]

        lines = []
        for ann in anns:
            cx, cy, nw, nh = coco_to_yolo(ann["bbox"], w, h)
            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_lbl.write_text("\n".join(lines))  # empty file = negative image
        copied += 1

    print(f"  {split_name}: {copied} images written")
    return copied


def merge_gabe(split_name):
    """Copy Gabe-o images+labels into the train split only."""
    src_img = GABE_DIR / "images" / split_name
    src_lbl = GABE_DIR / "labels" / split_name
    dst_img = OUT_DIR / "images" / split_name
    dst_lbl = OUT_DIR / "labels" / split_name

    if not src_img.exists():
        return 0

    count = 0
    for img_file in src_img.glob("*.*"):
        dst = dst_img / ("gabe_" + img_file.name)
        if not dst.exists():
            shutil.copy2(img_file, dst)
        lbl_file = src_lbl / (img_file.stem + ".txt")
        dst_lbl_file = dst_lbl / ("gabe_" + img_file.stem + ".txt")
        if lbl_file.exists() and not dst_lbl_file.exists():
            shutil.copy2(lbl_file, dst_lbl_file)
        elif not dst_lbl_file.exists():
            dst_lbl_file.write_text("")  # negative
        count += 1

    return count


def main():
    print("Converting COCO splits to YOLO format...")
    total = 0
    for split, ann_path in SPLITS.items():
        total += convert_split(split, ann_path)

    print(f"\nMerging Gabe-o dataset...")
    for split in ["train", "val", "test"]:
        n = merge_gabe(split)
        if n:
            print(f"  {split}: +{n} images from Gabe-o")

    # Final counts
    print("\nFinal dataset size:")
    for split in ["train", "val", "test"]:
        n_img = len(list((OUT_DIR / "images" / split).glob("*.*")))
        n_lbl = len(list((OUT_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_img} images, {n_lbl} labels")

    print(f"\nDone. Dataset at: {OUT_DIR}")


if __name__ == "__main__":
    main()
