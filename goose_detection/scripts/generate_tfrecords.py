"""
generate_tfrecords.py
Converts merged COCO JSON datasets into TFRecord files for EfficientDetD0 training
via the TensorFlow Model Garden pipeline.

Requires: tensorflow, Pillow

Usage:
    python scripts/generate_tfrecords.py \
        --images_dir data/processed/images \
        --annotations_dir data/processed \
        --output_dir data/tfrecords \
        --num_shards 4
"""

import argparse
import io
import json
import math
import os
from pathlib import Path

import tensorflow as tf
from PIL import Image


def load_coco(json_path: Path) -> tuple[dict, dict]:
    """Returns (id_to_image, id_to_annotations_list)."""
    with open(json_path) as f:
        coco = json.load(f)

    id_to_image = {img["id"]: img for img in coco["images"]}
    id_to_anns: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    return id_to_image, id_to_anns


def image_to_bytes(img_path: Path) -> bytes:
    with open(img_path, "rb") as f:
        return f.read()


def resize_bbox(bbox, orig_w, orig_h, target=512):
    """Normalize bbox to [0,1] range for TFRecord storage."""
    x, y, w, h = bbox
    return [
        y / orig_h,           # ymin
        x / orig_w,           # xmin
        (y + h) / orig_h,     # ymax
        (x + w) / orig_w,     # xmax
    ]


def create_tf_example(image_record: dict, anns: list, images_dir: Path) -> tf.train.Example:
    filename = image_record["file_name"]
    img_path = images_dir / filename

    img_bytes = image_to_bytes(img_path)

    # Use PIL to get actual dimensions (in case metadata is missing)
    with Image.open(io.BytesIO(img_bytes)) as img:
        width, height = img.size
        img_format = img.format.lower().encode() if img.format else b"jpeg"

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    class_ids, class_texts = [], []

    for ann in anns:
        norm = resize_bbox(ann["bbox"], width, height)
        ymins.append(norm[0])
        xmins.append(norm[1])
        ymaxs.append(norm[2])
        xmaxs.append(norm[3])
        class_ids.append(ann["category_id"])
        class_texts.append(b"canadian_goose")

    feature = {
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),
        "image/source_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_record["id"]).encode()])),
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
        "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=class_ids)),
        "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=class_texts)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(
    split_name: str,
    id_to_image: dict,
    id_to_anns: dict,
    images_dir: Path,
    output_dir: Path,
    num_shards: int,
) -> None:
    image_ids = list(id_to_image.keys())
    shard_size = math.ceil(len(image_ids) / num_shards)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for shard_idx in range(num_shards):
        shard_ids = image_ids[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        if not shard_ids:
            continue

        shard_path = output_dir / f"goose_{split_name}-{shard_idx:05d}-of-{num_shards:05d}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            for img_id in shard_ids:
                img_record = id_to_image[img_id]
                anns = id_to_anns.get(img_id, [])
                try:
                    example = create_tf_example(img_record, anns, images_dir)
                    writer.write(example.SerializeToString())
                    count += 1
                except Exception as e:
                    print(f"  [WARN] Skipping {img_record['file_name']}: {e}")

    print(f"  {split_name}: wrote {count} records across {num_shards} shards")


def main():
    parser = argparse.ArgumentParser(description="Generate TFRecords from COCO JSON datasets")
    parser.add_argument("--images_dir", required=True, help="Directory containing all images")
    parser.add_argument("--annotations_dir", required=True, help="Directory with train/val/test JSON files")
    parser.add_argument("--output_dir", required=True, help="Output directory for TFRecord files")
    parser.add_argument("--num_shards", type=int, default=4, help="Number of shards per split")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)

    for split in ("train", "val", "test"):
        ann_file = annotations_dir / f"{split}_annotations.json"
        if not ann_file.exists():
            print(f"[SKIP] {ann_file} not found")
            continue

        print(f"Processing {split}...")
        id_to_image, id_to_anns = load_coco(ann_file)
        write_tfrecords(split, id_to_image, id_to_anns, images_dir, output_dir, args.num_shards)

    print("TFRecord generation complete.")


if __name__ == "__main__":
    main()
