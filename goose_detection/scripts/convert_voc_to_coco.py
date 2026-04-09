"""
convert_voc_to_coco.py
Converts Pascal VOC XML annotations to COCO JSON format.
Used for the steggie3/goose-dataset which uses VOC format with "goose-head" class.
All classes are remapped to a single class: "canadian_goose" (id=1).

Usage:
    python scripts/convert_voc_to_coco.py \
        --images_dir data/raw/steggie3/images \
        --annotations_dir data/raw/steggie3/annotations \
        --output data/raw/steggie3/coco_annotations.json
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image


GOOSE_CLASS = {"id": 1, "name": "canadian_goose", "supercategory": "animal"}


def parse_voc_xml(xml_path: Path) -> dict:
    """Parse a single Pascal VOC XML file and return image + bbox info."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", default=xml_path.stem + ".jpg")
    size = root.find("size")
    width = int(size.findtext("width", default=0))
    height = int(size.findtext("height", default=0))

    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))
        # COCO format: [x, y, width, height]
        bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

    return {"filename": filename, "width": width, "height": height, "bboxes": bboxes}


def resolve_image_size(images_dir: Path, filename: str, fallback_w: int, fallback_h: int):
    """If VOC XML has 0x0 dimensions, read actual size from image file."""
    if fallback_w > 0 and fallback_h > 0:
        return fallback_w, fallback_h
    img_path = images_dir / filename
    if img_path.exists():
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    return fallback_w, fallback_h


def convert(images_dir: Path, annotations_dir: Path, output_path: Path) -> None:
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {annotations_dir}")

    coco = {
        "info": {"description": "Goose dataset converted from Pascal VOC", "version": "1.0"},
        "licenses": [],
        "categories": [GOOSE_CLASS],
        "images": [],
        "annotations": [],
    }

    image_id = 1
    annotation_id = 1

    for xml_file in xml_files:
        parsed = parse_voc_xml(xml_file)
        width, height = resolve_image_size(
            images_dir, parsed["filename"], parsed["width"], parsed["height"]
        )

        coco["images"].append(
            {
                "id": image_id,
                "file_name": parsed["filename"],
                "width": width,
                "height": height,
            }
        )

        for bbox in parsed["bboxes"]:
            x, y, w, h = bbox
            area = w * h
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Converted {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC to COCO JSON")
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--annotations_dir", required=True, help="Directory containing VOC XML files")
    parser.add_argument("--output", required=True, help="Output COCO JSON file path")
    args = parser.parse_args()

    convert(
        images_dir=Path(args.images_dir),
        annotations_dir=Path(args.annotations_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
