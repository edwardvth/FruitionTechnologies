"""
train_yolo.py
Fine-tunes YOLOv11s on the merged goose dataset using Ultralytics.

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 100 --batch 4
"""

import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

BASE_DIR   = Path(__file__).parent.parent
YOLO_DIR   = BASE_DIR / "data" / "yolo"
MODELS_DIR = BASE_DIR / "models"


def make_data_yaml():
    """Write a temporary YAML with absolute paths for Ultralytics."""
    cfg = {
        "path":  str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["canadian_goose"],
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=8,
                        help="Reduce to 4 if GPU runs out of memory")
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from models/goose_yolo/weights/last.pt")
    args = parser.parse_args()

    if args.resume:
        model = YOLO(str(MODELS_DIR / "goose_yolo" / "weights" / "last.pt"))
    else:
        model = YOLO("yolo11s.pt")   # downloads pretrained COCO weights automatically

    data_yaml = make_data_yaml()

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(MODELS_DIR),
        name="goose_yolo",
        patience=15,         # early stopping if no improvement for 15 epochs
        exist_ok=True,       # overwrite previous run folder
        device=0,            # GPU 0; falls back to CPU automatically if no GPU
        verbose=True,
    )

    best = MODELS_DIR / "goose_yolo" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")


if __name__ == "__main__":
    main()
