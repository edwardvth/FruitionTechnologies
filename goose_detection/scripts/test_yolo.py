"""
test_yolo.py
Runs the trained YOLOv11s goose detector on images or videos
and saves annotated results to results/.

Usage:
    python scripts/test_yolo.py --image path/to/goose.jpg
    python scripts/test_yolo.py --video path/to/video.mp4
    python scripts/test_yolo.py --folder path/to/folder
    python scripts/test_yolo.py --test_split --n 20
    python scripts/test_yolo.py --image goose.jpg --conf 0.3
"""

import argparse
import json
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR    = Path(__file__).parent.parent
MODELS_DIR  = BASE_DIR / "models"
DATA_DIR    = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_WEIGHTS = MODELS_DIR / "goose_yolo" / "weights" / "best.pt"
DEFAULT_CONF    = 0.5


def process_image(img_path: Path, model, conf_threshold: float, verbose=True):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] Cannot read: {img_path}")
        return

    results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
    result  = results[0]

    boxes = result.boxes
    if verbose:
        if boxes and len(boxes):
            print(f"  {img_path.name}: {len(boxes)} goose(s) detected")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = box.conf[0].item()
                print(f"    [{i}] conf={score:.3f}  box=({int(x1)},{int(y1)})->({int(x2)},{int(y2)})")
        else:
            print(f"  {img_path.name}: no detections above {conf_threshold:.2f}")

    # Draw and save
    annotated = result.plot()   # BGR image with boxes drawn
    out_path  = RESULTS_DIR / f"{img_path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved → {out_path}")


def process_video(video_path: Path, model, conf_threshold: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = RESULTS_DIR / f"{video_path.stem}_detected.mp4"
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (width, height))

    frame_num   = 0
    detections  = 0
    print(f"Processing {total} frames from {video_path.name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results  = model.predict(source=frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

        n = len(results[0].boxes) if results[0].boxes else 0
        detections += n
        frame_num  += 1
        if frame_num % 100 == 0:
            print(f"  {frame_num}/{total} frames...")

    cap.release()
    writer.release()
    print(f"  Total goose detections across all frames: {detections}")
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image",      help="Path to a single image")
    source.add_argument("--video",      help="Path to a video file")
    source.add_argument("--folder",     help="Path to a folder of images")
    source.add_argument("--test_split", action="store_true",
                        help="Sample from held-out test split")
    parser.add_argument("--n",          type=int, default=20)
    parser.add_argument("--conf",       type=float, default=DEFAULT_CONF)
    parser.add_argument("--weights",    default=str(DEFAULT_WEIGHTS))
    args = parser.parse_args()

    print(f"Loading weights: {args.weights}")
    model = YOLO(args.weights)
    print(f"Confidence threshold: {args.conf}\n")

    if args.image:
        process_image(Path(args.image), model, args.conf)

    elif args.video:
        process_video(Path(args.video), model, args.conf)

    elif args.folder:
        exts   = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [p for p in Path(args.folder).rglob("*") if p.suffix.lower() in exts]
        print(f"Found {len(images)} images in {args.folder}")
        for img_path in sorted(images):
            process_image(img_path, model, args.conf)

    elif args.test_split:
        ann_path = DATA_DIR / "test_annotations.json"
        with open(ann_path) as f:
            coco = json.load(f)
        images = coco["images"]
        sample = random.sample(images, min(args.n, len(images)))
        print(f"Testing on {len(sample)} random images from test split\n")
        for rec in sample:
            img_path = Path(rec["file_name"])
            if not img_path.exists():
                img_path = DATA_DIR / "images" / Path(rec["file_name"]).name
            process_image(img_path, model, args.conf)

    print(f"\nAll results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
