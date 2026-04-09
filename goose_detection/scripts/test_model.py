"""
test_model.py
Runs the trained EfficientDetD0 goose detector on one or more images
and saves annotated results so you can visually verify detections.

Usage:
    # Test on a single image:
    python scripts/test_model.py --image path/to/goose.jpg

    # Test on every image in a folder:
    python scripts/test_model.py --folder path/to/folder

    # Test on the held-out test split (random sample):
    python scripts/test_model.py --test_split --n 20

    # Lower confidence threshold if nothing is detected:
    python scripts/test_model.py --image goose.jpg --conf 0.3
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from effdet import create_model
from effdet.bench import DetBenchPredict

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE        = 512
DEFAULT_CONF    = 0.5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint_path: Path, device: torch.device) -> DetBenchPredict:
    net = create_model(
        "tf_efficientdet_d0",
        bench_task="",
        num_classes=1,
        pretrained=False,
    )
    bench = DetBenchPredict(net)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"]

    # Strip "model." prefix added by DetBenchTrain if present
    cleaned = {}
    for k, v in state.items():
        new_k = k[len("model."):] if k.startswith("model.") else k
        cleaned[new_k] = v

    bench.model.load_state_dict(cleaned, strict=False)
    bench.to(device).eval()
    return bench


def preprocess(bgr_image: np.ndarray) -> tuple[torch.Tensor, float, float]:
    orig_h, orig_w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)  # [1,3,H,W]
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE
    return tensor, scale_x, scale_y


def run_inference(model, tensor: torch.Tensor, device: torch.device,
                  scale_x: float, scale_y: float, conf_threshold: float):
    tensor = tensor.to(device)
    img_size  = torch.tensor([[IMG_SIZE, IMG_SIZE]], dtype=torch.float32).to(device)
    img_scale = torch.tensor([1.0]).to(device)

    with torch.no_grad():
        detections = model(tensor, img_info={"img_size": img_size, "img_scale": img_scale})

    # detections: [1, max_det, 6] — [y1, x1, y2, x2, score, class]
    dets = detections[0].cpu().numpy()
    results = []
    for det in dets:
        y1, x1, y2, x2, score, cls = det
        if score < conf_threshold:
            continue
        # Scale back to original image size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        results.append((x1, y1, x2, y2, float(score)))
    return results


def draw_and_save(bgr_image: np.ndarray, detections: list, out_path: Path):
    annotated = bgr_image.copy()
    for (x1, y1, x2, y2, score) in detections:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Goose {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(out_path), annotated)
    return annotated


def process_image(img_path: Path, model, device, conf_threshold, verbose=True):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"[SKIP] Cannot read: {img_path}")
        return

    tensor, sx, sy = preprocess(bgr)
    detections = run_inference(model, tensor, device, sx, sy, conf_threshold)

    out_path = RESULTS_DIR / f"{img_path.stem}_detected.jpg"
    draw_and_save(bgr, detections, out_path)

    if verbose:
        if detections:
            print(f"  {img_path.name}: {len(detections)} goose(s) detected")
            for i, (x1, y1, x2, y2, score) in enumerate(detections):
                print(f"    [{i}] conf={score:.3f}  box=({x1},{y1})->({x2},{y2})")
        else:
            print(f"  {img_path.name}: no detections above {conf_threshold:.2f}")
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image",      help="Path to a single image")
    source.add_argument("--folder",     help="Path to a folder of images")
    source.add_argument("--test_split", action="store_true",
                        help="Sample from the held-out test split")
    parser.add_argument("--n",          type=int, default=20,
                        help="Number of test-split images to sample (default 20)")
    parser.add_argument("--conf",       type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default {DEFAULT_CONF})")
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "goose_best.pth"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(Path(args.checkpoint), device)
    print(f"Confidence threshold: {args.conf}\n")

    if args.image:
        process_image(Path(args.image), model, device, args.conf)

    elif args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [p for p in Path(args.folder).rglob("*") if p.suffix.lower() in exts]
        print(f"Found {len(images)} images in {args.folder}")
        for img_path in sorted(images):
            process_image(img_path, model, device, args.conf)

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
            process_image(img_path, model, device, args.conf)

    print(f"\nAll results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
