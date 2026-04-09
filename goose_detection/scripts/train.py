"""
train.py
Fine-tunes EfficientDetD0 on the merged goose dataset using PyTorch + effdet.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch_size 4 --lr 0.005
"""

import argparse
import json
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from effdet import create_model
from effdet.bench import DetBenchTrain
from effdet.config import get_efficientdet_config
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 512

# ── Dataset ───────────────────────────────────────────────────────────────────

class GooseDataset(Dataset):
    def __init__(self, ann_path: Path, transforms=None):
        with open(ann_path) as f:
            coco = json.load(f)

        self.transforms = transforms
        self.images = coco["images"]

        self.ann_by_image = {}
        for ann in coco.get("annotations", []):
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_record = self.images[idx]

        # Resolve image path — stored name may be bare filename or relative path
        img_path = Path(img_record["file_name"])
        if not img_path.exists():
            img_path = DATA_DIR / "images" / Path(img_record["file_name"]).name
        if not img_path.exists():
            return self._empty_sample()

        image = cv2.imread(str(img_path))
        if image is None:
            return self._empty_sample()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        anns = self.ann_by_image.get(img_record["id"], [])

        # COCO [x, y, w, h] → Pascal VOC [x1, y1, x2, y2]
        bboxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0.0, x),      max(0.0, y)
            x2, y2 = min(x + bw, w),   min(y + bh, h)
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
                labels.append(1)  # 1-indexed: 1 = canadian_goose

        if self.transforms:
            try:
                out    = self.transforms(image=image, bboxes=bboxes, labels=labels)
                image  = out["image"]
                bboxes = out["bboxes"]
                labels = out["labels"]
            except Exception:
                image  = self._resize_only(image)
                bboxes, labels = [], []

        # Build target — bbox in [y1, x1, y2, x2] pixel coords (effdet convention)
        if bboxes:
            boxes = torch.tensor(bboxes, dtype=torch.float32)
            boxes = boxes[:, [1, 0, 3, 2]]          # [x1,y1,x2,y2] → [y1,x1,y2,x2]
            cls   = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((1, 4), dtype=torch.float32)  # dummy
            cls   = torch.tensor([-1], dtype=torch.long)       # -1 = ignore

        target = {
            "bbox":      boxes,
            "cls":       cls,
            "img_size":  torch.tensor([IMG_SIZE, IMG_SIZE], dtype=torch.float32),
            "img_scale": torch.tensor(1.0),
        }
        return image, target

    def _resize_only(self, image):
        t = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return t(image=image)["image"]

    def _empty_sample(self):
        image  = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        target = {
            "bbox":      torch.zeros((1, 4), dtype=torch.float32),
            "cls":       torch.tensor([-1], dtype=torch.long),
            "img_size":  torch.tensor([IMG_SIZE, IMG_SIZE], dtype=torch.float32),
            "img_scale": torch.tensor(1.0),
        }
        return image, target


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)

    # DetBenchTrain.anchor_labeler.batch_label_anchors expects lists of tensors
    return images, {
        "bbox":      [t["bbox"]      for t in targets],
        "cls":       [t["cls"]       for t in targets],
        "img_size":  torch.stack([t["img_size"]  for t in targets]),
        "img_scale": torch.stack([t["img_scale"] for t in targets]),
    }


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                               border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.3
        ),
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=1, pretrained=True):
    # Create raw network (no bench wrapper), then wrap with DetBenchTrain
    # which internally creates AnchorLabeler for automatic anchor assignment
    config = get_efficientdet_config("tf_efficientdet_d0")
    config.update({"num_classes": num_classes, "image_size": (IMG_SIZE, IMG_SIZE)})
    net = create_model(
        "tf_efficientdet_d0",
        bench_task="",          # no bench wrapper yet
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = DetBenchTrain(net, create_labeler=True)
    return model


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = {
            "bbox":      [b.to(device) for b in targets["bbox"]],
            "cls":       [c.to(device) for c in targets["cls"]],
            "img_size":  targets["img_size"].to(device),
            "img_scale": targets["img_scale"].to(device),
        }

        optimizer.zero_grad()
        output = model(images, targets)
        loss   = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        if step % 20 == 0:
            print(f"  Epoch {epoch} [{step}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  "
                  f"cls={output['class_loss'].item():.4f}  "
                  f"box={output['box_loss'].item():.4f}")

    avg_loss = total_loss / len(loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def validate(model, loader, device, epoch, writer):
    model.eval()
    total_conf = 0.0
    total_det  = 0
    for images, targets in loader:
        images = images.to(device)
        targets = {
            "bbox":      [b.to(device) for b in targets["bbox"]],
            "cls":       [c.to(device) for c in targets["cls"]],
            "img_size":  targets["img_size"].to(device),
            "img_scale": targets["img_scale"].to(device),
        }
        output     = model(images, targets)
        detections = output.get("detections")        # [B, max_det, 6]
        if detections is not None:
            scores = detections[:, :, 4]
            mask   = scores > 0.3
            total_conf += scores[mask].sum().item()
            total_det  += mask.sum().item()

    avg_conf = total_conf / max(total_det, 1)
    writer.add_scalar("Val/avg_confidence", avg_conf, epoch)
    print(f"  Val: {total_det} detections  avg_conf={avg_conf:.3f}")
    return avg_conf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=4,   help="Reduce to 2 if OOM")
    parser.add_argument("--lr",         type=float, default=0.005)
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--resume",     type=str,   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_ds = GooseDataset(DATA_DIR / "train_annotations.json", get_train_transforms())
    val_ds   = GooseDataset(DATA_DIR / "val_annotations.json",   get_val_transforms())
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    print("Loading EfficientDetD0 (pretrained COCO weights)...")
    model = build_model(num_classes=1, pretrained=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_conf   = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_conf   = ckpt.get("best_conf", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}")

    writer = SummaryWriter(log_dir=str(MODELS_DIR / "runs"))
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        scheduler.step()
        val_conf   = validate(model, val_loader, device, epoch, writer)
        elapsed    = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_conf={val_conf:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s\n")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_conf": best_conf}
        torch.save(ckpt, MODELS_DIR / "goose_last.pth")

        if val_conf > best_conf:
            best_conf = val_conf
            torch.save(ckpt, MODELS_DIR / "goose_best.pth")
            print(f"  *** Best checkpoint saved (val_conf={best_conf:.3f}) ***\n")

    writer.close()
    print(f"Done. Best val_conf={best_conf:.3f}")
    print(f"Model: {MODELS_DIR / 'goose_best.pth'}")


if __name__ == "__main__":
    main()
