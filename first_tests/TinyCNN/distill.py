"""
Training script for TinyCNN with knowledge distillation from YOLOv26 (best.pt).

How the teacher works
─────────────────────
best.pt is a YOLO *detection* model trained to detect fire/other/smoke boxes.
For each training image we run it at low confidence (conf=0.01) to extract
a weak signal per class, then turn that into a soft probability vector:

  for each class c:  teacher_score[c] = max confidence among all detected boxes of class c
  soft_probs = softmax(teacher_scores / T)   (T=4 smooths the distribution)

These soft probs are precomputed once before training starts (fast, ~3 min for
~7 600 images on CPU) so the training loop only does student forward/backward.

Distillation loss
─────────────────
  L = alpha * T² * KLDiv(student_soft || teacher_soft)
    + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

alpha=0.7 means the teacher's soft targets dominate, which transfers the most
knowledge.  Hard labels come from the dataset folder structure (dominant class
by bounding-box area, assigned by prepare_dataset.py).

Dataset layout expected (created by prepare_dataset.py)
────────────────────────────────────────────────────────
  TinyCNN_data/
    train/ fire/ other/ smoke/
    val/   fire/ other/ smoke/

Usage
─────
  python distill.py --data /path/to/TinyCNN_data --epochs 50
  python distill.py --data /path/to/TinyCNN_data --no-distill  # plain CE
  python distill.py --data /path/to/TinyCNN_data --resume tinycnn.pth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(__file__))
from model import TinyCNN, INPUT_SIZE, CLASS_NAMES

_HERE = Path(__file__).parent


# ────────────────────────────────────────────────────────────────────────────
# Teacher: precompute soft labels from YOLOv26 detection model
# ────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x / temperature
    e = np.exp(x - x.max())
    return e / e.sum()


def precompute_soft_labels(
    weights_path: str,
    image_paths: list[str],
    temperature: float = 4.0,
    conf_thresh: float = 0.01,
) -> dict[str, np.ndarray]:
    """
    Run the YOLO detection teacher on every training image and return a dict
    mapping absolute image path → soft probability vector (shape: [3,]).

    The teacher processes images directly from disk (no normalisation),
    exactly as in the original main.py inference pipeline.
    """
    from ultralytics import YOLO

    print(f"Loading teacher: {weights_path}")
    teacher = YOLO(weights_path)
    num_classes = len(CLASS_NAMES)  # 3

    cache: dict[str, np.ndarray] = {}
    total = len(image_paths)

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Teacher inference {i+1}/{total} …")

        # Exactly how main.py calls the model: pass the file path directly
        results = teacher.predict(img_path, verbose=False, conf=conf_thresh)

        scores = np.zeros(num_classes, dtype=np.float32)

        result = results[0]
        if result.boxes is not None and len(result.boxes):
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs   = result.boxes.conf.cpu().numpy().astype(np.float32)
            for cls_id, conf in zip(cls_ids, confs):
                if 0 <= cls_id < num_classes and conf > scores[cls_id]:
                    scores[cls_id] = conf

        cache[img_path] = _softmax(scores, temperature)

    print(f"  Done. Cached {len(cache)} soft-label vectors.")
    return cache


# ────────────────────────────────────────────────────────────────────────────
# Dataset wrapper: adds soft labels to ImageFolder
# ────────────────────────────────────────────────────────────────────────────

class DistillDataset(Dataset):
    """
    Wraps ImageFolder and attaches precomputed teacher soft labels.

    Returns: (image_tensor, hard_label, soft_label_tensor)
    """

    def __init__(self, folder: ImageFolder, transform, soft_labels: dict[str, np.ndarray]):
        self.folder      = folder
        self.transform   = transform
        self.soft_labels = soft_labels
        self.uniform     = np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        img_path, hard_label = self.folder.imgs[idx]

        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)

        soft = self.soft_labels.get(img_path, self.uniform)
        return img_t, hard_label, torch.tensor(soft, dtype=torch.float32)


# ────────────────────────────────────────────────────────────────────────────
# Transforms
# ────────────────────────────────────────────────────────────────────────────

def build_transforms(is_train: bool):
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ────────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    kl       = nn.KLDivLoss(reduction="batchmean")
    s_soft   = torch.log_softmax(student_logits / temperature, dim=-1)
    kd_loss  = temperature ** 2 * kl(s_soft, teacher_probs)
    ce_loss  = nn.functional.cross_entropy(student_logits, hard_labels)
    return alpha * kd_loss + (1.0 - alpha) * ce_loss


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir   = data_root / "val"

    if not train_dir.is_dir():
        sys.exit(f"[ERROR] Training directory not found: {train_dir}\n"
                 "Run prepare_dataset.py first.")

    # ImageFolder just to discover file paths and hard labels
    train_folder = ImageFolder(str(train_dir))
    val_folder   = ImageFolder(str(val_dir)) if val_dir.is_dir() else None

    print(f"Class mapping: {train_folder.class_to_idx}")

    # ── Precompute teacher soft labels ────────────────────────────────────────
    soft_labels: dict[str, np.ndarray] = {}
    if not args.no_distill:
        teacher_path = str(_HERE / ".." / "YOLOv26" / "best.pt")
        teacher_path = str(Path(teacher_path).resolve())
        if not Path(teacher_path).exists():
            print(f"[WARN] Teacher not found at {teacher_path}. Using standard CE loss.")
        else:
            train_paths = [p for p, _ in train_folder.imgs]
            soft_labels = precompute_soft_labels(
                teacher_path,
                train_paths,
                temperature=args.temperature,
                conf_thresh=0.01,
            )

    # ── Datasets / loaders ───────────────────────────────────────────────────
    train_ds = DistillDataset(train_folder, build_transforms(True), soft_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = None
    if val_folder:
        val_ds = DistillDataset(val_folder, build_transforms(False), {})
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    # ── Model ────────────────────────────────────────────────────────────────
    model = TinyCNN(num_classes=len(CLASS_NAMES)).to(device)
    if args.resume and Path(args.resume).exists():
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_distill = bool(soft_labels)
    out_path = str(_HERE / "tinycnn.pth")
    best_val_acc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = total = 0

        for imgs, hard_labels, soft in train_loader:
            imgs        = imgs.to(device)
            hard_labels = hard_labels.to(device)
            soft        = soft.to(device)

            logits = model(imgs)

            if use_distill:
                loss = distillation_loss(logits, soft, hard_labels,
                                         temperature=args.temperature,
                                         alpha=args.alpha)
            else:
                loss = nn.functional.cross_entropy(logits, hard_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == hard_labels).sum().item()
            total      += imgs.size(0)

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_str = ""
        if val_loader:
            model.eval()
            v_correct = v_total = 0
            with torch.no_grad():
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(1)
                    v_correct += (preds == labels).sum().item()
                    v_total   += imgs.size(0)
            val_acc = v_correct / v_total
            val_str = f"  val_acc={val_acc:.3f}"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), out_path)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={total_loss/total:.4f}  "
              f"train_acc={correct/total:.3f}{val_str}")

    if not val_loader:
        torch.save(model.state_dict(), out_path)

    print(f"\nSaved → {out_path}")
    print("Next: python export.py")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        required=True,
                   help="Root of TinyCNN_data/ (must have train/ sub-dir)")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--alpha",       type=float, default=0.7,
                   help="KD loss weight: 0=pure CE, 1=pure KD (default 0.7)")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Distillation temperature (default 4.0)")
    p.add_argument("--no-distill",  action="store_true",
                   help="Disable KD; train with plain cross-entropy")
    p.add_argument("--resume",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
