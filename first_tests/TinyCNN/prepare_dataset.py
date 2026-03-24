"""
Converts the Roboflow YOLO detection dataset to an ImageFolder classification
dataset that distill.py can consume.

Source layout (YOLO detection format)
──────────────────────────────────────
  datasets/Fire Detection.v1i.yolo26/
    train/
      images/  *.jpg
      labels/  *.txt   (one line per box: class_id cx cy w h, all normalised)

Target layout (PyTorch ImageFolder format)
──────────────────────────────────────────
  datasets/TinyCNN_data/
    train/
      fire/   (symlinks or copies of images whose dominant class is fire)
      other/
      smoke/
    val/
      fire/
      other/
      smoke/

Class label assignment
──────────────────────
Each image is assigned the class whose bounding boxes cover the largest
total area in that image.  Images with no annotations are skipped.

The original train split is split 85/15 into train/val (stratified by class).

Usage
─────
  python prepare_dataset.py
  python prepare_dataset.py --dataset-dir /custom/path --out-dir /custom/out
  python prepare_dataset.py --val-split 0.2  # use 20% for validation
  python prepare_dataset.py --copy           # copy files instead of symlinking
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["fire", "other", "smoke"]  # indices 0, 1, 2

_HERE       = Path(__file__).parent
_REPO_ROOT  = _HERE.parent.parent.parent          # Green AI Hackathon/
_DATASET    = _REPO_ROOT / "datasets" / "Fire Detection.v1i.yolo26"
_OUT_DIR    = _REPO_ROOT / "datasets" / "TinyCNN_data"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def dominant_class(label_path: Path) -> int | None:
    """
    Returns the class index whose boxes cover the largest total area,
    or None if the label file is empty / missing.
    """
    area_per_class = defaultdict(float)
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                w, h   = float(parts[3]), float(parts[4])
                area_per_class[cls_id] += w * h
    except FileNotFoundError:
        return None

    if not area_per_class:
        return None
    return max(area_per_class, key=area_per_class.get)


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def prepare(args):
    dataset_dir = Path(args.dataset_dir)
    out_dir     = Path(args.out_dir)
    images_dir  = dataset_dir / "train" / "images"
    labels_dir  = dataset_dir / "train" / "labels"

    if not images_dir.is_dir():
        sys.exit(f"[ERROR] Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"[ERROR] Labels directory not found: {labels_dir}")

    # ── Assign class to each image ──────────────────────────────────────────
    print("Scanning annotations …")
    buckets = defaultdict(list)   # class_id → [image_path, ...]
    skipped = 0

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in img_extensions:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        cls = dominant_class(label_path)
        if cls is None:
            skipped += 1
            continue
        buckets[cls].append(img_path)

    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"  class {cls_id} ({name:6s}): {len(buckets[cls_id]):5d} images")
    if skipped:
        print(f"  skipped (no annotations): {skipped}")

    # ── Stratified train / val split ────────────────────────────────────────
    random.seed(42)
    splits = {"train": [], "val": []}

    for cls_id, paths in buckets.items():
        random.shuffle(paths)
        n_val = max(1, int(len(paths) * args.val_split))
        splits["val"]   += [(p, cls_id) for p in paths[:n_val]]
        splits["train"] += [(p, cls_id) for p in paths[n_val:]]

    # ── Write output ─────────────────────────────────────────────────────────
    use_copy = args.copy
    mode_str = "Copying" if use_copy else "Symlinking"
    print(f"\n{mode_str} files to {out_dir} …")

    counts = defaultdict(lambda: defaultdict(int))
    for split, items in splits.items():
        for img_path, cls_id in items:
            cls_name = CLASS_NAMES[cls_id]
            dst = out_dir / split / cls_name / img_path.name
            link_or_copy(img_path, dst, use_copy)
            counts[split][cls_name] += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\nDataset ready:")
    for split in ("train", "val"):
        total = sum(counts[split].values())
        class_info = "  ".join(f"{n}={counts[split][n]}" for n in CLASS_NAMES)
        print(f"  {split:5s}: {total:5d} images  ({class_info})")

    print(f"\nOutput directory: {out_dir}")
    print("\nNext step — train TinyCNN:")
    print(f"  python first_tests/TinyCNN/distill.py --data {out_dir} --epochs 50")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert YOLO detection dataset to ImageFolder classification layout"
    )
    p.add_argument("--dataset-dir", default=str(_DATASET),
                   help="Root of the Roboflow YOLO dataset")
    p.add_argument("--out-dir", default=str(_OUT_DIR),
                   help="Output directory for ImageFolder layout")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of data to use for validation (default 0.15)")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of creating symlinks (uses more disk)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args)
