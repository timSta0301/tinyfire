"""
Evaluate TinyCNN INT8 TFLite model and compare against test images.

Two evaluation modes
────────────────────
1. Single image (quick sanity check):
     python evaluate.py --image ../../data/image5.png

2. Folder benchmark (accuracy over a labelled directory):
     python evaluate.py --data /path/to/val_dir
     (expects val_dir/fire/*.png, val_dir/smoke/*.png, val_dir/other/*.png)

Optionally compare against the baseline YOLOv26n-cls INT8 model
(note: that model was trained on ImageNet 1000 classes so its class IDs
differ; comparison is shown as-is for reference).

Usage
─────
  python evaluate.py --image ../../data/image5.png
  python evaluate.py --data /path/to/val --model tiny_cnn_int8.tflite
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import INPUT_SIZE, CLASS_NAMES

_HERE = os.path.dirname(os.path.abspath(__file__))


# ────────────────────────────────────────────────────────────────────────────
# TFLite inference helper
# ────────────────────────────────────────────────────────────────────────────

class TFLiteClassifier:
    """Thin wrapper around a TFLite INT8 classification model."""

    def __init__(self, model_path: str):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp  = self.interpreter.get_input_details()[0]
        self.out  = self.interpreter.get_output_details()[0]
        self.size = self.inp["shape"][1]  # spatial size (assumes square)

        # Quantisation parameters for INT8 models
        self.inp_scale, self.inp_zero = self.inp["quantization"]
        self.out_scale, self.out_zero = self.out["quantization"]

    def preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        """Resize, convert to RGB, normalise to [0,1], then quantise to INT8."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size)).astype("float32") / 255.0
        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std  = np.array([0.229, 0.224, 0.225], dtype="float32")
        rgb  = (rgb - mean) / std
        # Scale to INT8
        if self.inp_scale != 0:
            rgb = rgb / self.inp_scale + self.inp_zero
        return np.clip(np.round(rgb), -128, 127).astype("int8")[np.newaxis]

    def predict(self, bgr_img: np.ndarray) -> np.ndarray:
        """Returns float32 logits / scores of shape (num_classes,)."""
        inp_data = self.preprocess(bgr_img)
        self.interpreter.set_tensor(self.inp["index"], inp_data)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.out["index"])[0].astype("float32")
        # De-quantise INT8 output
        if self.out_scale != 0:
            raw = (raw - self.out_zero) * self.out_scale
        return raw

    def classify(self, bgr_img: np.ndarray) -> tuple[str, float]:
        """Returns (class_name, confidence)."""
        scores = self.predict(bgr_img)
        probs  = softmax(scores)
        idx    = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ────────────────────────────────────────────────────────────────────────────
# Single image evaluation
# ────────────────────────────────────────────────────────────────────────────

def eval_single(model_path: str, image_path: str) -> None:
    print(f"\nModel : {model_path}")
    print(f"Image : {image_path}")

    clf = TFLiteClassifier(model_path)
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"[ERROR] Cannot load image: {image_path}")

    scores = clf.predict(img)
    probs  = softmax(scores)
    order  = np.argsort(probs)[::-1]

    print("\nTop predictions:")
    for i in order:
        bar = "█" * int(probs[i] * 30)
        print(f"  {CLASS_NAMES[i]:8s} {probs[i]*100:5.1f}%  {bar}")

    # Visualise
    label, conf = CLASS_NAMES[order[0]], probs[order[0]]
    vis = img.copy()
    cv2.putText(vis, f"{label}: {conf:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    cv2.imshow("TinyCNN", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ────────────────────────────────────────────────────────────────────────────
# Folder benchmark
# ────────────────────────────────────────────────────────────────────────────

def eval_folder(model_path: str, data_dir: str) -> None:
    clf = TFLiteClassifier(model_path)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    results = {}  # class_name → {correct, total}

    for class_name in CLASS_NAMES:
        class_dir = Path(data_dir) / class_name
        if not class_dir.is_dir():
            print(f"[WARN] Skipping missing directory: {class_dir}")
            continue

        images = [p for p in class_dir.iterdir() if p.suffix.lower() in extensions]
        if not images:
            print(f"[WARN] No images in {class_dir}")
            continue

        correct = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            pred_class, _ = clf.classify(img)
            if pred_class == class_name:
                correct += 1

        results[class_name] = {"correct": correct, "total": len(images)}

    # Print summary
    print(f"\n{'Class':10s}  {'Correct':>8s}  {'Total':>8s}  {'Accuracy':>10s}")
    print("─" * 45)
    total_correct = total_all = 0
    for cls, r in results.items():
        acc = r["correct"] / r["total"] if r["total"] else 0.0
        print(f"{cls:10s}  {r['correct']:8d}  {r['total']:8d}  {acc*100:9.1f}%")
        total_correct += r["correct"]
        total_all += r["total"]

    if total_all:
        overall = total_correct / total_all
        print("─" * 45)
        print(f"{'Overall':10s}  {total_correct:8d}  {total_all:8d}  {overall*100:9.1f}%")

    size_kb = os.path.getsize(model_path) / 1024
    print(f"\nModel size: {size_kb:.1f} kB")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TinyCNN TFLite model")
    p.add_argument("--model", default=os.path.join(_HERE, "tiny_cnn_int8.tflite"),
                   help="Path to tiny_cnn_int8.tflite")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image for quick test")
    group.add_argument("--data",  help="Path to labelled val directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"[ERROR] Model not found: {args.model}\n"
                 "Run export.py first to generate tiny_cnn_int8.tflite.")

    if args.image:
        eval_single(args.model, args.image)
    else:
        eval_folder(args.model, args.data)
