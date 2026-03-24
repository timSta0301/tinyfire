"""
Build MobileNetV1 Alpha 0.25 for fire classification and export to INT8 TFLite.

The base model (ImageNet weights) is downloaded automatically by Keras.
Add your own training step before export for accurate INT8 results.

Usage:
    python download.py
    python download.py --classes fire smoke neutral --imgsz 128 --calib ../../data
"""

import os
import argparse
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(num_classes: int, imgsz: int, alpha: float = 0.25) -> Model:
    base = MobileNet(
        alpha=alpha,
        input_shape=(imgsz, imgsz, 3),
        include_top=False,
        weights="imagenet",
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=x)


def representative_dataset(calib_dir: str, imgsz: int):
    """Yields calibration batches from all images in calib_dir."""
    paths = glob.glob(os.path.join(calib_dir, "*.*"))
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not paths:
        raise FileNotFoundError(f"No images found in calibration dir: {calib_dir}")
    print(f"Calibration: {len(paths)} images")

    def gen():
        for p in paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            img = preprocess_input(img.astype(np.float32))  # [0,255] → [-1,1]
            yield [img[np.newaxis]]  # [1, H, W, 3]

    return gen


def export_tflite(model: Model, calib_dir: str, imgsz: int, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(calib_dir, imgsz)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"TFLite INT8 saved: {out_path} ({size_kb:.1f} KB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=["fire", "smoke", "neutral"],
                   help="Class names in label order")
    p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.25,
                   help="MobileNetV1 width multiplier")
    p.add_argument("--calib", default=os.path.join(_DIR, "..", "..", "data"),
                   help="Directory of calibration images for INT8 quantization")
    return p.parse_args()


def main():
    args = parse_args()
    calib = os.path.abspath(args.calib)
    num_classes = len(args.classes)

    print(f"MobileNetV1 alpha={args.alpha} | {args.imgsz}x{args.imgsz} | {num_classes} classes: {args.classes}")

    model = build_model(num_classes, args.imgsz, args.alpha)
    model.summary(line_length=80)

    # Save Keras model for fine-tuning
    keras_path = os.path.join(_DIR, "mobilenetv1_fire.keras")
    model.save(keras_path)
    print(f"Keras model saved: {keras_path}")

    # Save class names
    names_path = os.path.join(_DIR, "names.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(args.classes))
    print(f"Class names saved: {names_path}")

    # Export INT8 TFLite
    tflite_path = os.path.join(_DIR, "mobilenetv1_fire_int8.tflite")
    export_tflite(model, calib, args.imgsz, tflite_path)


if __name__ == "__main__":
    main()
