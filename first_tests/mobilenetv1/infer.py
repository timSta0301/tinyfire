"""
Run MobileNetV1 INT8 TFLite inference on a fire image.
No Keras/ultralytics dependency — mirrors what runs on the microcontroller.

Usage:
    python infer.py --image ../../data/image.png
    python infer.py --image ../../data/image.png --conf 0.4 --top-k 3
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.join(_DIR, "mobilenetv1_fire_int8.tflite")
_DEFAULT_NAMES = os.path.join(_DIR, "names.txt")


def load_names(path: str) -> dict[int, str]:
    if os.path.isfile(path):
        with open(path) as f:
            return {i: line.strip() for i, line in enumerate(f) if line.strip()}
    return {0: "fire", 1: "smoke", 2: "neutral"}


def preprocess(image_path: str, h: int, w: int, dtype) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, 0)  # [1, H, W, 3]
    if dtype == np.uint8:
        return img.astype(np.uint8)
    return preprocess_input(img.astype(np.float32))  # [0,255] → [-1,1]


def dequantize(output: np.ndarray, quantization: tuple) -> np.ndarray:
    scale, zero_point = quantization
    return (output.astype(np.float32) - zero_point) * scale


def infer(model_path: str, image_path: str, top_k: int = 3, conf: float = 0.0):
    names = load_names(_DEFAULT_NAMES)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    h, w = in_det["shape"][1], in_det["shape"][2]
    img = preprocess(image_path, h, w, in_det["dtype"])

    interpreter.set_tensor(in_det["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(out_det["index"])[0]  # [num_classes]

    if out_det["dtype"] == np.uint8:
        output = dequantize(output, out_det["quantization"])

    top_indices = np.argsort(output)[::-1][:top_k]

    print(f"\n{os.path.basename(image_path)} | model: {os.path.basename(model_path)}")
    print(f"{'Class':<16} {'Confidence':>10}")
    print("-" * 28)
    for idx in top_indices:
        confidence = float(output[idx])
        if confidence < conf:
            break
        print(f"{names.get(int(idx), idx):<16} {confidence:>10.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--conf", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.model, args.image, top_k=args.top_k, conf=args.conf)
