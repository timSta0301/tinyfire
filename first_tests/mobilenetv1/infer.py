"""
Run inference on the latest MobileNetV1 fire/smoke/other model.

Supports:
- Keras checkpoints (*.keras, *.h5) from training
- INT8/float TFLite (*.tflite) for deployment checks

Usage:
    python infer.py --image ../../data/image.png
    python infer.py --image ../../data/image.png --model mobilenetv1_fire_best.keras
    python infer.py --image ../../data/image.png --model mobilenetv1_fire_int8.tflite --conf 0.4 --top-k 3
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_KERAS_MODEL = os.path.join(_DIR, "mobilenetv1_fire_best.keras")
_DEFAULT_TFLITE_MODEL = os.path.join(_DIR, "mobilenetv1_fire_int8.tflite")
_DEFAULT_NAMES = os.path.join(_DIR, "names.txt")


def load_names(path: str) -> dict[int, str]:
    if os.path.isfile(path):
        with open(path) as f:
            return {i: line.strip() for i, line in enumerate(f) if line.strip()}
    return {0: "fire", 1: "smoke", 2: "neutral"}


def resolve_default_model() -> str:
    # Prefer latest Keras checkpoint for validating freshest training results.
    if os.path.isfile(_DEFAULT_KERAS_MODEL):
        return _DEFAULT_KERAS_MODEL
    return _DEFAULT_TFLITE_MODEL


def load_image_rgb(image_path: str, h: int, w: int) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    return img


def dequantize(output: np.ndarray, quantization: tuple) -> np.ndarray:
    scale, zero_point = quantization
    return (output.astype(np.float32) - zero_point) * scale


def run_tflite(model_path: str, image_path: str) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    h, w = in_det["shape"][1], in_det["shape"][2]
    img = load_image_rgb(image_path, h, w)
    img = np.expand_dims(img, 0)

    if in_det["dtype"] == np.uint8:
        input_tensor = img.astype(np.uint8)
    else:
        input_tensor = preprocess_input(img.astype(np.float32))

    interpreter.set_tensor(in_det["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(out_det["index"])[0]

    if out_det["dtype"] == np.uint8:
        output = dequantize(output, out_det["quantization"])

    return output.astype(np.float32)


def run_keras(model_path: str, image_path: str) -> np.ndarray:
    model = tf.keras.models.load_model(model_path)
    h, w = model.input_shape[1], model.input_shape[2]

    img = load_image_rgb(image_path, h, w)
    x = np.expand_dims(img.astype(np.float32), 0)
    x = preprocess_input(x)
    output = model.predict(x, verbose=0)[0]
    return output.astype(np.float32)


def infer(model_path: str, image_path: str, top_k: int = 3, conf: float = 0.0):
    names = load_names(_DEFAULT_NAMES)

    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".keras", ".h5"):
        output = run_keras(model_path, image_path)
    elif ext == ".tflite":
        output = run_tflite(model_path, image_path)
    else:
        raise ValueError(f"Unsupported model type: {model_path}")

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
    p.add_argument("--model", default=resolve_default_model())
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--conf", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.model, args.image, top_k=args.top_k, conf=args.conf)
