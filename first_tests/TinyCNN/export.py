"""
Export TinyCNN to INT8 TFLite for Arduino Nano 33 BLE (TFLite Micro).

Pipeline
────────
  tinycnn.pth  →  model.onnx  →  TF SavedModel  →  tiny_cnn_int8.tflite

Tools used (all available in the project venv):
  torch.onnx.export   — PyTorch → ONNX
  onnxslim            — optional ONNX graph optimisation
  onnx2tf             — ONNX → TF SavedModel
  tf.lite             — SavedModel → INT8-quantised TFLite

Usage
─────
  python export.py                         # uses default paths
  python export.py --weights my_model.pth  # custom checkpoint
  python export.py --skip-onnxslim         # skip optional ONNX slim step
"""

import argparse
import os
import sys

import struct

import numpy as np
import onnx.helper
import torch

# onnx_graphsurgeon (onnx2tf dependency) references onnx.helper.float32_to_bfloat16
# which was removed in onnx 1.16+.  Restore it before onnx2tf is imported.
if not hasattr(onnx.helper, "float32_to_bfloat16"):
    def _float32_to_bfloat16(val: float) -> int:
        packed = struct.pack(">f", val)   # big-endian float32
        return struct.unpack(">H", packed[:2])[0]  # upper 2 bytes = bfloat16
    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16

# ── path hack ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import TinyCNN, INPUT_SIZE

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


# ────────────────────────────────────────────────────────────────────────────
# Step 1 – PyTorch → ONNX
# ────────────────────────────────────────────────────────────────────────────

def export_onnx(weights_path: str, onnx_path: str) -> None:
    print(f"[1/4] Exporting {weights_path} → {onnx_path}")
    model = TinyCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    # dynamo=False forces the legacy TorchScript-based exporter, which avoids
    # the onnxscript InlinePass crash that occurs with AvgPool2d in PyTorch 2.x.
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"    Saved {onnx_path}  ({os.path.getsize(onnx_path) / 1024:.1f} kB)")


# ────────────────────────────────────────────────────────────────────────────
# Step 2 – ONNX → slim ONNX (optional)
# ────────────────────────────────────────────────────────────────────────────

def slim_onnx(onnx_path: str, slim_path: str) -> str:
    print(f"[2/4] Slimming ONNX graph → {slim_path}")
    try:
        from onnxslim import slim
        slim(onnx_path, slim_path)
        print(f"    Saved {slim_path}  ({os.path.getsize(slim_path) / 1024:.1f} kB)")
        return slim_path
    except Exception as e:
        print(f"    [WARN] onnxslim failed ({e}), using un-slimmed ONNX.")
        return onnx_path


# ────────────────────────────────────────────────────────────────────────────
# Step 3 – ONNX → TF SavedModel (via onnx2tf)
# ────────────────────────────────────────────────────────────────────────────

def export_saved_model(onnx_path: str, saved_model_dir: str) -> None:
    print(f"[3/4] Converting {onnx_path} → TF SavedModel at {saved_model_dir}")
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        not_use_onnxsim=False,
        verbosity="error",
    )
    print(f"    SavedModel written to {saved_model_dir}/")


# ────────────────────────────────────────────────────────────────────────────
# Step 4 – SavedModel → INT8 TFLite
# ────────────────────────────────────────────────────────────────────────────

def _make_calibration_dataset(saved_model_dir: str):
    """
    Returns a representative dataset generator for INT8 calibration.

    Tries to reuse the project's existing calibration NPY file (resized to
    INPUT_SIZE×INPUT_SIZE).  Falls back to random noise if the file is not
    found.
    """
    import tensorflow as tf

    # Try the pre-existing calibration data (20×128×128×3 float32)
    cal_npy = os.path.join(_REPO_ROOT, "calibration_image_sample_data_20x128x128x3_float32.npy")
    if os.path.exists(cal_npy):
        data = np.load(cal_npy)  # shape: (20, 128, 128, 3), range [0, 1]
        # Resize to INPUT_SIZE × INPUT_SIZE
        resized = tf.image.resize(data, [INPUT_SIZE, INPUT_SIZE]).numpy()
        print(f"    Using calibration data from {cal_npy} (resized to {INPUT_SIZE}×{INPUT_SIZE})")
    else:
        print("    [WARN] Calibration NPY not found; using random noise for quantization.")
        resized = np.random.rand(20, INPUT_SIZE, INPUT_SIZE, 3).astype("float32")

    def representative_dataset():
        for img in resized:
            yield [img[np.newaxis].astype("float32")]

    return representative_dataset


def export_tflite_int8(saved_model_dir: str, tflite_path: str) -> None:
    print(f"[4/4] Quantising → {tflite_path}")
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _make_calibration_dataset(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    limit_kb = 500
    status = "✓ FITS" if size_kb < limit_kb else "✗ TOO LARGE"
    print(f"    Saved {tflite_path}")
    print(f"    Size: {size_kb:.1f} kB  (limit {limit_kb} kB)  [{status}]")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Export TinyCNN to INT8 TFLite")
    p.add_argument("--weights", default=os.path.join(_HERE, "tinycnn.pth"),
                   help="Path to trained .pth weights")
    p.add_argument("--out-dir", default=_HERE,
                   help="Directory to write intermediate and final files")
    p.add_argument("--skip-onnxslim", action="store_true",
                   help="Skip ONNX graph slimming step")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    onnx_path       = os.path.join(out, "tinycnn.onnx")
    slim_path       = os.path.join(out, "tinycnn_slim.onnx")
    saved_model_dir = os.path.join(out, "tinycnn_saved_model")
    tflite_path     = os.path.join(out, "tiny_cnn_int8.tflite")

    if not os.path.exists(args.weights):
        sys.exit(f"[ERROR] Weights not found: {args.weights}\n"
                 "Run distill.py first to train the model.")

    export_onnx(args.weights, onnx_path)

    active_onnx = onnx_path
    if not args.skip_onnxslim:
        active_onnx = slim_onnx(onnx_path, slim_path)

    export_saved_model(active_onnx, saved_model_dir)
    export_tflite_int8(saved_model_dir, tflite_path)

    print(f"\nDone.  INT8 TFLite model: {tflite_path}")
    print("Next steps:")
    print("  1. Run evaluate.py to compare accuracy vs YOLOv26n baseline.")
    print("  2. Run to_c_array.py to generate the Arduino C header.")


if __name__ == "__main__":
    main()
