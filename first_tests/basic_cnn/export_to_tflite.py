"""
Export TinyFireCNN to INT8 TFLite for microcontroller use.

Pipeline
--------
tinyfirecnn_best.pt -> tinyfirecnn.onnx -> SavedModel -> tinyfirecnn_int8.tflite

Usage
-----
python first_tests/basic_cnn/export_to_tflite.py
python first_tests/basic_cnn/export_to_tflite.py --weights first_tests/basic_cnn/tinyfirecnn_best.pt
"""

import argparse
import os
import sys
import struct
import shutil
import subprocess

import numpy as np
import onnx.helper
import torch

# Fix for some onnx2tf dependency stacks that still reference this helper.
if not hasattr(onnx.helper, "float32_to_bfloat16"):
    def _float32_to_bfloat16(val: float) -> int:
        packed = struct.pack(">f", val)
        return struct.unpack(">H", packed[:2])[0]
    onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16

# Make local imports work when run from project root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from classifier import TinyFireCNN


def export_onnx(weights_path: str, onnx_path: str, input_size: int) -> None:
    print(f"[1/4] Exporting {weights_path} -> {onnx_path}")

    model = TinyFireCNN()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        dynamo=False,   # often more stable for compatibility
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"    Saved {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} kB)")


def slim_onnx(onnx_path: str, slim_path: str) -> str:
    print(f"[2/4] Slimming ONNX -> {slim_path}")
    try:
        from onnxslim import slim
        slim(onnx_path, slim_path)
        print(f"    Saved {slim_path} ({os.path.getsize(slim_path) / 1024:.1f} kB)")
        return slim_path
    except Exception as e:
        print(f"    [WARN] onnxslim failed ({e}), using original ONNX.")
        return onnx_path


def export_saved_model(onnx_path: str, saved_model_dir: str) -> None:
    print(f"[3/4] Converting ONNX -> SavedModel at {saved_model_dir}")
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        not_use_onnxsim=False,
        verbosity="error",
    )

    print(f"    SavedModel written to {saved_model_dir}/")


def make_representative_dataset(input_size: int, calib_npy: str | None = None):
    import tensorflow as tf

    if calib_npy and os.path.exists(calib_npy):
        data = np.load(calib_npy).astype("float32")

        # Expecting NHWC for TFLite calibration
        # If source is NCHW, convert it.
        if data.ndim == 4 and data.shape[1] == 3:
            data = np.transpose(data, (0, 2, 3, 1))

        if data.shape[1] != input_size or data.shape[2] != input_size:
            data = tf.image.resize(data, [input_size, input_size]).numpy()

        print(f"    Using calibration data from {calib_npy}, shape={data.shape}")
    else:
        print("    [WARN] No calibration NPY found; using random calibration data.")
        data = np.random.rand(20, input_size, input_size, 3).astype("float32")

    def representative_dataset():
        for img in data:
            yield [img[np.newaxis].astype("float32")]

    return representative_dataset


def export_tflite_int8(saved_model_dir: str, tflite_path: str, input_size: int, calib_npy: str | None):
    print(f"[4/4] Quantizing -> {tflite_path}")
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(input_size, calib_npy)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"    Saved {tflite_path}")
    print(f"    Size: {size_kb:.1f} kB")


def parse_args():
    p = argparse.ArgumentParser(description="Export TinyFireCNN to INT8 TFLite")
    p.add_argument("--weights", default=os.path.join(_HERE, "tinyfirecnn_best.pth"),
                   help="Path to trained .pth/.pt weights")
    p.add_argument("--out-dir", default=_HERE,
                   help="Directory for intermediate and final files")
    p.add_argument("--input-size", type=int, default=128,
                   help="Input image size used by the model")
    p.add_argument("--calib-npy", default=None,
                   help="Optional .npy file for representative dataset")
    p.add_argument("--skip-onnxslim", action="store_true",
                   help="Skip ONNX graph slimming")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    onnx_path = os.path.join(args.out_dir, "tinyfirecnn.onnx")
    slim_path = os.path.join(args.out_dir, "tinyfirecnn_slim.onnx")
    saved_model_dir = os.path.join(args.out_dir, "tinyfirecnn_saved_model")
    tflite_path = os.path.join(args.out_dir, "tinyfirecnn_int8.tflite")

    if not os.path.exists(args.weights):
        sys.exit(f"[ERROR] Weights not found: {args.weights}")

    export_onnx(args.weights, onnx_path, args.input_size)

    active_onnx = onnx_path
    if not args.skip_onnxslim:
        active_onnx = slim_onnx(onnx_path, slim_path)

    export_saved_model(active_onnx, saved_model_dir)
    export_tflite_int8(saved_model_dir, tflite_path, args.input_size, args.calib_npy)

    print(f"\nDone. INT8 TFLite model: {tflite_path}")
    print("Next: convert the .tflite file into a C array/header for Arduino.")
    

if __name__ == "__main__":
    main()