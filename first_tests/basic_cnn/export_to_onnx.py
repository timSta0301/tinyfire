"""
Export TinyFireCNN (basic cnn) .pth/.pt weights to ONNX.

Usage:
    python first_tests/basic_cnn/export_to_onnx.py
    python first_tests/basic_cnn/export_to_onnx.py --weights first_tests/basic_cnn/tinyfirecnn_best.pth
"""

import argparse
import os
import sys
import torch

from classifier import TinyFireCNN


def export_onnx(weights_path: str, output_path: str, input_size: int = 128):
    print(f"Loading weights from: {weights_path}")

    model = TinyFireCNN()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Exporting ONNX to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )

    print(f"Done. Saved ONNX model to: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.1f} kB")


def parse_args():
    parser = argparse.ArgumentParser(description="Export TinyFireCNN to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="first_tests/basic_cnn/tinyfirecnn_best.pth",
        help="Path to trained .pth/.pt weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="first_tests/basic_cnn/tinyfirecnn.onnx",
        help="Path to output ONNX file",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="Model input image size",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"ERROR: weights file not found: {args.weights}")

    export_onnx(
        weights_path=args.weights,
        output_path=args.output,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()