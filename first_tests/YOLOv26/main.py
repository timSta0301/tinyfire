import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from visualize import visualize


def prune(model, amount):
    for m in model.model.modules():
        if isinstance(m, nn.Conv2d):
            torch_prune.l1_unstructured(m, name="weight", amount=amount)
            torch_prune.remove(m, "weight")


_dir = os.path.dirname(os.path.abspath(__file__))
_data_dir = os.path.join(_dir, "..", "..", "data")
model_original = YOLO(os.path.join(_dir, "best.pt"))
model_pruned = copy.deepcopy(model_original)
prune(model_pruned, 0.3)

results_original = model_original.predict(os.path.join(_data_dir, "image5.png"), conf=0.25)
for result in results_original:
    visualize(result)
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected in Original Model: {model_original.names[cls]} ({conf:.2f})")

results_pruned = model_pruned.predict(os.path.join(_data_dir, "image5.png"), conf=0.25)
for result in results_pruned:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected in Pruned Model: {model_pruned.names[cls]} ({conf:.2f})")

# Export Original Model if not already done yet
if not os.path.exists(os.path.join(_dir, "exported_model_original")):
    print("Skipping export of original model - Folder already exists")
    model_original.export(format="tflite", optimize=True, int8=True)
    os.rename(os.path.join(_dir, "best_saved_model"), os.path.join(_dir, "exported_model_original"))

# Export experimental pruned model
model_pruned.export(format="tflite", optimize=True, int8=True)
os.rename(os.path.join(_dir, "best_saved_model"), os.path.join(_dir, "exported_model_pruned"))

# Load the exported TFLite model
model_tflite = YOLO(os.path.join(_dir, "exported_model_pruned", "best_int8.tflite"))

results_tflite = model_tflite.predict(os.path.join(_data_dir, "image5.png"), conf=0.25)
for result in results_tflite:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected in tflite Model: {model_pruned.names[cls]} ({conf:.2f})")
