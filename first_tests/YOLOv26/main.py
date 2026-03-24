import os
import shutil
from ultralytics import YOLO
import torch
import utils
import copy
from utils.torch_utils import prune

# Load model
_dir = os.path.dirname(os.path.abspath(__file__))
model_original = YOLO(os.path.join(_dir, "best.pt"))
model_pruned = copy.deepcopy(model_original)
prune(model_pruned, 0.3)



# Run inference
results_original = model_original.predict(os.path.join(_dir, "image5.png"), conf=0.25)
# Process results
for result in results_original:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_original.names[cls]
        print(f"Detected in Original Model: {label} ({conf:.2f})")



results_pruned = model_pruned.predict(os.path.join(_dir, "image5.png"), conf=0.25)
# Process results
for result in results_pruned:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_pruned.names[cls]
        print(f"Detected in Pruned Model: {label} ({conf:.2f})")


# Export Original Model if not already done yet 
if not os.path.exists(os.path.join(_dir,"exported_model_original")):
    print("Skipping export of original model - Folder already exists")
    model_original.export(format="tflite", optimize = True, int8 = True)
    # Output path cannot be specified (bruh), therefore we have to rename it 
    os.rename(os.path.join(_dir,"best_saved_model"),os.path.join(_dir,"exported_model_original"))

# Export experimental pruned model
model_pruned.export(format="tflite", optimize = True, int8 = True)
os.rename(os.path.join(_dir,"best_saved_model"),os.path.join(_dir,"exported_model_pruned"))

# Load the exported TFLite model
model_tflite = YOLO(os.path.join(_dir,"exported_model_pruned","best_int8.tflite"))


# Run inference on tflite model
results_tflite = model_tflite.predict(os.path.join(_dir, "image5.png"), conf=0.25)
# Process results
for result in results_tflite:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model_pruned.names[cls]
        print(f"Detected in tflite Model: {label} ({conf:.2f})")
