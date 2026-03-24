import os
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visualize import visualize

_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(_dir, "yolov26n.pt"))

results = model.predict(os.path.join(_dir, "../../data/image7.png"), conf=0.25)

for result in results:
    visualize(result)
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: {model.names[cls]} ({conf:.2f})")
