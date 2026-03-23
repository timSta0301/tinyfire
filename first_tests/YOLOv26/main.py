import os
from ultralytics import YOLO

# Load model
_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(_dir, "best.pt"))

# Run inference
results = model.predict(os.path.join(_dir, "image5.png"), conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"Detected: {label} ({conf:.2f})")
