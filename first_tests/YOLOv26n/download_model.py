import os
from ultralytics import YOLO

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov26n.pt")

print("Downloading yolov26n base model via ultralytics...")
model = YOLO("yolov26n.pt")
model.save(dest)
print(f"Model saved to: {dest}")
