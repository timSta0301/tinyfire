import os
from ultralytics import YOLO

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26n-cls.pt")

print("Downloading yolo26n-cls base model via ultralytics...")
model = YOLO("yolo26n-cls.pt")
model.save(dest)
print(f"Model saved to: {dest}")
