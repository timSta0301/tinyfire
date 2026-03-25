from ultralytics import YOLO
import os

# Load trained model
model = YOLO(os.path.join(os.path.dirname(__file__), "wildfire-smoke-fire.pt"))

# Run inference on CPU
results = model(os.path.join(os.path.dirname(__file__), "../data/image1.jpg"), device="cpu")

# Ultralytics returns a list (even for a single image)
r = results[0]

# Print detected boxes
print("Detected boxes:")
print(r.boxes)

# Print class names
print("Class names:")
print(r.names)
