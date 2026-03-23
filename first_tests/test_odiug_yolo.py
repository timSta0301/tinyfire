from ultralytics import YOLO

# Load trained model
model = YOLO("wildfire-smoke-fire.pt")

# Run inference on CPU
results = model("image.jpg", device="cpu")

# Ultralytics returns a list (even for a single image)
r = results[0]

# Print detected boxes
print("Detected boxes:")
print(r.boxes)

# Print class names
print("Class names:")
print(r.names)
