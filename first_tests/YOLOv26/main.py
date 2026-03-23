from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Run inference
results = model.predict("image.png", conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"Detected: {label} ({conf:.2f})")
