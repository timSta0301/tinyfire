import os
import cv2
from ultralytics import YOLO

_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(_dir, "yolo26n-cls.pt"))

results = model.predict(os.path.join(_dir, "..", "..", "data", "image7.png"), conf=0.25)

for result in results:
    probs = result.probs
    names = result.names

    img = result.orig_img.copy()
    for i, (cls, conf) in enumerate(zip(probs.top5, probs.top5conf.tolist())):
        label = f"{names[cls]}: {conf:.2f}"
        cv2.putText(img, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        print(label)

    cv2.imshow("Classification", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
