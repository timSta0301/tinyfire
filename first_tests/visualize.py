import cv2
import numpy as np


def visualize(result, window_title="YOLO Result"):
    img = result.orig_img.copy()
    names = result.names

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if result.masks is not None:
        for mask in result.masks.data:
            m = mask.cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            img[m == 1] = img[m == 1] * 0.5 + color * 0.5

    cv2.imshow(window_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
