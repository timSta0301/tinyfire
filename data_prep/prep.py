import os
from ultralytics import YOLO
import random
import shutil

def count_classes(label_dir):
    counts = {}
    for file in os.listdir(label_dir):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1
    return counts

src_img = "dataset/train/images"
src_lbl = "dataset/train/labels"

dst_img = "dataset/valid/images"
dst_lbl = "dataset/valid/labels"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_lbl, exist_ok=True)

files = [f for f in os.listdir(src_img) if f.endswith(".jpg") or f.endswith(".png")]

random.shuffle(files)

split_idx = int(0.8 * len(files))
valid_files = files[split_idx:]

for f in valid_files:
    # move image
    shutil.move(os.path.join(src_img, f), os.path.join(dst_img, f))
    
    # move label
    label_file = f.rsplit(".", 1)[0] + ".txt"
    shutil.move(os.path.join(src_lbl, label_file), os.path.join(dst_lbl, label_file))

print("Split done.")

print("Train:", count_classes("dataset/train/labels"))
print("Valid:", count_classes("dataset/valid/labels"))
