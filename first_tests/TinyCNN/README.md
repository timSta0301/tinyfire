# TinyCNN — Fire/Smoke Classifier for Arduino Nano 33 BLE

A micro-scale image classifier that detects **fire**, **smoke**, or **neither**,
small enough to run on a microcontroller.

| | Baseline (YOLOv26, `best.pt`) | TinyCNN INT8 |
|---|---|---|
| Task | Detection (bounding boxes) | Classification |
| Weights format | FP32 PyTorch `.pt` (teacher, training only) | INT8 TFLite (deployed on MCU) |
| Model size | 9.6 MB (INT8 TFLite export — too big for MCU) | ~178 kB |
| Fits on MCU (<500 kB flash) | ✗ | ✓ |
| Fits in MCU RAM | ✗ (307 kB arena @ 96×96) | ✓ (100 kB arena @ 48×48) |
| Classes | 3 (fire / other / smoke) | 3 (fire / other / smoke) |
| Role | Teacher — soft labels via knowledge distillation | Student — runs on Arduino Nano 33 BLE |

---

## Prerequisites

**Python environment** — from the `tinyfire/` root:
```bash
pip install -r requirements.txt
# tensorflow and tf_keras must be installed with --no-deps (see note in requirements.txt):
pip install tensorflow==2.20.0 --no-deps
pip install tf_keras==2.19.0 --no-deps
```

**Dataset** — download the Roboflow dataset and place it at:
```
datasets/Fire Detection.v1i.yolo26/
```
The dataset is available at:
https://universe.roboflow.com/personal-bodxv/fire-detection-sejra-fognw/dataset/1

Export format: **YOLOv8** (YOLO detection `.txt` labels).

---

## Step-by-step pipeline

All commands are run from the `tinyfire/` root directory.

### 1 — Prepare dataset

Converts the YOLO detection annotations into an image classification layout
and splits into train (85%) / val (15%):

```bash
python3 first_tests/TinyCNN/prepare_dataset.py
```

Output: `datasets/TinyCNN_data/train/{fire,other,smoke}/` and `val/`.

---

### 2 — Train with knowledge distillation

Trains TinyCNN using the fine-tuned YOLOv26 detection model (`best.pt`) as
a teacher to produce soft probability targets:

```bash
python3 first_tests/TinyCNN/distill.py \
  --data "datasets/TinyCNN_data" \
  --epochs 50
```

Output: `first_tests/TinyCNN/tinycnn.pth`

Optional flags:
| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--alpha` | 0.7 | KD loss weight (0 = pure CE, 1 = pure KD) |
| `--temperature` | 4.0 | Distillation temperature |
| `--no-distill` | off | Disable KD, use plain cross-entropy |
| `--resume` | — | Path to checkpoint `.pth` to continue from |

---

### 3 — Export to INT8 TFLite

Converts the trained weights to a quantised TFLite model:

```bash
python3 first_tests/TinyCNN/export.py
```

Output: `first_tests/TinyCNN/tiny_cnn_int8.tflite` (~178 kB)

---

### 4 — Evaluate accuracy

Quick test on a single image:
```bash
python3 first_tests/TinyCNN/evaluate.py --image data/image5.png
```

Full accuracy benchmark over the validation set:
```bash
python3 first_tests/TinyCNN/evaluate.py --data datasets/TinyCNN_data/val
```

---

### 5 — Generate Arduino C header

```bash
python3 first_tests/TinyCNN/to_c_array.py
```

Output: `first_tests/TinyCNN/model_data.h`

Include it in your Arduino sketch alongside the
[TFLite Micro library](https://github.com/tensorflow/tflite-micro-arduino-examples):

```cpp
#include "model_data.h"

const tflite::Model* model = tflite::GetModel(g_model_data);
// Input tensor: INT8 [1, 48, 48, 3]
// Output tensor: INT8 [1, 3]  →  index 0=fire, 1=other, 2=smoke
```

---

## Architecture

```
Input 48×48×3
    │
    ▼
Conv(3→32, stride=2)         → 24×24×32
DWSep(32→64,  stride=1)      → 24×24×64
DWSep(64→128, stride=2)      → 12×12×128
DWSep(128→128,stride=1)      → 12×12×128
DWSep(128→256,stride=2)      →  6×6×256
DWSep(256→256,stride=1)      →  6×6×256
AvgPool(6) → Linear(256→3)
    │
    ▼
[fire, other, smoke]
```

DWSep = depthwise-separable convolution (MobileNet-style).

---

## File overview

| File | Purpose |
|---|---|
| `model.py` | TinyCNN architecture definition |
| `prepare_dataset.py` | YOLO detection dataset → ImageFolder classification layout |
| `distill.py` | Training loop with knowledge distillation from `best.pt` |
| `export.py` | `.pth` → ONNX → TF SavedModel → INT8 TFLite |
| `evaluate.py` | Accuracy benchmark and per-image inference |
| `to_c_array.py` | `.tflite` → Arduino C header (`model_data.h`) |
