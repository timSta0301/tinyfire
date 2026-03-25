# TinyCNN Deployment Notes

## Status (2026-03-24)

Training works, export pipeline works, TFLite file is valid.
Model currently trained for **1 epoch only** — needs 50 epochs for real accuracy.

---

## Invoke() crash on MCU — root cause

`AllocateTensors()` succeeds and reports only ~14 kB used.
`Invoke()` crashes immediately.

**The `arena_used_bytes()` value is misleading.** It only counts TFLM's internal
bookkeeping, not the peak activation memory needed during inference.

The actual peak comes from two tensors that must coexist at block 1→2 boundary:

| Tensor | Shape | Size |
|---|---|---|
| Block 1 output | [1, 48, 48, 64] | 147,456 B |
| PAD input for block 2 | [1, 50, 50, 64] | 160,000 B |
| **Both live simultaneously** | | **≈ 307 KB** |

The arena was set to 50 KB. TFLM silently overruns it and corrupts adjacent
BSS memory, which only surfaces as a hard fault during `Invoke()`.

---

## Fix for dev kit (448 KB RAM)

In `main.c`, change:
```c
// Before
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// After
constexpr int kTensorArenaSize = 350 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // alignas(16) required by CMSIS-NN
```

`alignas(16)` is required for CMSIS-NN SIMD kernels — missing alignment is a
second common cause of Invoke() failures on Cortex-M even when memory is fine.

---

## Fix for Arduino Nano 33 BLE final target (256 KB RAM)

The nRF52840 has only 256 KB total RAM. With ~16 KB for stack/BSS, only ~240 KB
is available for the arena — 67 KB short of the required 307 KB.

**Solution: reduce `INPUT_SIZE` from 96 → 48.**

With 48×48 input the peak activation drops to ~80 KB (fits in a 100 KB arena):

| Tensor (48×48 model) | Shape | Size |
|---|---|---|
| Block 1 output | [1, 24, 24, 64] | 36,864 B |
| PAD input for block 2 | [1, 26, 26, 64] | 43,264 B |
| Peak | | **≈ 80 KB** |

### Changes needed in model.py

```python
INPUT_SIZE = 48

# Pool kernel: 48 → stride-2 ×3 → 6×6 before pool
self.pool = nn.AvgPool2d(kernel_size=6)
```

### Changes needed in main.c

```c
constexpr int kTensorArenaSize = 100 * 1024;  // 100 KB is enough for 48×48
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
```

### Retraining required

The weight shapes (filter sizes) are unchanged — only the spatial resolution of
activations differs. However, the model was trained on 96×96 images so accuracy
will be slightly lower. Retraining on 48×48 crops is recommended:

```bash
python3 first_tests/TinyCNN/distill.py \
  --data "/home/sebastian/FAU/7. Semester/Green AI Hackathon/datasets/TinyCNN_data" \
  --epochs 50
```

Then re-export:
```bash
python3 first_tests/TinyCNN/export.py
```

---

## TRANSPOSE op in the TFLite model

The model contains a `TRANSPOSE` op (between `AVERAGE_POOL_2D` and `RESHAPE`)
inserted by `onnx2tf` during its NCHW → NHWC layout conversion.

- All operators (including TRANSPOSE) are registered in the op resolver → not the crash cause
- TRANSPOSE is present in both old and new model exports
- It does not cause a crash on its own but adds one unnecessary op

To remove it cleanly, the final `nn.Linear` layer should be replaced with
`nn.Conv2d(256, num_classes, kernel_size=1)`, which maps directly to
`CONV_2D` in TFLite without needing a preceding TRANSPOSE + RESHAPE.
This requires a small weight migration (reshape linear weights to 4D) and retraining.

---

## Full operator list (current model, 96×96 input)

```
 0: PAD
 1: CONV_2D
 2: DEPTHWISE_CONV_2D
 3: CONV_2D
 4: PAD
 5: DEPTHWISE_CONV_2D
 6: CONV_2D
 7: DEPTHWISE_CONV_2D
 8: CONV_2D
 9: PAD
10: DEPTHWISE_CONV_2D
11: CONV_2D
12: DEPTHWISE_CONV_2D
13: CONV_2D
14: AVERAGE_POOL_2D
15: TRANSPOSE          ← unnecessary, see note above
16: RESHAPE
17: FULLY_CONNECTED
```

---

## Model size summary

| Model | Size | MCU fit (<500 kB flash) |
|---|---|---|
| YOLOv26 FP32 `best.pt` (distillation teacher, training only) | 9.6 MB (INT8 TFLite export) | ✗ |
| TinyCNN INT8 96×96 | 177.7 kB | ✓ flash, ✗ RAM (307 kB arena) |
| TinyCNN INT8 48×48 | ~177 kB | ✓ flash, ✓ RAM (100 kB arena) |

---

## TODO

- [x] Change `INPUT_SIZE` to 48 in `model.py`, update `AvgPool2d(6)`
- [ ] Retrain for 50 epochs on 48×48 images
- [ ] Re-export and verify `tiny_cnn_int8.tflite` still < 500 kB
- [ ] Update `kTensorArenaSize = 100 * 1024` and add `alignas(16)` in `main.c`
- [ ] Flash and verify `Invoke()` succeeds
- [ ] Optional: replace `nn.Linear` with `nn.Conv2d(256, 3, 1)` to eliminate TRANSPOSE op
