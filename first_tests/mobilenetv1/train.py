"""
Fine-tune MobileNetV1 Alpha 0.25 on fire/smoke/other data and export INT8 TFLite.

Expected data layout:
    data_split/
      train/fire/   train/smoke/   train/other/
      val/fire/     val/smoke/     val/other/   (optional — auto-split used if val/ is empty)

Usage:
    python train.py
    python train.py --data ../../data_split --epochs 20 --imgsz 128
    python train.py --data ../../data_split --val-split 0.15
"""

import os
import argparse
import glob
import re
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available: {[g.name for g in gpus]}")
    else:
        print("No GPU found — training on CPU.")


def build_model(num_classes: int, imgsz: int, alpha: float = 0.25) -> tuple[Model, Model]:
    """Returns (full_model, base_model) for two-phase training."""
    base = MobileNet(
        alpha=alpha,
        input_shape=(imgsz, imgsz, 3),
        include_top=False,
        weights="imagenet",
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=x), base


def make_dataset(directory: str, imgsz: int, batch_size: int, augment: bool,
                  val_split: float = 0.0, subset: str = None, seed: int = 42):
    kwargs = dict(
        image_size=(imgsz, imgsz),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=augment or (subset is not None),
        seed=seed if subset else None,
    )
    if subset:
        kwargs["validation_split"] = val_split
        kwargs["subset"] = subset

    ds = tf.keras.utils.image_dataset_from_directory(directory, **kwargs)
    class_names = ds.class_names

    if augment:
        # Apply augmentation in image space [0,255] before MobileNet preprocessing.
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.06),
            layers.RandomBrightness(0.08, value_range=(0.0, 255.0)),
            layers.RandomContrast(0.08),
        ])
        ds = ds.map(
            lambda x, y: (aug(tf.cast(x, tf.float32), training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Preprocess: [0,255] → [-1,1]
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE), class_names


def compute_class_weights(train_dir: str, class_names: list[str]) -> dict:
    counts = {i: len(os.listdir(os.path.join(train_dir, cls)))
              for i, cls in enumerate(class_names)
              if os.path.isdir(os.path.join(train_dir, cls))}
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {i: total / (n_classes * c) for i, c in counts.items()}
    print(f"Class weights: { {class_names[i]: f'{w:.2f}' for i, w in weights.items()} }")
    return weights


def representative_dataset(calib_dir: str, imgsz: int):
    paths = glob.glob(os.path.join(calib_dir, "**", "*.*"), recursive=True)
    paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))][:200]

    def gen():
        for p in paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            img = preprocess_input(img.astype(np.float32))
            yield [img[np.newaxis]]

    return gen


def export_tflite(model: Model, calib_dir: str, imgsz: int, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(calib_dir, imgsz)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite INT8: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


def get_next_versioned_path(models_dir: str, base_name: str, extension: str) -> str:
    """Returns models_dir/base_name_<n><extension> with the next available integer n."""
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+){re.escape(extension)}$")
    next_version = 1

    for name in os.listdir(models_dir):
        match = pattern.match(name)
        if match:
            next_version = max(next_version, int(match.group(1)) + 1)

    return os.path.join(models_dir, f"{base_name}_{next_version}{extension}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(_DIR, "..", "..", "data_split"))
    p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--unfreeze-epochs", type=int, default=15,
                   help="Extra epochs after unfreezing the base")
    p.add_argument("--unfreeze-ratio", type=float, default=0.35,
                   help="Fraction of top base-model layers to unfreeze in phase 2")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of train data used for validation when val/ is empty")
    return p.parse_args()


def main():
    setup_gpu()
    args = parse_args()
    data = os.path.abspath(args.data)
    train_dir = os.path.join(data, "train")
    val_dir = os.path.join(data, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Directory not found: {train_dir}\n"
            "Populate data_split/train/<class>/ with images."
        )

    # Auto-split from train if val/ is missing or empty
    val_has_images = os.path.isdir(val_dir) and any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for _, _, files in os.walk(val_dir)
        for f in files
    )
    if val_has_images:
        print(f"Using existing val split: {val_dir}")
        train_ds, class_names = make_dataset(train_dir, args.imgsz, args.batch, augment=True)
        val_ds, _ = make_dataset(val_dir, args.imgsz, args.batch, augment=False)
    else:
        print(f"val/ is empty — auto-splitting train with val_split={args.val_split}")
        train_ds, class_names = make_dataset(train_dir, args.imgsz, args.batch, augment=True,
                                             val_split=args.val_split, subset="training")
        val_ds, _ = make_dataset(train_dir, args.imgsz, args.batch, augment=False,
                                 val_split=args.val_split, subset="validation")

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    class_weights = compute_class_weights(train_dir, class_names)

    # Save class names
    names_path = os.path.join(_DIR, "names.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(class_names))

    models_dir = os.path.join(_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    model, base = build_model(num_classes, args.imgsz, args.alpha)
    best_path = get_next_versioned_path(models_dir, "mobilenetv1_fire_best", ".keras")
    tflite_path = get_next_versioned_path(models_dir, "mobilenetv1_fire_int8", ".tflite")
    print(f"Saving Keras checkpoint to: {best_path}")
    print(f"Saving INT8 TFLite export to: {tflite_path}")

    callbacks_base = [
        EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_path, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    # --- Phase 1: train head only ---
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\nPhase 1: head only ({args.epochs} epochs)")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks_base,
    )

    # --- Phase 2: fine-tune top layers only (keep BatchNorm frozen) ---
    base.trainable = True
    total_layers = len(base.layers)
    n_unfreeze = max(1, int(total_layers * args.unfreeze_ratio))
    split_idx = total_layers - n_unfreeze
    for i, layer in enumerate(base.layers):
        trainable = i >= split_idx
        if isinstance(layer, layers.BatchNormalization):
            trainable = False
        layer.trainable = trainable

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"Phase 2 unfreezing {trainable_count}/{total_layers} base layers")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr / 5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
    )
    print(f"\nPhase 2: full fine-tune ({args.unfreeze_epochs} epochs)")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.unfreeze_epochs,
        class_weight=class_weights,
        callbacks=callbacks_base,
    )

    # Export INT8 TFLite using training data for calibration
    export_tflite(model, train_dir, args.imgsz, tflite_path)


if __name__ == "__main__":
    main()
