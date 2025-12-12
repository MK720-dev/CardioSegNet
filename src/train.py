"""
src/train.py

Training script for CardioSegNet.
Supports:
- Phase 1: Binary LV segmentation (baseline U-Net)
- Phase 2: Multi-class segmentation (deep U-Net)
"""

import sys
from pathlib import Path

# Ensure project root is in path
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras

from config import (
    MODEL_DIR,
    LR,
    EPOCHS,
    SEG_MODE,
    IMG_SIZE,
)

from .data_loader import build_datasets
from .model_unet import build_unet_baseline, build_unet_deep
from .losses import (
    bce_dice_loss,
    dice_coef,
    sparse_cce_multi_class_dice_loss,
    multi_class_dice,
)


def main():
    print("[INFO] Building datasets...")
    train_ds, val_ds = build_datasets()

    print(f"[INFO] Segmentation mode: {SEG_MODE}")

    # -------------------------
    # MODEL SELECTION
    # -------------------------
    if SEG_MODE == "binary":
        print("[INFO] Building baseline U-Net (binary LV)...")
        model = build_unet_baseline(input_shape=(IMG_SIZE, IMG_SIZE, 1))

        loss_fn = bce_dice_loss
        metrics = [dice_coef, "accuracy"]

        model_name = "unet_lv_baseline_slice128"

    elif SEG_MODE == "multi-class":
        print("[INFO] Building deep U-Net (multi-class)...")
        model = build_unet_deep(input_shape=(IMG_SIZE, IMG_SIZE, 1))

        loss_fn = sparse_cce_multi_class_dice_loss
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="pixel_acc"),
            multi_class_dice,
        ]

        model_name = "unet_deep_256"

    else:
        raise ValueError(f"Unknown SEG_MODE: {SEG_MODE}")

    # -------------------------
    # COMPILE
    # -------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=loss_fn,
        metrics=metrics,
    )

    model.summary()

    # -------------------------
    # CALLBACKS
    # -------------------------
    best_model = MODEL_DIR / f"{model_name}.weights.h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_model),
            monitor="val_multi_class_dice" if SEG_MODE == "multiclass" else "val_dice_coef",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_multi_class_dice" if SEG_MODE == "multiclass" else "val_dice_coef",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # -------------------------
    # TRAIN
    # -------------------------
    print("[INFO] Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # -------------------------
    # SAVE FULL MODEL
    # -------------------------
    full_path = MODEL_DIR / f"{model_name}_full.keras"
    model.save(full_path)

    print(f"[INFO] Training complete.")
    print(f"[INFO] Saved full model to {full_path}")


if __name__ == "__main__":
    main()

