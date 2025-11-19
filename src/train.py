
"""
src/train.py

Training script for the baseline 2D U-Net on ACDC 2D slices.
"""

import sys
from pathlib import Path

# Ensure project root is in path
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras

from config import MODEL_DIR, LR, EPOCHS
from src.data_loader import build_datasets
from src.model_unet import build_unet_baseline
from src.losses import bce_dice_loss, dice_coef


def main():
    print("[INFO] Building datasets...")
    train_ds, val_ds = build_datasets()

    print("[INFO] Building model...")
    model = build_unet_baseline(input_shape=(128, 128, 1))

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=bce_dice_loss,
        metrics=[dice_coef, 'accuracy'],
    )

    model.summary()

    best_model = MODEL_DIR / "unet_lv_baseline_slice128.h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_model),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("[INFO] Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("[INFO] Training complete.")
    full_path = MODEL_DIR / "unet_lv_baseline_slice128_full.keras"
    model.save(full_path)
    print(f"[INFO] Saved full model to {full_path}")


if __name__ == "__main__":
    main()
