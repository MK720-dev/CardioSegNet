"""
viewer/utils.py

Tools for viewing full volumes (10 slices) from HDF5 files.
"""

import numpy as np
import h5py
import cv2
from pathlib import Path
from tensorflow import keras

from config import IMG_SIZE, MODEL_DIR


def load_volume(path: Path):
    """
    Load a 3D HDF5 cardiac MRI volume:
    image → (Slices, H, W)
    label → same

    Return only the image volume for visualization.
    """
    with h5py.File(path, "r") as f:
        vol = f["image"][:]      # shape (10, 256, 216)
    return vol


def preprocess_slice_for_model(slice_2d):
    """
    Resize + normalize a slice to feed the U-Net.
    """
    slice_resized = cv2.resize(slice_2d, (IMG_SIZE, IMG_SIZE))
    slice_resized = slice_resized.astype(np.float32)

    max_val = slice_resized.max()
    if max_val > 0:
        slice_resized /= max_val

    slice_resized = np.expand_dims(slice_resized, -1)
    slice_resized = np.expand_dims(slice_resized, 0)
    return slice_resized


def load_trained_model():
    model_path = MODEL_DIR / "unet_lv_baseline_slice128_full.keras"
    model = keras.models.load_model(model_path, compile=False)
    return model


def predict_mask(model, slice_2d):
    orig_shape = slice_2d.shape

    x = preprocess_slice_for_model(slice_2d)
    pred = model.predict(x, verbose=0)[0, ..., 0]

    pred_resized = cv2.resize(pred, (orig_shape[1], orig_shape[0]))
    return (pred_resized > 0.5).astype(np.uint8)


def create_overlay(slice_2d, pred_mask, alpha=0.3):
    img = slice_2d.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = (img * 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    mask_rgb = np.zeros_like(img_rgb)
    mask_rgb[..., 0] = pred_mask * 255

    return cv2.addWeighted(img_rgb, 1.0, mask_rgb, alpha, 0)

