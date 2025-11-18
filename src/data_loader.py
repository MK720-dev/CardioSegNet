"""
src/data_loader.py

HDF5 loader for the preprocessed ACDC dataset.

We use the 2D slice files from ACDC_training_slices/:

Each .h5 file contains:
    image   → (H, W) float32
    label   → (H, W) uint8
    scribble → (H, W) uint16   (ignored)

This loader:
- finds all slice files
- loads image + label
- binarizes LV = 3
- resizes to (IMG_SIZE, IMG_SIZE)
- normalizes image
- returns tf.data Dataset ready for training
"""

import os
import cv2
import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List
from ..config import  IMG_SIZE, BATCH_SIZE, SLICES_DIR, RANDOM_SEED, VAL_SPLIT


def load_h5_slice(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single 2D slice HDF5 file.
    Returns the MRI image and ground truth label.
    """
    with h5py.File(path, "r") as f:
        img = f["image"][:]       # (256, 216)
        mask = f["label"][:]      # (256, 216), uint8 labels (0–3)
    return img, mask


def preprocess(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize, normalize, and binarize the slice.

    - Resize to IMG_SIZE x IMG_SIZE
    - Normalize img to [0, 1]
    - Binarize LV mask: label==3
    - Add channel dimension
    """

    img = img.astype(np.float32)
    mask = mask.astype(np.uint8)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Normalize
    max_val = img.max()
    if max_val > 0:
        img /= max_val

    # Binarize LV
    mask = (mask == 3).astype(np.float32)

    # Add channel dims
    img = np.expand_dims(img, -1)
    mask = np.expand_dims(mask, -1)

    return img, mask


def _load_py(path: bytes):
    """
    2D slice loading function wrapped by tf.py_function.
    """
    path = path.decode("utf-8")
    img, mask = load_h5_slice(Path(path))
    img, mask = preprocess(img, mask)
    return img, mask


def tf_load(path):
    """
    Convert HDF5 read into TF dataset elements.
    """
    img, mask = tf.py_function(
        func=_load_py,
        inp=[path],
        Tout=[tf.float32, tf.float32],
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 1])
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img, mask


def build_datasets():
    """
    Build train & val datasets from .h5 slice files.
    """
    slice_files = sorted([str(p) for p in Path(SLICES_DIR).glob("*.h5")])
    slice_files = np.array(slice_files)

    # shuffle + split
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(slice_files))
    rng.shuffle(indices)

    split = int(len(indices) * (1 - VAL_SPLIT))
    train_idx, val_idx = indices[:split], indices[split:]

    train_files = slice_files[train_idx]
    val_files = slice_files[val_idx]

    # TF datasets
    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_files)
        .shuffle(len(train_files))
        .map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_files)
        .map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds

build_datasets()