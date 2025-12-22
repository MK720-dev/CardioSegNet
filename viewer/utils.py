# viewer/utils.py

"""
Utilities for CardioSegNet viewer.

- Load 3D HDF5 volumes (image + label)
- Run the trained 2D U-Net slice-wise
- Build RGB overlays for GT and predictions
- Provide hooks for adding more models later (Phase 2)
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import h5py
import cv2
from tensorflow import keras

from config import IMG_SIZE, MODEL_DIR


# --------- Volume I/O --------- #

def load_volume_and_label(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D ACDC volume from an HDF5 file.

    File structure:
        image  -> (S, H, W) float32
        label  -> (S, H, W) uint8 (0..3, where 3 = LV)
        scribble -> ignored

    Returns
    -------
    vol_img : np.ndarray, shape (S, H, W)
    vol_lbl : np.ndarray, shape (S, H, W)
    """
    with h5py.File(path, "r") as f:
        vol_img = f["image"][:]   # (S, 256, 216)
        vol_lbl = f["label"][:]   # (S, 256, 216)
    return vol_img, vol_lbl


# --------- Model loading & inference --------- #

def load_models() -> Dict[str, keras.Model]:
    """
    Load one or more trained models for visualization.

    Keys in the returned dict are model IDs used in the UI.
    For Phase 1 we only load the baseline model; Phase 2 can
    easily add a second, e.g. 'model_b'.

    Returns
    -------
    models : dict[str, keras.Model]
    """
    models: Dict[str, keras.Model] = {}

    # Baseline model (Model A)
    model_a_path = MODEL_DIR / "unet_lv_baseline_slice128_full.keras"
    if model_a_path.exists():
        models["baseline_unet"] = keras.models.load_model(
            model_a_path,
            compile=False
        )

    # Example for Phase 2:
    model_b_path = MODEL_DIR / "unet_deep_256_full.keras"
    if model_b_path.exists():
        models["advanced_unet"] = keras.models.load_model(
        model_b_path,
        compile=False
        )

    return models


def preprocess_slice_for_model(slice_2d: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize + normalize a single 2D slice to feed the 2D U-Net.

    Parameters
    ----------
    slice_2d : (H, W) float32 or float64

    Returns
    -------
    x : (1, IMG_SIZE, IMG_SIZE, 1) float32
    """
    # Resize to training resolution
    resized = cv2.resize(slice_2d, (target_size, target_size))
    resized = resized.astype(np.float32)

    # Normalize to [0, 1]
    max_val = resized.max()
    if max_val > 0:
        resized /= max_val

    # Add channel + batch dimensions
    x = np.expand_dims(resized, axis=-1)   # (H, W, 1)
    x = np.expand_dims(x, axis=0)         # (1, H, W, 1)
    return x


def predict_mask(model_name: str, model: keras.Model,
                 slice_2d: np.ndarray,
                 orig_shape: Tuple[int, int]) -> np.ndarray:
    """
    Run model prediction on a single slice and upsample back to original size.

    Parameters
    ----------
    model : keras.Model
    slice_2d : (H, W) float32
    orig_shape : (H_orig, W_orig)

    Returns
    -------
    pred_bin : (H_orig, W_orig) uint8, {0,1}
    """
    
    if model_name == "baseline_unet":
        target_size = 128
        x = preprocess_slice_for_model(slice_2d, target_size)
        pred = model.predict(x, verbose=0)[0, ..., 0]   # (IMG_SIZE, IMG_SIZE)
        # Resize prediction back to original slice resolution
        pred_resized = cv2.resize(
            pred,
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        label_map = np.zeros_like(pred_resized, dtype=np.uint8)
        label_map[pred_resized > 0.5] = 3   # map LV → class 3

    elif model_name == "advanced_unet":
        target_size = 256
        x = preprocess_slice_for_model(slice_2d, target_size)
        pred = model.predict(x, verbose=0)[0] # Full prediction tensor (H, W, 4)
        label_map = np.argmax(pred, axis=-1)   # (H, W)
        label_map = cv2.resize(
            label_map,
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    return label_map



# --------- Overlay creation --------- #
def make_overlay(
    slice_2d: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_masks: Dict[str, Optional[np.ndarray]],
    modes: List[str],
    classes: List[int],
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Build an RGB overlay with multi-class ground truth and predictions.

    Parameters
    ----------
    slice_2d : (H, W) float32
        Raw MRI slice.
    gt_mask : (H, W) uint8 or None
        Ground truth class map {0,1,2,3}.
    pred_masks : dict[str, ndarray or None]
        Model predictions as class maps.
    modes : list[str]
        Active overlays (e.g. ["gt", "baseline_unet", "advanced_unet"])
    classes: List[int]
        Which classes need to be visible
    alpha : float
        Overlay transparency.

    Returns
    -------
    overlay_rgb : (H, W, 3) uint8
    """

    # -------------------------
    # Normalize MRI background
    # -------------------------
    img = slice_2d.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    base_rgb = np.stack([img, img, img], axis=-1)
    base_rgb = (base_rgb * 255).astype(np.uint8)

    overlay = base_rgb.copy()

    # -------------------------
    # Class color definitions
    # -------------------------
    CLASS_COLORS = {
        1: np.array([255,   0,   0]),   # RV
        2: np.array([255, 255,   0]),   # MYO
        3: np.array([  0,   0, 255]),   # LV
    }

    def apply_mask(label_map: np.ndarray):
        nonlocal overlay
        for cls, color in CLASS_COLORS.items():
            if cls not in classes: 
                continue 
            mask = label_map == cls
            overlay[mask] = (
                (1 - alpha) * overlay[mask] + alpha * color
            ).astype(np.uint8)

    # -------------------------
    # Ground truth overlay
    # -------------------------
    if gt_mask is not None and "gt" in modes:
        apply_mask(gt_mask)

    # -------------------------
    # Model predictions
    # -------------------------
    for model_name, label_map in pred_masks.items():
        if label_map is not None and model_name in modes:
            apply_mask(label_map)

    return overlay

