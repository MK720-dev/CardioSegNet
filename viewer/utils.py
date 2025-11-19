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
    # model_b_path = MODEL_DIR / "unet_lv_advanced_256.keras"
    # if model_b_path.exists():
    #     models["advanced_unet"] = keras.models.load_model(
    #         model_b_path,
    #         compile=False
    #     )

    return models


def preprocess_slice_for_model(slice_2d: np.ndarray) -> np.ndarray:
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
    resized = cv2.resize(slice_2d, (IMG_SIZE, IMG_SIZE))
    resized = resized.astype(np.float32)

    # Normalize to [0, 1]
    max_val = resized.max()
    if max_val > 0:
        resized /= max_val

    # Add channel + batch dimensions
    x = np.expand_dims(resized, axis=-1)   # (H, W, 1)
    x = np.expand_dims(x, axis=0)         # (1, H, W, 1)
    return x


def predict_mask(model: keras.Model,
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
    x = preprocess_slice_for_model(slice_2d)
    pred = model.predict(x, verbose=0)[0, ..., 0]   # (IMG_SIZE, IMG_SIZE)

    # Resize prediction back to original slice resolution
    pred_resized = cv2.resize(
        pred,
        (orig_shape[1], orig_shape[0])
    )
    pred_bin = (pred_resized > 0.5).astype(np.uint8)
    return pred_bin


# --------- Overlay creation --------- #

def make_overlay(
    slice_2d: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_masks: Dict[str, Optional[np.ndarray]],
    modes: List[str],
) -> np.ndarray:
    """
    Build an RGB overlay from:
      - base MRI slice
      - optional ground truth mask
      - optional 1 or more predicted masks

    Parameters
    ----------
    slice_2d : (H, W) float32
    gt_mask  : (H, W) uint8 or None
        Binary LV mask from ground truth.
    pred_masks : dict[model_id, mask]
        Each mask is (H, W) uint8 or None.
    modes : list of strings
        Selected overlays from the UI, e.g.
        ["gt", "baseline_unet"] or ["gt", "baseline_unet", "advanced_unet"]

    Color convention
    ----------------
    - Background      : grayscale MRI
    - GT only         : green
    - Baseline only   : red
    - Advanced only   : blue (Phase 2, if added)
    - Overlaps:
        GT + baseline : yellow (red + green)
        GT + advanced : cyan   (green + blue)
        baseline + advanced : magenta (red + blue)
        all three : white

    Returns
    -------
    overlay_rgb : (H, W, 3) uint8
    """
    img = slice_2d.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img_uint8 = (img * 255).astype(np.uint8)

    # Base grayscale to RGB
    overlay = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    H, W = overlay.shape[:2]

    # Prepare masks
    gt = gt_mask.astype(bool) if (gt_mask is not None and "gt" in modes) else np.zeros((H, W), dtype=bool)
    base = pred_masks.get("baseline_unet")
    adv = pred_masks.get("advanced_unet")  # optional, Phase 2

    base = base.astype(bool) if (base is not None and "baseline_unet" in modes) else np.zeros((H, W), dtype=bool)
    adv = adv.astype(bool) if (adv is not None and "advanced_unet" in modes) else np.zeros((H, W), dtype=bool)

    # Region combinations
    all_three = gt & base & adv
    gt_base = gt & base & ~adv
    gt_adv = gt & adv & ~base
    base_adv = base & adv & ~gt
    only_gt = gt & ~base & ~adv
    only_base = base & ~gt & ~adv
    only_adv = adv & ~gt & ~base

    # Apply colors
    # GT only: green
    overlay[only_gt] = [0, 255, 0]

    # Baseline only: red
    overlay[only_base] = [255, 0, 0]

    # Advanced only (Phase 2): blue
    overlay[only_adv] = [0, 0, 255]

    # Overlaps:
    overlay[gt_base] = [255, 255, 0]     # yellow
    overlay[gt_adv] = [0, 255, 255]      # cyan
    overlay[base_adv] = [255, 0, 255]    # magenta
    overlay[all_three] = [255, 255, 255] # white

    return overlay


