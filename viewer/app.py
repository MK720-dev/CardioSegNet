# viewer/app.py

"""
Interactive Dash viewer for CardioSegNet.

Features:
- Dropdown to choose patient
- Dropdown to choose frame (ED/ES etc.)
- Slider to browse slices
- Checklist to toggle:
    - Ground truth overlay
    - Baseline model prediction
    - (Future) Advanced model prediction
- Easily extendable to compare two models (baseline vs advanced)
"""

from pathlib import Path
from typing import Dict, List

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

from config import VOLUMES_DIR
from viewer.utils import (
    load_volume_and_label,
    load_models,
    predict_mask,
    make_overlay,
)

import numpy as np 
# ---------- Discover available patients & frames ---------- #

def parse_patient_frame(path: Path):
    """
    Parse filenames of the form 'patientXXX_frameYY.h5'.

    Returns
    -------
    patient_id : str, e.g. 'patient001'
    frame_id   : str, e.g. 'frame01'
    """
    stem = path.stem  # 'patient001_frame01'
    parts = stem.split("_")
    # Simple robust parse: assume "patientXXX_frameYY"
    patient_id = parts[0]        # 'patient001'
    frame_id = parts[1]          # 'frame01'
    return patient_id, frame_id


def build_patient_frame_index(vol_dir: Path) -> Dict[str, List[str]]:
    """
    Scan the volumes directory and build a mapping:
        patient_id -> list of frame_ids
    """
    mapping: Dict[str, List[str]] = {}

    for p in sorted(vol_dir.glob("patient*_frame*.h5")):
        patient_id, frame_id = parse_patient_frame(p)
        mapping.setdefault(patient_id, [])
        if frame_id not in mapping[patient_id]:
            mapping[patient_id].append(frame_id)

    # Sort frames for each patient
    for k in mapping:
        mapping[k] = sorted(mapping[k])

    return mapping


PATIENT_FRAMES = build_patient_frame_index(VOLUMES_DIR)
AVAILABLE_PATIENTS = sorted(PATIENT_FRAMES.keys())

# Load models once at startup
MODELS = load_models()  # e.g. {"baseline_unet": model}

# ---------- Dash app layout ---------- #

app = dash.Dash(__name__)
app.title = "CardioSegNet Viewer"

default_patient = AVAILABLE_PATIENTS[0] if AVAILABLE_PATIENTS else None
default_frame = PATIENT_FRAMES[default_patient][0] if default_patient else None

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H2("CardioSegNet LV Segmentation Viewer"),

        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Patient"),
                        dcc.Dropdown(
                            id="patient-dropdown",
                            options=[
                                {"label": pid, "value": pid}
                                for pid in AVAILABLE_PATIENTS
                            ],
                            value=default_patient,
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Frame"),
                        dcc.Dropdown(
                            id="frame-dropdown",
                            options=[
                                {"label": f, "value": f}
                                for f in (PATIENT_FRAMES.get(default_patient, []))
                            ],
                            value=default_frame,
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Overlays"),
                        dcc.Checklist(
                            id="overlay-checklist",
                            options=[
                                {"label": "Ground truth", "value": "gt"},
                                {"label": "Baseline model", "value": "baseline_unet"},
                                # For Phase 2:
                                # {"label": "Advanced model", "value": "advanced_unet"},
                            ],
                            value=["gt", "baseline_unet"],
                            labelStyle={"display": "block"},
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            style={"marginBottom": "10px"},
            children=[
                html.Label("Slice index"),
                dcc.Slider(
                    id="slice-slider",
                    min=0,
                    max=9,          # will be updated dynamically
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(10)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
        ),

        dcc.Graph(
            id="slice-view",
            style={"height": "650px"},
        ),
    ],
)


# ---------- Callbacks ---------- #

@app.callback(
    Output("frame-dropdown", "options"),
    Output("frame-dropdown", "value"),
    Input("patient-dropdown", "value"),
)
def update_frame_dropdown(selected_patient):
    """
    When the patient changes, update the list of frames.
    """
    if selected_patient is None:
        return [], None

    frames = PATIENT_FRAMES.get(selected_patient, [])
    options = [{"label": f, "value": f} for f in frames]
    value = frames[0] if frames else None
    return options, value


@app.callback(
    Output("slice-slider", "max"),
    Output("slice-slider", "marks"),
    Output("slice-slider", "value"),
    Input("patient-dropdown", "value"),
    Input("frame-dropdown", "value"),
)
def update_slice_slider(selected_patient, selected_frame):
    """
    Adjust slider range based on number of slices in the selected volume.
    """
    if not selected_patient or not selected_frame:
        return 0, {0: "0"}, 0

    vol_path = VOLUMES_DIR / f"{selected_patient}_{selected_frame}.h5"
    vol_img, _ = load_volume_and_label(vol_path)
    num_slices = vol_img.shape[0]

    max_idx = max(num_slices - 1, 0)
    marks = {i: str(i) for i in range(num_slices)}
    default_val = num_slices // 2

    return max_idx, marks, default_val


@app.callback(
    Output("slice-view", "figure"),
    Input("patient-dropdown", "value"),
    Input("frame-dropdown", "value"),
    Input("slice-slider", "value"),
    Input("overlay-checklist", "value"),
)
def update_slice_view(selected_patient, selected_frame, slice_idx, modes):
    """
    Main viewer callback: load volume, compute overlays, and render figure.
    """
    if not selected_patient or not selected_frame:
        return px.imshow([[0]], title="No data")

    vol_path = VOLUMES_DIR / f"{selected_patient}_{selected_frame}.h5"
    vol_img, vol_lbl = load_volume_and_label(vol_path)

    num_slices = vol_img.shape[0]
    if slice_idx is None:
        slice_idx = num_slices // 2
    slice_idx = int(max(0, min(num_slices - 1, slice_idx)))

    slice_img = vol_img[slice_idx]  # (H, W)
    slice_lbl = vol_lbl[slice_idx]  # (H, W), values 0..3

    # LV ground truth mask (class 3)
    gt_mask = (slice_lbl == 3).astype(np.uint8)

    # Predicted masks for each requested model
    pred_masks = {}
    H, W = slice_img.shape

    if "baseline_unet" in modes and "baseline_unet" in MODELS:
        pred_masks["baseline_unet"] = predict_mask(
            MODELS["baseline_unet"],
            slice_img,
            (H, W),
        )
    else:
        pred_masks["baseline_unet"] = None

    # Example for Phase 2:
    # if "advanced_unet" in modes and "advanced_unet" in MODELS:
    #     pred_masks["advanced_unet"] = predict_mask(
    #         MODELS["advanced_unet"],
    #         slice_img,
    #         (H, W),
    #     )
    # else:
    #     pred_masks["advanced_unet"] = None

    overlay_img = make_overlay(
        slice_2d=slice_img,
        gt_mask=gt_mask,
        pred_masks=pred_masks,
        modes=modes,
    )

    fig = px.imshow(overlay_img)
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{selected_patient} - {selected_frame} - slice {slice_idx}",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


if __name__ == "__main__":
    app.run(debug=True)


