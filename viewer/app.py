"""
viewer/app.py

Dash viewer for HDF5 ACDC volumes.
"""

from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

from config import VOLUMES_DIR
from viewer.utils import load_volume, load_trained_model, predict_mask, create_overlay


PATIENT = "patient001_frame01.h5"  # change to any file
VOLUME_PATH = VOLUMES_DIR / PATIENT

print("[VIEWER] Loading volume and model...")
volume = load_volume(VOLUME_PATH)    # shape (10, 256, 216)
model = load_trained_model()
num_slices = volume.shape[0]
print("[VIEWER] Ready.")


def fig_from_overlay(overlay):
    fig = px.imshow(overlay)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("CardioSegNet Viewer — LV Segmentation"),

    dcc.Slider(
        id="slice-slider",
        min=0,
        max=num_slices - 1,
        value=num_slices // 2,
        step=1,
        marks={0: "0", num_slices - 1: str(num_slices - 1)}
    ),

    dcc.Graph(id="slice-view", style={"height": "600px"}),
])


@app.callback(
    Output("slice-view", "figure"),
    Input("slice-slider", "value")
)
def update_slice(idx):
    slice_2d = volume[idx]
    pred = predict_mask(model, slice_2d)
    overlay = create_overlay(slice_2d, pred)
    return fig_from_overlay(overlay)


if __name__ == "__main__":
    app.run_server(debug=True)

