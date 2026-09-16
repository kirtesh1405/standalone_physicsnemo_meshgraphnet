"""Single-page viewer for saved PhysicsNeMo MeshGraphNet predictions."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from .pipeline import Config


def _empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 16})
    figure.update_layout(template="plotly_white", height=460, margin={"l": 20, "r": 20, "t": 35, "b": 20})
    return figure


def _field(values: np.ndarray, name: str) -> np.ndarray:
    return np.maximum(np.expm1(values[:, 3]), 0.0) if name == "stress" else np.linalg.norm(values[:, :3], axis=1)


def _label(name: str) -> str:
    return "Stress (source unit)" if name == "stress" else "Displacement magnitude (source length unit)"


def _mesh_figure(vertices: np.ndarray, faces: np.ndarray, values: np.ndarray, title: str, maximum: float, colorscale: str) -> go.Figure:
    figure = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=values, intensitymode="vertex", colorscale=colorscale, cmin=0, cmax=maximum, flatshading=False, colorbar={"title": "Field"}, hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>z=%{z:.4g}<br>value=%{intensity:.5g}<extra></extra>"))
    peak = vertices[int(np.argmax(values))]
    figure.add_scatter3d(x=[peak[0]], y=[peak[1]], z=[peak[2]], mode="markers", marker={"color": "black", "size": 4, "symbol": "diamond"}, name="Peak", hovertemplate="Peak<extra></extra>")
    figure.update_layout(template="plotly_white", title=title, height=470, margin={"l": 0, "r": 0, "t": 42, "b": 0}, scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"}, showlegend=False)
    return figure


def _scatter(actual: np.ndarray, predicted: np.ndarray, quantity: str) -> go.Figure:
    rng = np.random.default_rng(42); count = min(15_000, len(actual)); indices = rng.choice(len(actual), count, replace=False)
    x, y = actual[indices], predicted[indices]; maximum = max(float(x.max()), float(y.max()), np.finfo(np.float32).eps)
    residual = y - x; r2 = 1 - float(np.sum(residual**2)) / max(float(np.sum((x - x.mean())**2)), np.finfo(np.float32).eps)
    figure = go.Figure()
    figure.add_scattergl(x=x, y=y, mode="markers", marker={"size": 3, "opacity": 0.25, "color": "#76b900"}, name="Vertices")
    figure.add_scatter(x=[0, maximum], y=[0, maximum], mode="lines", line={"color": "#111827", "dash": "dash"}, name="Perfect prediction")
    figure.update_layout(template="plotly_white", title=f"Actual vs predicted | R²={r2:.3f}", height=420, margin={"l": 62, "r": 20, "t": 45, "b": 55}, xaxis_title=f"Actual {_label(quantity)}", yaxis_title=f"Predicted {_label(quantity)}", yaxis={"scaleanchor": "x", "scaleratio": 1}, legend={"orientation": "h", "y": -0.2})
    return figure


def _history(config: Config) -> go.Figure:
    path = config.output_root / "models" / "history.csv"
    if not path.is_file(): return _empty("Training history will appear here after training.")
    history = pd.read_csv(path); figure = go.Figure()
    figure.add_scatter(x=history.epoch, y=history.train_loss, mode="lines+markers", name="Train", line={"color": "#176b9b"})
    figure.add_scatter(x=history.epoch, y=history.validation_loss, mode="lines+markers", name="Validation", line={"color": "#e07a1f"})
    figure.update_layout(template="plotly_white", title="Weighted normalized training loss", height=330, margin={"l": 62, "r": 20, "t": 45, "b": 50}, xaxis_title="Epoch", yaxis_title="Loss", legend={"orientation": "h", "y": -0.25})
    return figure


def create_app(config_path: str | Path) -> Dash:
    config = Config.load(config_path); prediction_root = config.output_root / "predictions" / "test"
    pairs = sorted((path.stem.rsplit("_", 1)[0], path.stem.rsplit("_", 1)[1]) for path in prediction_root.glob("*.npz")) if prediction_root.is_dir() else []
    case_ids = sorted({int(case_id) for case_id, _ in pairs}); default_case = case_ids[0] if case_ids else None
    loads = sorted({load for _, load in pairs}); default_load = loads[0] if loads else None
    app = Dash(__name__, title="PhysicsNeMo MeshGraphNet Results")
    app.layout = html.Main([
        html.Header([html.Div([html.P("PORTABLE EXPERIMENT", style={"letterSpacing": "0.12em", "color": "#76b900", "fontWeight": 700}), html.H1("PhysicsNeMo MeshGraphNet — surface-field results"), html.P("Review saved final predictions. Actual and predicted surfaces use the same physical colour scale; black diamonds mark each maximum.")]), html.Div([html.Strong("RTX 3050 profile"), html.Span("2,048 faces · 48 hidden width · 6 blocks · 50 epochs")], className="facts")], className="header"),
        html.Section([html.Label("Test design"), dcc.Dropdown(id="case", options=[{"label": f"Design {case_id}", "value": case_id} for case_id in case_ids], value=default_case, clearable=False), html.Label("Load case"), dcc.Dropdown(id="load", options=[{"label": {"ver":"Vertical", "hor":"Horizontal", "dia":"Diagonal", "tor":"Torsion"}.get(value, value), "value": value} for value in loads], value=default_load, clearable=False), html.Label("Field"), dcc.RadioItems(id="quantity", options=[{"label": "Stress", "value": "stress"}, {"label": "Displacement magnitude", "value": "displacement"}], value="stress", inline=True)], className="controls"),
        html.Div(id="notice", className="notice"),
        html.Section([dcc.Loading(html.Div([html.Div(dcc.Graph(id="actual", config={"displaylogo": False}), className="panel"), html.Div(dcc.Graph(id="predicted", config={"displaylogo": False}), className="panel"), html.Div(dcc.Graph(id="error", config={"displaylogo": False}), className="panel")], className="mesh-grid"))]),
        html.Section([html.Div(id="metrics", className="metrics"), html.Div(dcc.Graph(id="scatter", config={"displaylogo": False}), className="panel")], className="lower-grid"),
        html.Section(dcc.Graph(figure=_history(config), config={"displaylogo": False}), className="panel"),
    ], className="page")

    @lru_cache(maxsize=32)
    def prediction(case_id: int, load: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(prediction_root / f"{case_id}_{load}.npz", allow_pickle=False) as source:
            return (np.asarray(source["vertices"], dtype=np.float32), np.asarray(source["faces"], dtype=np.int32), np.asarray(source["actual"], dtype=np.float32), np.asarray(source["predicted"], dtype=np.float32))

    @app.callback(Output("actual", "figure"), Output("predicted", "figure"), Output("error", "figure"), Output("scatter", "figure"), Output("metrics", "children"), Output("notice", "children"), Input("case", "value"), Input("load", "value"), Input("quantity", "value"))
    def update(case_id, load, quantity):
        if case_id is None or load is None:
            figure = _empty("Run preparation, training, and final test evaluation first.")
            return figure, figure, figure, figure, "", "No saved final-test predictions were found."
        try:
            vertices, faces, raw_actual, raw_predicted = prediction(int(case_id), str(load))
        except FileNotFoundError:
            figure = _empty("The selected prediction is unavailable.")
            return figure, figure, figure, figure, "", "Saved prediction is unavailable."
        actual, predicted = _field(raw_actual, quantity), _field(raw_predicted, quantity); error = np.abs(predicted - actual)
        maximum = max(float(np.quantile(np.concatenate((actual, predicted)), 0.995)), np.finfo(np.float32).eps); error_max = max(float(np.quantile(error, 0.995)), np.finfo(np.float32).eps)
        metrics_path = config.output_root / "metrics" / "meshgraphnet_test.csv"; text = "Metrics file not found."
        if metrics_path.is_file():
            frame = pd.read_csv(metrics_path); row = frame.loc[(frame.case_id == int(case_id)) & (frame.load_case == str(load))]
            if not row.empty:
                item = row.iloc[0]; prefix = "stress" if quantity == "stress" else "displacement"
                text = [html.Div([html.Strong("MAE"), html.Span(f"{item[f'{prefix}_mae']:.5g}")]), html.Div([html.Strong("Relative L2"), html.Span(f"{item[f'{prefix}_relative_l2']:.3f}")]), html.Div([html.Strong("Top 5% overlap"), html.Span(f"{item[f'{prefix}_top_5_overlap']:.3f}")]), html.Div([html.Strong("Inference"), html.Span(f"{item['inference_seconds']:.4f} s")])]
        return _mesh_figure(vertices, faces, actual, f"Actual {_label(quantity)}", maximum, "Viridis"), _mesh_figure(vertices, faces, predicted, f"Predicted {_label(quantity)}", maximum, "Viridis"), _mesh_figure(vertices, faces, error, f"Absolute {_label(quantity)} error", error_max, "Magma"), _scatter(actual, predicted, quantity), text, f"Test design {case_id} · {str(load).upper()} · saved model output reconstructed on the original OBJ surface."
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="configs/experiment.yaml"); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8054)
    args = parser.parse_args(argv); create_app(args.config).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__": main()
