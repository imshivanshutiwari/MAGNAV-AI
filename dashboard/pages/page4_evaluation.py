"""PAGE 4 — Evaluation (VIZ18-21)."""
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

_DARK = "plotly_dark"


def get_layout():
    return html.Div([
        html.H4("Evaluation & Performance", className="text-info mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz18-rmse-distance", style={"height": "350px"}), width=6),
            dbc.Col(dcc.Graph(id="viz19-drift-per-km", style={"height": "350px"}), width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz20-3d-trajectory", style={"height": "400px"}), width=8),
            dbc.Col(dcc.Graph(id="viz21-consistency", style={"height": "400px"}), width=4),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("viz18-rmse-distance", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_rmse(n):
        dist = np.linspace(0, 10, 50)
        rng = np.random.default_rng(42)
        ekf_rmse = 0.3 * dist + rng.standard_normal(50) * 0.1
        ukf_rmse = 0.25 * dist + rng.standard_normal(50) * 0.08
        fused_rmse = 0.15 * dist + rng.standard_normal(50) * 0.05
        fig = go.Figure()
        for vals, name, color in [(ekf_rmse, "EKF", "cyan"), (ukf_rmse, "UKF", "lime"),
                                   (fused_rmse, "Fused", "gold")]:
            fig.add_trace(go.Scatter(x=dist, y=np.maximum(vals, 0),
                                      name=name, line=dict(width=2, color=color)))
        fig.update_layout(title="VIZ18: RMSE vs Distance", template=_DARK,
                           xaxis_title="Distance (km)", yaxis_title="RMSE (m)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz19-drift-per-km", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_drift_km(n):
        rng = np.random.default_rng(n % 20)
        names = ["EKF", "UKF", "PF", "Fused"]
        base = [3.2, 2.8, 4.1, 1.9]
        noise = rng.standard_normal(4) * 0.3
        values = np.maximum(np.array(base) + noise, 0.1)
        fig = go.Figure(go.Bar(
            x=names, y=values,
            marker_color=["cyan", "lime", "orange", "gold"],
            text=[f"{v:.2f}" for v in values],
            textposition="auto",
        ))
        fig.update_layout(title="VIZ19: Drift per km", template=_DARK,
                           yaxis_title="Drift (m/km)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz20-3d-trajectory", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_3d_traj(n):
        t = np.linspace(0, 4 * np.pi, 100)
        x_gt = np.cumsum(np.cos(t) * 0.5)
        y_gt = np.cumsum(np.sin(t) * 0.5)
        z_gt = t * 0.5
        rng = np.random.default_rng(42)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x_gt, y=y_gt, z=z_gt, mode="lines",
                                    name="Ground Truth",
                                    line=dict(color="white", width=4)))
        for name, color, scale in [("EKF", "cyan", 0.3), ("UKF", "lime", 0.25),
                                     ("Fused", "gold", 0.1)]:
            noise = rng.standard_normal((100, 3)) * scale
            fig.add_trace(go.Scatter3d(
                x=x_gt + noise[:, 0], y=y_gt + noise[:, 1], z=z_gt + noise[:, 2],
                mode="lines", name=name,
                line=dict(color=color, width=2),
            ))
        fig.update_layout(title="VIZ20: 3D Trajectory Comparison", template=_DARK,
                           margin=dict(l=0, r=0, t=40, b=0),
                           scene=dict(
                               xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"
                           ))
        return fig

    @app.callback(
        Output("viz21-consistency", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_consistency(n):
        score = 0.7 + 0.2 * np.sin(n * 0.05)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(float(score), 3),
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "gold"},
                "steps": [
                    {"range": [0, 0.5], "color": "#3a1a1a"},
                    {"range": [0.5, 0.75], "color": "#3a3a1a"},
                    {"range": [0.75, 1.0], "color": "#1a3a1a"},
                ],
            },
            title={"text": "Consistency Score"},
            number={"valueformat": ".3f"},
        ))
        fig.update_layout(title="VIZ21: Filter Consistency Score", template=_DARK,
                           margin=dict(l=40, r=40, t=60, b=20))
        return fig
