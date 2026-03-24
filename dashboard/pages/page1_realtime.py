"""PAGE 1 — Real-Time Navigation (VIZ01-07)."""
import time
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

_DARK = "plotly_dark"


def get_layout():
    return html.Div([
        html.H4("Real-Time Navigation", className="text-info mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz01-trajectory", style={"height": "300px"}), width=6),
            dbc.Col(dcc.Graph(id="viz02-mag-heatmap", style={"height": "300px"}), width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz03-filter-comparison", style={"height": "300px"}), width=6),
            dbc.Col(dcc.Graph(id="viz04-drift-curve", style={"height": "300px"}), width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz05-velocity-vectors", style={"height": "300px"}), width=4),
            dbc.Col(dcc.Graph(id="viz06-orientation", style={"height": "300px"}), width=4),
            dbc.Col(dcc.Graph(id="viz07-latency-gauge", style={"height": "300px"}), width=4),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("viz01-trajectory", "figure"),
        Input("update-interval", "n_intervals"),
        State("history-store", "data"),
    )
    def update_trajectory(n, history):
        t = np.linspace(0, 4 * np.pi, max(n + 1, 10))
        x = np.cumsum(np.cos(t) * 0.5)
        y = np.cumsum(np.sin(t) * 0.5)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                  name="Trajectory",
                                  marker=dict(size=4, color="cyan"),
                                  line=dict(width=2, color="cyan")))
        fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers",
                                  marker=dict(size=10, color="red", symbol="star"),
                                  name="Current"))
        fig.update_layout(title="VIZ01: Trajectory", template=_DARK,
                           xaxis_title="X (m)", yaxis_title="Y (m)",
                           margin=dict(l=40, r=10, t=40, b=40), uirevision="traj")
        return fig

    @app.callback(
        Output("viz02-mag-heatmap", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_mag_heatmap(n):
        lat = np.linspace(38, 42, 40)
        lon = np.linspace(-76, -72, 40)
        lon_m, lat_m = np.meshgrid(lon, lat)
        Bz = -30000 * np.sin(np.deg2rad(lat_m)) + 200 * np.sin(lon_m + n * 0.05)
        fig = go.Figure(go.Heatmap(z=Bz, x=lon, y=lat,
                                    colorscale="RdBu_r", showscale=True,
                                    colorbar=dict(title="Bz (nT)")))
        fig.update_layout(title="VIZ02: Magnetic Anomaly Map", template=_DARK,
                           xaxis_title="Longitude", yaxis_title="Latitude",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz03-filter-comparison", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_filter_comparison(n):
        t = np.arange(min(n + 1, 100))
        noise = np.random.default_rng(n).standard_normal(len(t)) * 0.5
        ekf = np.cumsum(np.cos(t * 0.1)) + noise * 0.3
        ukf = np.cumsum(np.cos(t * 0.1)) + noise * 0.25
        pf = np.cumsum(np.cos(t * 0.1)) + noise * 0.4
        fig = go.Figure()
        for vals, name, color in [(ekf, "EKF", "cyan"), (ukf, "UKF", "lime"), (pf, "PF", "orange")]:
            fig.add_trace(go.Scatter(x=t, y=vals, name=name,
                                      line=dict(width=2, color=color)))
        fig.update_layout(title="VIZ03: EKF vs UKF vs PF", template=_DARK,
                           xaxis_title="Step", yaxis_title="X Position (m)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz04-drift-curve", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_drift(n):
        t = np.arange(min(n + 1, 200))
        drift = np.abs(np.cumsum(np.random.default_rng(42).standard_normal(len(t)) * 0.02))
        fig = go.Figure(go.Scatter(x=t, y=drift, fill="tozeroy",
                                    line=dict(color="tomato", width=2)))
        fig.update_layout(title="VIZ04: Drift Over Time", template=_DARK,
                           xaxis_title="Step", yaxis_title="Drift (m)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz05-velocity-vectors", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_velocity(n):
        rng = np.random.default_rng(n % 20)
        x = rng.standard_normal(8) * 5
        y = rng.standard_normal(8) * 5
        u = rng.standard_normal(8)
        v = rng.standard_normal(8)
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="markers+text",
            marker=dict(color="lime", size=8),
        ))
        for xi, yi, ui, vi in zip(x, y, u, v):
            fig.add_annotation(x=xi + ui, y=yi + vi, ax=xi, ay=yi,
                                arrowhead=2, arrowwidth=2, arrowcolor="lime")
        fig.update_layout(title="VIZ05: Velocity Vectors", template=_DARK,
                           xaxis_title="X", yaxis_title="Y",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz06-orientation", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_orientation(n):
        t = np.arange(min(n + 1, 100))
        roll = np.sin(t * 0.1) * 10
        pitch = np.cos(t * 0.08) * 5
        yaw = (t * 0.5) % 360
        fig = go.Figure()
        for vals, name, color in [(roll, "Roll", "cyan"), (pitch, "Pitch", "lime"), (yaw, "Yaw", "orange")]:
            fig.add_trace(go.Scatter(x=t, y=vals, name=name,
                                      line=dict(width=2, color=color)))
        fig.update_layout(title="VIZ06: Orientation (RPY)", template=_DARK,
                           xaxis_title="Step", yaxis_title="Degrees",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz07-latency-gauge", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_latency(n):
        latency = 20 + 10 * np.sin(n * 0.1) + np.random.default_rng(n).standard_normal() * 3
        latency = float(np.clip(latency, 1, 200))
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latency,
            delta={"reference": 100, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 200]},
                "bar": {"color": "cyan"},
                "steps": [
                    {"range": [0, 100], "color": "#1a3a1a"},
                    {"range": [100, 150], "color": "#3a3a1a"},
                    {"range": [150, 200], "color": "#3a1a1a"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 100},
            },
            title={"text": "Latency (ms)"},
            number={"suffix": " ms", "valueformat": ".1f"},
        ))
        fig.update_layout(title="VIZ07: Processing Latency", template=_DARK,
                           margin=dict(l=40, r=40, t=60, b=20))
        return fig
