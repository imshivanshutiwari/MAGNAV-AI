"""PAGE 2 — Sensor Analysis (VIZ08-13)."""
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

_DARK = "plotly_dark"


def get_layout():
    return html.Div([
        html.H4("Sensor Analysis", className="text-info mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz08-imu-psd", style={"height": "300px"}), width=6),
            dbc.Col(dcc.Graph(id="viz09-mag-timeline", style={"height": "300px"}), width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz10-gradient-field", style={"height": "300px"}), width=6),
            dbc.Col(dcc.Graph(id="viz11-residuals", style={"height": "300px"}), width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz12-sensor-weights", style={"height": "300px"}), width=6),
            dbc.Col(dcc.Graph(id="viz13-bias-estimation", style={"height": "300px"}), width=6),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("viz08-imu-psd", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_imu_psd(n):
        rng = np.random.default_rng(n % 10)
        N = 256
        signal = rng.standard_normal(N) * 0.001 + 0.0001 * np.sin(np.linspace(0, 20 * np.pi, N))
        fft = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(N, d=1.0 / 100)
        fig = go.Figure(go.Scatter(x=freqs[1:], y=10 * np.log10(fft[1:] + 1e-20),
                                    fill="tozeroy", line=dict(color="cyan", width=1.5)))
        fig.update_layout(title="VIZ08: IMU Noise PSD", template=_DARK,
                           xaxis_title="Frequency (Hz)", yaxis_title="PSD (dB)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz09-mag-timeline", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_mag_timeline(n):
        t = np.arange(min(n + 1, 100))
        rng = np.random.default_rng(42)
        Bx = -1450 + rng.standard_normal(len(t)) * 5
        By = 4650 + rng.standard_normal(len(t)) * 5
        Bz = -29000 + rng.standard_normal(len(t)) * 8
        fig = go.Figure()
        for vals, name, color in [(Bx, "Bx", "cyan"), (By, "By", "lime"), (Bz, "Bz", "orange")]:
            fig.add_trace(go.Scatter(x=t, y=vals, name=name, line=dict(width=1.5, color=color)))
        fig.update_layout(title="VIZ09: Magnetometer Timeline", template=_DARK,
                           xaxis_title="Step", yaxis_title="nT",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz10-gradient-field", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_gradient(n):
        lat = np.linspace(39, 41, 30)
        lon = np.linspace(-75, -73, 30)
        lm, ltm = np.meshgrid(lon, lat)
        grad = np.sqrt((200 * np.cos(ltm)) ** 2 + (100 * np.sin(lm + n * 0.02)) ** 2)
        fig = go.Figure(go.Heatmap(z=grad, x=lon, y=lat,
                                    colorscale="Viridis",
                                    colorbar=dict(title="|∇B| (nT/m)")))
        fig.update_layout(title="VIZ10: Gradient Field", template=_DARK,
                           xaxis_title="Longitude", yaxis_title="Latitude",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz11-residuals", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_residuals(n):
        t = np.arange(min(n + 1, 100))
        rng = np.random.default_rng(n)
        rx = rng.standard_normal(len(t)) * 8
        ry = rng.standard_normal(len(t)) * 6
        rz = rng.standard_normal(len(t)) * 10
        fig = go.Figure()
        for vals, name, color in [(rx, "ΔBx", "cyan"), (ry, "ΔBy", "lime"), (rz, "ΔBz", "orange")]:
            fig.add_trace(go.Scatter(x=t, y=vals, name=name, line=dict(width=1.5, color=color)))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig.update_layout(title="VIZ11: Magnetic Residuals", template=_DARK,
                           xaxis_title="Step", yaxis_title="Residual (nT)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz12-sensor-weights", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_sensor_weights(n):
        rng = np.random.default_rng(n % 30)
        raw = rng.uniform(1, 5, 3)
        weights = raw / raw.sum()
        fig = go.Figure(go.Bar(
            x=["EKF", "UKF", "PF"],
            y=weights,
            marker_color=["cyan", "lime", "orange"],
            text=[f"{w:.3f}" for w in weights],
            textposition="auto",
        ))
        fig.update_layout(title="VIZ12: Filter Fusion Weights", template=_DARK,
                           yaxis_title="Weight", yaxis_range=[0, 1],
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz13-bias-estimation", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_bias(n):
        t = np.arange(min(n + 1, 100))
        rng = np.random.default_rng(42)
        bias_x = np.cumsum(rng.standard_normal(len(t)) * 0.00001)
        bias_y = np.cumsum(rng.standard_normal(len(t)) * 0.00001)
        bias_z = np.cumsum(rng.standard_normal(len(t)) * 0.00001)
        fig = go.Figure()
        for vals, name, color in [(bias_x, "bg_x", "cyan"), (bias_y, "bg_y", "lime"), (bias_z, "bg_z", "orange")]:
            fig.add_trace(go.Scatter(x=t, y=vals, name=name, line=dict(width=1.5, color=color)))
        fig.update_layout(title="VIZ13: Gyro Bias Estimation", template=_DARK,
                           xaxis_title="Step", yaxis_title="Bias (rad/s)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig
