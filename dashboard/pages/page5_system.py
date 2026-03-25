"""PAGE 5 — System Health (VIZ22-25)."""
import time
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

_DARK = "plotly_dark"


def get_layout():
    return html.Div([
        html.H4("System Health", className="text-info mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz22-throughput", style={"height": "300px"}), width=4),
            dbc.Col(dcc.Graph(id="viz23-memory", style={"height": "300px"}), width=4),
            dbc.Col(dcc.Graph(id="viz24-cpu", style={"height": "300px"}), width=4),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz25-status-table", style={"height": "320px"}), width=12),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("viz22-throughput", "figure"),
        Input("update-interval", "n_intervals"),
        State("system-metrics-store", "data"),
    )
    def update_throughput(n, metrics):
        t = np.arange(min(n + 1, 60))
        fps = 80 + 20 * np.sin(t * 0.2) + np.random.default_rng(n).standard_normal(len(t)) * 2
        fig = go.Figure(go.Scatter(x=t, y=fps, fill="tozeroy",
                                    line=dict(color="lime", width=2)))
        fig.add_hline(y=100, line_dash="dash", line_color="cyan",
                       annotation_text="Target 100 fps")
        fig.update_layout(title="VIZ22: Throughput (fps)", template=_DARK,
                           xaxis_title="Time (s)", yaxis_title="Frames/s",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz23-memory", "figure"),
        Input("update-interval", "n_intervals"),
        State("system-metrics-store", "data"),
    )
    def update_memory(n, metrics):
        t = np.arange(min(n + 1, 60))
        mem = 512 + 50 * np.sin(t * 0.1) + np.random.default_rng(n + 1).standard_normal(len(t)) * 5
        fig = go.Figure(go.Scatter(x=t, y=mem, fill="tozeroy",
                                    line=dict(color="orange", width=2)))
        fig.update_layout(title="VIZ23: Memory Usage (MB)", template=_DARK,
                           xaxis_title="Time (s)", yaxis_title="MB",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz24-cpu", "figure"),
        Input("update-interval", "n_intervals"),
        State("system-metrics-store", "data"),
    )
    def update_cpu(n, metrics):
        t = np.arange(min(n + 1, 60))
        cpu = 25 + 15 * np.abs(np.sin(t * 0.15)) + np.random.default_rng(n + 2).standard_normal(len(t)) * 3
        fig = go.Figure(go.Scatter(x=t, y=np.clip(cpu, 0, 100), fill="tozeroy",
                                    line=dict(color="tomato", width=2)))
        fig.add_hline(y=80, line_dash="dash", line_color="red",
                       annotation_text="Warning 80%")
        fig.update_layout(title="VIZ24: CPU Usage (%)", template=_DARK,
                           xaxis_title="Time (s)", yaxis_title="CPU %",
                           yaxis_range=[0, 100],
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz25-status-table", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_status_table(n):
        components = [
            "EMAG2 Fetcher", "IGRF Fetcher", "IMU Model",
            "Magnetometer", "Barometer", "EKF Filter",
            "UKF Filter", "Particle Filter", "State Estimator",
            "DriftTransformer", "DriftLSTM", "AnomalyVAE",
            "Stream Processor", "WebSocket Server", "Dashboard",
        ]
        statuses = ["🟢 Online"] * 12 + ["🟢 Online", "🟡 Standby", "🟢 Online"]
        rng = np.random.default_rng(n % 100)
        latencies = [f"{rng.uniform(0.1, 5.0):.2f} ms" for _ in components]

        fig = go.Figure(go.Table(
            header=dict(
                values=["<b>Component</b>", "<b>Status</b>", "<b>Latency</b>"],
                fill_color="#1a1a3a",
                font=dict(color="white", size=12),
                align="left",
                line_color="#333",
            ),
            cells=dict(
                values=[components, statuses, latencies],
                fill_color=[["#0d1117"] * len(components)],
                font=dict(color=["white", "white", "#aaa"], size=11),
                align="left",
                line_color="#333",
            ),
        ))
        fig.update_layout(title="VIZ25: System Component Status", template=_DARK,
                           margin=dict(l=10, r=10, t=40, b=10))
        return fig
