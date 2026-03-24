"""PAGE 3 — ML Models (VIZ14-17)."""
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

_DARK = "plotly_dark"


def get_layout():
    return html.Div([
        html.H4("ML Model Analysis", className="text-info mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz14-drift-pred", style={"height": "350px"}), width=6),
            dbc.Col(dcc.Graph(id="viz15-transformer-attn", style={"height": "350px"}), width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="viz16-lstm-hidden", style={"height": "350px"}), width=6),
            dbc.Col(dcc.Graph(id="viz17-anomaly-alerts", style={"height": "350px"}), width=6),
        ]),
    ])


def register_callbacks(app):
    @app.callback(
        Output("viz14-drift-pred", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_drift_pred(n):
        t = np.arange(min(n + 1, 100))
        rng = np.random.default_rng(42)
        actual = np.cumsum(rng.standard_normal(len(t)) * 0.05)
        predicted = actual + rng.standard_normal(len(t)) * 0.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=actual, name="Actual", line=dict(color="cyan", width=2)))
        fig.add_trace(go.Scatter(x=t, y=predicted, name="Predicted",
                                  line=dict(color="lime", width=2, dash="dash")))
        fig.update_layout(title="VIZ14: Drift Prediction vs Actual", template=_DARK,
                           xaxis_title="Step", yaxis_title="Drift (m)",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz15-transformer-attn", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_transformer_attn(n):
        rng = np.random.default_rng(n % 20)
        seq_len = 20
        attn = rng.dirichlet(np.ones(seq_len), size=seq_len)
        fig = go.Figure(go.Heatmap(
            z=attn,
            colorscale="Plasma",
            colorbar=dict(title="Attention"),
        ))
        fig.update_layout(title="VIZ15: Transformer Attention Map", template=_DARK,
                           xaxis_title="Key Position", yaxis_title="Query Position",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz16-lstm-hidden", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_lstm_hidden(n):
        rng = np.random.default_rng(n % 15)
        hidden = np.tanh(rng.standard_normal((20, 32)))
        fig = go.Figure(go.Heatmap(
            z=hidden,
            colorscale="RdBu",
            colorbar=dict(title="Activation"),
        ))
        fig.update_layout(title="VIZ16: LSTM Hidden States", template=_DARK,
                           xaxis_title="Hidden Unit", yaxis_title="Time Step",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig

    @app.callback(
        Output("viz17-anomaly-alerts", "figure"),
        Input("update-interval", "n_intervals"),
    )
    def update_anomaly(n):
        t = np.arange(min(n + 1, 100))
        rng = np.random.default_rng(42)
        scores = rng.exponential(scale=50, size=len(t))
        # Inject a few spikes
        for spike in [20, 45, 70]:
            if spike < len(t):
                scores[spike] = 350
        threshold = 200.0
        anomalies = scores > threshold

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=scores, name="Anomaly Score",
                                  line=dict(color="orange", width=1.5)))
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                       annotation_text="Threshold")
        if anomalies.any():
            fig.add_trace(go.Scatter(
                x=t[anomalies], y=scores[anomalies],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="Anomaly",
            ))
        fig.update_layout(title="VIZ17: Anomaly Detection Alerts", template=_DARK,
                           xaxis_title="Step", yaxis_title="Reconstruction Error",
                           margin=dict(l=40, r=10, t=40, b=40))
        return fig
