"""Navigation state callbacks for dashboard."""
import time
import numpy as np
from dash import Input, Output


def register_callbacks(app):
    @app.callback(
        Output("nav-state-store", "data"),
        Input("update-interval", "n_intervals"),
    )
    def update_nav_state(n):
        """Generate a nav state for the stores."""
        t = n * 0.5
        angle = 2 * np.pi * t / 60.0
        return {
            "timestamp": time.time(),
            "position": [np.cos(angle) * 100, np.sin(angle) * 100, t * 0.1],
            "velocity": [-np.sin(angle) * 5, np.cos(angle) * 5, 0.1],
            "orientation": [
                np.sin(angle) * 5.0,
                np.cos(angle) * 3.0,
                (angle * 180 / np.pi) % 360,
            ],
            "latency_ms": 15 + 5 * np.sin(n * 0.1),
        }

    @app.callback(
        Output("history-store", "data"),
        Input("nav-state-store", "data"),
    )
    def update_history(nav_state):
        """Append latest state to history."""
        if not nav_state:
            return {"ekf": [], "ukf": [], "pf": [], "fused": []}
        pos = nav_state.get("position", [0, 0, 0])
        rng = np.random.default_rng(int(nav_state.get("timestamp", 0)) % 1000)
        return {
            "ekf": [pos[0] + rng.standard_normal() * 0.3,
                    pos[1] + rng.standard_normal() * 0.3,
                    pos[2]],
            "ukf": [pos[0] + rng.standard_normal() * 0.25,
                    pos[1] + rng.standard_normal() * 0.25,
                    pos[2]],
            "pf": [pos[0] + rng.standard_normal() * 0.4,
                   pos[1] + rng.standard_normal() * 0.4,
                   pos[2]],
            "fused": pos,
        }
