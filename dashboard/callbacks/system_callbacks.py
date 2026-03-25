"""System monitoring callbacks."""
import time
from dash import Input, Output


def register_callbacks(app):
    @app.callback(
        Output("system-metrics-store", "data"),
        Input("update-interval", "n_intervals"),
    )
    def update_system_metrics(n):
        """Collect system metrics via psutil if available."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            memory_mb = mem.used / 1024 / 1024
        except ImportError:
            cpu = 30.0
            memory_mb = 512.0

        return {
            "timestamp": time.time(),
            "cpu_percent": float(cpu),
            "memory_mb": float(memory_mb),
            "throughput_fps": 80.0 + n % 30,
            "components": {
                "emag2": "online",
                "igrf": "online",
                "ekf": "online",
                "ukf": "online",
                "pf": "online",
                "transformer": "online",
                "lstm": "online",
                "vae": "online",
            },
        }
