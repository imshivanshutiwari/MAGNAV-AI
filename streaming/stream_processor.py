"""Streaming processor for real-time navigation pipeline."""
import time
import queue
import threading
import logging
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Processes navigation sensor frames in a dedicated thread.

    Targets <100ms per-frame latency.
    """

    def __init__(self, pipeline=None, buffer_size: int = 1000):
        self._pipeline = pipeline
        self._buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._latest_state: Optional[Dict[str, Any]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Metrics
        self._processed_count = 0
        self._latency_history = []
        self._start_time = time.time()

    def process_frame(self, sensor_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sensor frame synchronously.

        Args:
            sensor_data: dict with keys 'imu', 'mag', 'baro', 'timestamp'

        Returns:
            Navigation state dict or None on error
        """
        t_start = time.perf_counter()
        try:
            if self._pipeline is not None:
                imu = sensor_data.get("imu", np.zeros(6))
                mag = sensor_data.get("mag", np.zeros(3))
                baro = sensor_data.get("baro", 0.0)
                ts = sensor_data.get("timestamp", time.time())

                accel = np.asarray(imu[:3])
                gyro = np.asarray(imu[3:6])
                state = self._pipeline.step(accel, gyro, mag, baro, ts)
            else:
                # Simulate a navigation state when no pipeline is attached
                state = self._simulate_state(sensor_data)

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self._latency_history.append(latency_ms)
            if len(self._latency_history) > 1000:
                self._latency_history = self._latency_history[-1000:]
            self._processed_count += 1
            self._latest_state = state
            return state
        except Exception as exc:
            logger.error(f"Frame processing error: {exc}")
            return None

    @staticmethod
    def _simulate_state(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plausible navigation state when no real pipeline exists."""
        ts = sensor_data.get("timestamp", time.time())
        return {
            "timestamp": ts,
            "position": np.zeros(3).tolist(),
            "velocity": np.zeros(3).tolist(),
            "orientation": np.zeros(3).tolist(),
            "drift": 0.0,
        }

    def _worker(self):
        """Background worker thread."""
        while self._running:
            try:
                sensor_data = self._buffer.get(timeout=0.1)
                self.process_frame(sensor_data)
                self._buffer.task_done()
            except queue.Empty:
                continue

    def start(self):
        """Start the background processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("StreamProcessor started.")

    def stop(self):
        """Stop the background processing thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("StreamProcessor stopped.")

    def enqueue(self, sensor_data: Dict[str, Any]) -> bool:
        """Non-blocking enqueue of sensor data."""
        try:
            self._buffer.put_nowait(sensor_data)
            return True
        except queue.Full:
            return False

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """Return the most recently computed navigation state."""
        return self._latest_state

    def get_metrics(self) -> Dict[str, float]:
        """Return performance metrics."""
        elapsed = max(time.time() - self._start_time, 1e-9)
        latencies = self._latency_history or [0.0]
        return {
            "throughput_fps": self._processed_count / elapsed,
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_max_ms": float(np.max(latencies)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "total_frames": self._processed_count,
        }
