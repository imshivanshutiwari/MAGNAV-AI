"""Main navigation pipeline: integrates all components."""
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class NavigationState:
    """Complete navigation state at one timestep."""
    timestamp: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drift_correction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ekf_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ukf_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pf_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fused_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    magnetic_residual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    anomaly_score: float = 0.0
    latency_ms: float = 0.0
    filter_weights: np.ndarray = field(default_factory=lambda: np.ones(3) / 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "orientation": self.orientation.tolist(),
            "drift_correction": self.drift_correction.tolist(),
            "ekf_position": self.ekf_position.tolist(),
            "ukf_position": self.ukf_position.tolist(),
            "pf_position": self.pf_position.tolist(),
            "fused_position": self.fused_position.tolist(),
            "magnetic_residual": self.magnetic_residual.tolist(),
            "anomaly_score": float(self.anomaly_score),
            "latency_ms": float(self.latency_ms),
            "filter_weights": self.filter_weights.tolist(),
        }


class NavigationPipeline:
    """Full 10-step magnetic navigation pipeline.

    Steps per frame:
        1. IMU propagation
        2. Magnetometer measurement
        3. Map interpolation (Bx, By, Bz)
        4. Residual = measured - expected
        5. EKF update
        6. UKF update
        7. Particle Filter update
        8. Fusion of estimates
        9. ML drift correction (Transformer + LSTM)
       10. Final state output
    """

    SEQ_LEN = 50  # residual history length for ML models

    def __init__(self, config_path: str = "configs/config.yaml"):
        self._cfg = self._load_config(config_path)
        self._history: List[NavigationState] = []
        self._residual_buffer: List[np.ndarray] = []
        self._initialized = False

        # Components (set up in setup())
        self._imu = None
        self._mag = None
        self._baro = None
        self._emag2 = None
        self._igrf = None
        self._ekf = None
        self._ukf = None
        self._pf = None
        self._state_estimator = None
        self._transformer = None
        self._lstm = None
        self._vae = None
        self._lat = 40.0
        self._lon = -74.0
        self._alt = 0.0

    @staticmethod
    def _load_config(path: str) -> dict:
        """Load YAML config; return defaults if file not found."""
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def setup(self) -> None:
        """Initialize all components."""
        from sensors.imu_model import IMUModel
        from sensors.magnetometer_model import MagnetometerModel
        from sensors.barometer_model import BarometerModel
        from data.fetchers.emag2_fetcher import EMAG2Fetcher
        from data.fetchers.igrf_fetcher import IGRFFetcher
        from fusion.ekf import ExtendedKalmanFilter
        from fusion.ukf import UnscentedKalmanFilter
        from fusion.particle_filter import ParticleFilter
        from fusion.state_estimator import StateEstimator
        from models.drift_transformer import DriftTransformer
        from models.drift_lstm import DriftLSTM
        from models.anomaly_vae import AnomalyVAE

        cfg = self._cfg

        # Sensors
        sc = cfg.get("sensors", {})
        self._imu = IMUModel(**self._sensor_kwargs(sc.get("imu", {})))
        self._mag = MagnetometerModel(**self._mag_kwargs(sc.get("magnetometer", {})))
        self._baro = BarometerModel(**self._baro_kwargs(sc.get("barometer", {})))

        # Data fetchers
        dc = cfg.get("data", {})
        self._emag2 = EMAG2Fetcher(cache_dir=dc.get("emag2", {}).get("cache_dir", "data/cache/emag2"))
        self._emag2.load_grid()
        self._igrf = IGRFFetcher()
        self._igrf.load()

        # Reference position
        pc = cfg.get("pipeline", {})
        self._lat = float(pc.get("initial_lat", 40.0))
        self._lon = float(pc.get("initial_lon", -74.0))
        self._alt = float(pc.get("initial_alt", 0.0))

        # Filters
        fc = cfg.get("filters", {})
        self._ekf = ExtendedKalmanFilter(**self._ekf_kwargs(fc.get("ekf", {})))
        self._ukf = UnscentedKalmanFilter(**self._ukf_kwargs(fc.get("ukf", {})))
        pf_cfg = fc.get("particle_filter", {})
        self._pf = ParticleFilter(
            n_particles=int(pf_cfg.get("n_particles", 500)),
            state_dim=int(pf_cfg.get("state_dim", 9)),
            process_noise_pos=float(pf_cfg.get("process_noise_pos", 0.1)),
            process_noise_vel=float(pf_cfg.get("process_noise_vel", 0.5)),
            process_noise_att=float(pf_cfg.get("process_noise_att", 0.01)),
            measurement_noise=float(pf_cfg.get("measurement_noise", 25.0)),
        )
        self._pf.initialize_particles(np.zeros(9), np.eye(9) * 0.1)

        self._state_estimator = StateEstimator(self._ekf, self._ukf, self._pf)

        # ML models
        mc = cfg.get("models", {})
        tc = mc.get("drift_transformer", {})
        self._transformer = DriftTransformer(
            input_dim=int(tc.get("input_dim", 3)),
            d_model=int(tc.get("d_model", 256)),
            nhead=int(tc.get("nhead", 8)),
            num_encoder_layers=int(tc.get("num_encoder_layers", 6)),
            dim_feedforward=int(tc.get("dim_feedforward", 1024)),
            dropout=float(tc.get("dropout", 0.1)),
            output_dim=int(tc.get("output_dim", 3)),
        )
        lc = mc.get("drift_lstm", {})
        self._lstm = DriftLSTM(
            input_dim=int(lc.get("input_dim", 3)),
            hidden_dim=int(lc.get("hidden_dim", 256)),
            num_layers=int(lc.get("num_layers", 3)),
            output_dim=int(lc.get("output_dim", 3)),
        )
        vc = mc.get("anomaly_vae", {})
        self._vae = AnomalyVAE(
            input_dim=int(vc.get("input_dim", 3)),
            latent_dim=int(vc.get("latent_dim", 32)),
        )

        self._initialized = True
        logger.info("NavigationPipeline initialized.")

    # -----------------------------------------------------------------------
    # Config helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _sensor_kwargs(cfg: dict) -> dict:
        keys = ["accel_noise_std", "gyro_noise_std", "accel_bias_std",
                "gyro_bias_std", "update_rate_hz"]
        return {k: float(cfg[k]) for k in keys if k in cfg}

    @staticmethod
    def _mag_kwargs(cfg: dict) -> dict:
        out = {}
        if "noise_std" in cfg:
            out["noise_std"] = float(cfg["noise_std"])
        if "hard_iron" in cfg:
            out["hard_iron"] = cfg["hard_iron"]
        if "soft_iron" in cfg:
            out["soft_iron"] = cfg["soft_iron"]
        return out

    @staticmethod
    def _baro_kwargs(cfg: dict) -> dict:
        out = {}
        for k in ["noise_std", "bias", "update_rate_hz"]:
            if k in cfg:
                out[k] = float(cfg[k])
        return out

    @staticmethod
    def _ekf_kwargs(cfg: dict) -> dict:
        out = {}
        for k in ["state_dim", "obs_dim"]:
            if k in cfg:
                out[k] = int(cfg[k])
        for k in ["process_noise_pos", "process_noise_vel", "process_noise_att",
                  "process_noise_bias", "measurement_noise"]:
            if k in cfg:
                out[k] = float(cfg[k])
        return out

    @staticmethod
    def _ukf_kwargs(cfg: dict) -> dict:
        out = {}
        for k in ["state_dim", "obs_dim"]:
            if k in cfg:
                out[k] = int(cfg[k])
        for k in ["alpha", "beta", "kappa", "process_noise_pos", "process_noise_vel",
                  "process_noise_att", "process_noise_bias", "measurement_noise"]:
            if k in cfg:
                out[k] = float(cfg[k])
        return out

    # -----------------------------------------------------------------------
    # Core pipeline step
    # -----------------------------------------------------------------------
    def step(self, imu_accel: np.ndarray, imu_gyro: np.ndarray,
             mag_measurement: np.ndarray, baro_alt: float,
             timestamp: float = None, dt: float = 0.01) -> NavigationState:
        """Execute one pipeline step.

        Args:
            imu_accel: (3,) accelerometer reading (m/s^2)
            imu_gyro:  (3,) gyroscope reading (rad/s)
            mag_measurement: (3,) magnetometer reading (nT)
            baro_alt: Barometric altitude (m)
            timestamp: Unix timestamp
            dt: Time step (s)

        Returns:
            NavigationState
        """
        if not self._initialized:
            self.setup()

        t0 = time.perf_counter()
        if timestamp is None:
            timestamp = time.time()

        # 1. IMU propagation
        imu_state = self._imu.propagate_state(
            np.asarray(imu_accel), np.asarray(imu_gyro), dt
        )
        euler = self._imu.get_euler_angles()

        # 2. Magnetometer measurement (already provided)
        B_measured = np.asarray(mag_measurement, dtype=float)

        # 3. Map interpolation: update lat/lon from position estimate
        pos = imu_state[:3]
        m_per_deg = 111320.0
        self._lat += pos[1] / m_per_deg * dt
        self._lon += pos[0] / m_per_deg * dt
        self._alt = max(0.0, baro_alt)

        Bx_exp, By_exp, Bz_exp = self._emag2.interpolate(self._lat, self._lon, self._alt / 1000.0)
        B_expected = np.array([Bx_exp, By_exp, Bz_exp])

        # 4. Residual
        residual = B_measured - B_expected

        # 5. EKF update
        H = np.zeros((3, 15))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        self._ekf.predict(dt, imu_accel, imu_gyro)
        self._ekf.update(B_measured, H)

        # 6. UKF update
        def mag_obs_fn(state):
            return state[:3]
        self._ukf.predict(dt, imu_accel, imu_gyro)
        self._ukf.update(B_measured, obs_fn=mag_obs_fn)

        # 7. Particle filter update
        def pf_mag_fn(particle):
            return particle[:3]
        self._pf.predict(dt, imu_accel, imu_gyro)
        self._pf.update(B_measured, measurement_fn=pf_mag_fn)

        # 8. Fusion
        fused_state, fused_cov = self._state_estimator.step()
        _fw = self._state_estimator.get_filter_weights()
        filter_weights = _fw if _fw is not None else np.ones(3) / 3

        # 9. ML drift correction
        self._residual_buffer.append(residual.copy())
        if len(self._residual_buffer) > self.SEQ_LEN:
            self._residual_buffer = self._residual_buffer[-self.SEQ_LEN:]

        drift_correction = np.zeros(3)
        if len(self._residual_buffer) >= 10:
            seq = np.array(self._residual_buffer[-self.SEQ_LEN:], dtype=np.float32)
            if len(seq) < self.SEQ_LEN:
                pad = np.zeros((self.SEQ_LEN - len(seq), 3), dtype=np.float32)
                seq = np.vstack([pad, seq])
            try:
                t_corr = self._transformer.predict(seq)
                l_corr = self._lstm.predict(seq)
                drift_correction = 0.5 * t_corr + 0.5 * l_corr
            except Exception:
                pass

        # Anomaly detection
        anomaly_is, anomaly_score = self._vae.detect_anomaly(residual)

        # 10. Final state
        ekf_s = self._ekf.get_state()
        ukf_s = self._ukf.get_state()
        pf_s = self._pf.get_estimate()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        nav_state = NavigationState(
            timestamp=timestamp,
            position=fused_state[:3].copy(),
            velocity=fused_state[3:6].copy(),
            orientation=euler.copy(),
            drift_correction=drift_correction,
            ekf_position=ekf_s[:3].copy(),
            ukf_position=ukf_s[:3].copy(),
            pf_position=pf_s[:3].copy(),
            fused_position=fused_state[:3].copy(),
            magnetic_residual=residual,
            anomaly_score=anomaly_score,
            latency_ms=latency_ms,
            filter_weights=filter_weights,
        )
        self._history.append(nav_state)
        return nav_state

    def run_simulation(self, n_steps: int = 1000, dt: float = 0.01) -> List[NavigationState]:
        """Run a synthetic simulation for testing.

        Args:
            n_steps: Number of time steps
            dt: Time step in seconds

        Returns:
            List of NavigationState
        """
        if not self._initialized:
            self.setup()

        logger.info(f"Running {n_steps}-step simulation (dt={dt}s)...")
        rng = np.random.default_rng(42)
        t = 0.0
        for i in range(n_steps):
            # Synthetic trajectory: circular motion
            angle = 2 * np.pi * i / n_steps
            accel = np.array([
                np.cos(angle) * 0.1,
                np.sin(angle) * 0.1,
                9.81,
            ])
            gyro = np.array([0.0, 0.0, 2 * np.pi / (n_steps * dt)])
            mag = self._emag2.interpolate(self._lat, self._lon) + rng.standard_normal(3) * 10.0
            baro = 100.0 + rng.standard_normal() * 0.5
            self.step(accel, gyro, mag, baro, timestamp=t, dt=dt)
            t += dt

        logger.info(f"Simulation complete. {len(self._history)} states recorded.")
        return self._history

    def get_history(self) -> List[NavigationState]:
        """Return all recorded navigation states."""
        return self._history

    def get_trajectory_array(self) -> np.ndarray:
        """Return (N, 3) array of fused positions."""
        if not self._history:
            return np.zeros((0, 3))
        return np.array([s.fused_position for s in self._history])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipeline = NavigationPipeline("configs/config.yaml")
    pipeline.setup()
    states = pipeline.run_simulation(n_steps=200, dt=0.01)
    final = states[-1]
    print(f"\nSimulation complete. {len(states)} steps.")
    print(f"Final position: {final.fused_position}")
    print(f"Final latency: {final.latency_ms:.2f} ms")
    traj = pipeline.get_trajectory_array()
    print(f"Trajectory shape: {traj.shape}")
