"""
Kalman Filter - Optimal State Estimation

Phase 6: This module implements a Kalman filter for tracking targets with
proper uncertainty quantification.

KEY CONCEPTS:

1. STATE VECTOR: [px, py, pz, vx, vy, vz] - position and velocity
   - 6-dimensional state space
   - Tracks both position and velocity

2. COVARIANCE MATRIX: 6x6 matrix P
   - Diagonal: variances of each state
   - Off-diagonal: correlations between states
   - Trace of position block = position uncertainty

3. PREDICT STEP: Project state forward in time
   - x_pred = F * x (state transition)
   - P_pred = F * P * F^T + Q (uncertainty grows)
   - Q = process noise (how much motion model is wrong)

4. UPDATE STEP: Incorporate new measurement
   - K = P * H^T * (H * P * H^T + R)^-1 (Kalman gain)
   - x_new = x_pred + K * (z - H * x_pred) (correction)
   - P_new = (I - K * H) * P_pred (uncertainty shrinks)

ADVANTAGES OVER ALPHA-BETA:
- Optimal for linear Gaussian systems
- Proper uncertainty quantification (covariance)
- Adapts gain based on measurement quality
- Handles measurement gaps gracefully
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from .vector import Vec3


@dataclass
class KalmanConfig:
    """Configuration for Kalman filter."""
    # Process noise (how much we trust the motion model)
    process_noise_pos: float = 1.0      # m² (position variance per second)
    process_noise_vel: float = 0.1      # (m/s)² (velocity variance per second)

    # Measurement noise (how much we trust measurements)
    measurement_noise_pos: float = 50.0  # m (1-sigma position noise)

    # Initial uncertainty
    initial_pos_variance: float = 100.0   # m² (initial position uncertainty)
    initial_vel_variance: float = 25.0    # (m/s)² (initial velocity uncertainty)


@dataclass
class KalmanState:
    """
    State of a Kalman filter.

    Attributes:
        x: State vector [px, py, pz, vx, vy, vz]
        P: 6x6 covariance matrix
        timestamp: Time of last update
        num_updates: Number of measurement updates
    """
    x: np.ndarray           # 6x1 state vector
    P: np.ndarray           # 6x6 covariance matrix
    timestamp: float = 0.0
    num_updates: int = 0

    def get_position(self) -> Vec3:
        """Extract position from state."""
        return Vec3(float(self.x[0]), float(self.x[1]), float(self.x[2]))

    def get_velocity(self) -> Vec3:
        """Extract velocity from state."""
        return Vec3(float(self.x[3]), float(self.x[4]), float(self.x[5]))

    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (sqrt of trace of position covariance)."""
        pos_cov = self.P[:3, :3]
        return float(np.sqrt(np.trace(pos_cov)))

    def get_velocity_uncertainty(self) -> float:
        """Get velocity uncertainty (sqrt of trace of velocity covariance)."""
        vel_cov = self.P[3:6, 3:6]
        return float(np.sqrt(np.trace(vel_cov)))

    def get_position_ellipsoid(self) -> Tuple[Vec3, Vec3, Vec3]:
        """
        Get principal axes of position uncertainty ellipsoid.

        Returns:
            Tuple of (semi_axes_lengths, axis1_direction, axis2_direction, axis3_direction)
            where semi_axes_lengths is Vec3(a, b, c) representing the 1-sigma ellipsoid
        """
        pos_cov = self.P[:3, :3]
        eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)

        # Eigenvalues are variances, take sqrt for standard deviations
        semi_axes = np.sqrt(np.maximum(eigenvalues, 0))

        return (
            Vec3(float(semi_axes[0]), float(semi_axes[1]), float(semi_axes[2])),
            Vec3(float(eigenvectors[0, 0]), float(eigenvectors[1, 0]), float(eigenvectors[2, 0])),
            Vec3(float(eigenvectors[0, 1]), float(eigenvectors[1, 1]), float(eigenvectors[2, 1])),
            Vec3(float(eigenvectors[0, 2]), float(eigenvectors[1, 2]), float(eigenvectors[2, 2])),
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "position": self.get_position().to_dict(),
            "velocity": self.get_velocity().to_dict(),
            "position_uncertainty": self.get_position_uncertainty(),
            "velocity_uncertainty": self.get_velocity_uncertainty(),
            "timestamp": self.timestamp,
            "num_updates": self.num_updates,
            "covariance_trace": float(np.trace(self.P)),
        }


class KalmanFilter:
    """
    6-state Kalman filter for position/velocity tracking.

    State: [px, py, pz, vx, vy, vz]

    Motion model: Constant velocity
        p(t+dt) = p(t) + v(t) * dt
        v(t+dt) = v(t)

    Measurement model: Direct position observation
        z = [px, py, pz] + noise
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        self.config = config or KalmanConfig()

        # State transition matrix (constant velocity model)
        # Will be updated each predict step based on dt
        self._F = np.eye(6)

        # Measurement matrix (we observe position)
        self._H = np.zeros((3, 6))
        self._H[0, 0] = 1.0  # observe px
        self._H[1, 1] = 1.0  # observe py
        self._H[2, 2] = 1.0  # observe pz

        # Measurement noise matrix R (3x3)
        r = self.config.measurement_noise_pos ** 2
        self._R = np.diag([r, r, r])

    def initialize(
        self,
        position: Vec3,
        velocity: Optional[Vec3] = None,
        timestamp: float = 0.0
    ) -> KalmanState:
        """
        Initialize a new Kalman filter state.

        Args:
            position: Initial position
            velocity: Initial velocity (optional, defaults to zero)
            timestamp: Initial timestamp

        Returns:
            New KalmanState
        """
        # Initial state vector
        x = np.zeros(6)
        x[0] = position.x
        x[1] = position.y
        x[2] = position.z
        if velocity:
            x[3] = velocity.x
            x[4] = velocity.y
            x[5] = velocity.z

        # Initial covariance matrix
        P = np.diag([
            self.config.initial_pos_variance,
            self.config.initial_pos_variance,
            self.config.initial_pos_variance,
            self.config.initial_vel_variance,
            self.config.initial_vel_variance,
            self.config.initial_vel_variance,
        ])

        return KalmanState(x=x, P=P, timestamp=timestamp, num_updates=0)

    def predict(self, state: KalmanState, dt: float) -> KalmanState:
        """
        Predict state forward in time.

        Args:
            state: Current state
            dt: Time step (seconds)

        Returns:
            Predicted state (uncertainty grows)
        """
        if dt <= 0:
            return state

        # Update state transition matrix for this dt
        F = np.eye(6)
        F[0, 3] = dt  # px += vx * dt
        F[1, 4] = dt  # py += vy * dt
        F[2, 5] = dt  # pz += vz * dt

        # Process noise matrix Q
        # Using discrete white noise model
        q_pos = self.config.process_noise_pos * dt
        q_vel = self.config.process_noise_vel * dt
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        # Predict state: x_pred = F * x
        x_pred = F @ state.x

        # Predict covariance: P_pred = F * P * F^T + Q
        P_pred = F @ state.P @ F.T + Q

        return KalmanState(
            x=x_pred,
            P=P_pred,
            timestamp=state.timestamp + dt,
            num_updates=state.num_updates
        )

    def update(
        self,
        state: KalmanState,
        measurement: Vec3,
        timestamp: float,
        measurement_noise: Optional[float] = None
    ) -> KalmanState:
        """
        Update state with new measurement.

        Args:
            state: Current (predicted) state
            measurement: Position measurement
            timestamp: Measurement timestamp
            measurement_noise: Optional override for measurement noise

        Returns:
            Updated state (uncertainty shrinks)
        """
        # Measurement vector
        z = np.array([measurement.x, measurement.y, measurement.z])

        # Measurement noise matrix
        if measurement_noise is not None:
            R = np.diag([measurement_noise**2] * 3)
        else:
            R = self._R

        # Innovation (measurement residual)
        y = z - self._H @ state.x

        # Innovation covariance
        S = self._H @ state.P @ self._H.T + R

        # Kalman gain
        K = state.P @ self._H.T @ np.linalg.inv(S)

        # Update state
        x_new = state.x + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ self._H
        P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        return KalmanState(
            x=x_new,
            P=P_new,
            timestamp=timestamp,
            num_updates=state.num_updates + 1
        )

    def predict_and_update(
        self,
        state: KalmanState,
        measurement: Vec3,
        timestamp: float,
        measurement_noise: Optional[float] = None
    ) -> KalmanState:
        """
        Combined predict and update step.

        Args:
            state: Current state
            measurement: New position measurement
            timestamp: Measurement timestamp
            measurement_noise: Optional override for measurement noise

        Returns:
            Updated state
        """
        dt = timestamp - state.timestamp
        predicted = self.predict(state, dt)
        return self.update(predicted, measurement, timestamp, measurement_noise)


def create_kalman_from_detection(
    position: Vec3,
    velocity: Optional[Vec3] = None,
    timestamp: float = 0.0,
    config: Optional[KalmanConfig] = None
) -> Tuple[KalmanFilter, KalmanState]:
    """
    Convenience function to create a Kalman filter from an initial detection.

    Args:
        position: Initial detected position
        velocity: Initial velocity estimate (optional)
        timestamp: Detection timestamp
        config: Kalman filter configuration

    Returns:
        Tuple of (KalmanFilter, initial KalmanState)
    """
    kf = KalmanFilter(config)
    state = kf.initialize(position, velocity, timestamp)
    return kf, state
